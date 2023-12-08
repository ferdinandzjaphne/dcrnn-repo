import torch
import time
import numpy as np
import os
from model.dcrnn_model import DCRNNModel
from torch.utils.tensorboard import SummaryWriter
from model.loss import masked_mae_loss, masked_mape, masked_rmse
from lib import load_dataset as ld, utils
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        # logging
        self._log_dir = self._get_log_dir(kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)

        # load dataset
        self._data = ld.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # training variables
        self.base_lr = self._train_kwargs.get('base_lr')
        self.epsilon = self._train_kwargs.get('epsilon')
        self.steps = self._train_kwargs.get('steps')
        self.lr_decay_ratio = self._train_kwargs.get('lr_decay_ratio')
        self._epoch_num = self._train_kwargs.get('epoch')
        self.epoch = self._train_kwargs.get('epochs')

        # model variables
        self.seq_len = self._model_kwargs.get('seq_len')
        self.num_nodes = self._model_kwargs.get('num_nodes')
        self.input_dim = self._model_kwargs.get('input_dim')
        self.output_dim = self._model_kwargs.get('output_dim')
        self.horizon = self._model_kwargs.get('horizon')
        self.max_grad_norm = self._model_kwargs.get('max_grad_norm', 1.)

        # dcrnn model
        dcrnn_model = DCRNNModel(adj_mx, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()
    
    @staticmethod
    def _get_log_dir(kwargs):
        batch_size = kwargs['data'].get('batch_size')
        learning_rate = kwargs['train'].get('base_lr')
        max_diffusion_step = kwargs['model'].get('max_diffusion_step')
        num_rnn_layers = kwargs['model'].get('num_rnn_layers')
        rnn_units = kwargs['model'].get('rnn_units')
        structure = '-'.join(
            ['%d' % rnn_units for _ in range(num_rnn_layers)])
        horizon = kwargs['model'].get('horizon')
        filter_type = kwargs['model'].get('filter_type')
        filter_type_abbr = 'L'
        if filter_type == 'random_walk':
            filter_type_abbr = 'R'
        elif filter_type == 'dual_random_walk':
            filter_type_abbr = 'DR'
        run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
            filter_type_abbr, max_diffusion_step, horizon,
            structure, learning_rate, batch_size,
            time.strftime('%m%d%H%M%S'))
        base_dir = kwargs.get('base_dir')
        log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def load_model(self):
        self._setup_graph()
        print(self._epoch_num)
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, patience=50, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8):
        
        min_val_loss = float('inf')

        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=self.base_lr, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.steps, gamma=self.lr_decay_ratio)

        self._logger.info('Start training ...')

        num_batches = self._data['train_loader'].num_batch
        print("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, self.epoch):

            self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data['train_loader'].get_iterator()
            losses = []

            start_time = time.time()

            for _, (x, y) in enumerate(train_iterator):
                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x, y, batches_seen)

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=self.base_lr, eps=epsilon)

                loss = self._compute_loss(y, output)

                print("loss item: ", loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            val_loss, _ = self.evaluate(dataset='val', batches_seen=batches_seen)

            end_time = time.time()

            self._writer.add_scalar('training loss',
                                    np.mean(losses),
                                    batches_seen)

            if (epoch_num % log_every) == log_every - 1:
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, val_mae: {:.4f}, lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, self.epoch, batches_seen,
                                           np.mean(losses), val_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, self.epoch, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                self._logger.info(message)

            if val_loss < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    print(
                        'Val loss decrease from {:.4f} to {:.4f}, '
                        'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss

            elif val_loss >= min_val_loss:
                wait += 1
                if wait == patience:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_num)
                    break

    def _prepare_data(self, x, y):
            x, y = self._get_x_y(x, y)
            print(y.shape)
            x, y = self._get_x_y_in_correct_dims(x, y)
            return x.to(device), y.to(device)

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        print(x.shape)
        print(y.shape)
        batch_size = x.size(1)

        print(self.seq_len, batch_size, self.num_nodes, self.input_dim)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y.view(self.horizon, batch_size, self.num_nodes)
        return x, y
    
    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                    y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        print("X: {}".format(x.size()))
        print("y: {}".format(y.size()))
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0, 2)
        return x, y

    def _compute_loss(self, y_true, y_predicted):
        y_true = self.standard_scaler.inverse_transform(y_true)
        y_predicted = self.standard_scaler.inverse_transform(y_predicted)
        return masked_mae_loss(y_predicted, y_true), masked_mape(y_predicted, y_true, 0.0), masked_rmse(y_predicted, y_true, 0.0)

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses = []
            mape_loss_arr = []
            rmse_loss_arr = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)

                output = self.dcrnn_model(x)
                mae_loss, mape_loss, rmse_loss = self._compute_loss(y, output)
                losses.append(mae_loss.item())
                mape_loss_arr.append(mape_loss.item())
                rmse_loss_arr.append(rmse_loss.item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

            mean_loss = np.mean(losses)
            mape_loss_f = np.mean(mape_loss_arr)
            rmse_loss_f = np.mean(rmse_loss_arr)

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            return mean_loss, mape_loss_f, rmse_loss_f, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        print("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch