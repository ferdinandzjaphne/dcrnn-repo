import torch
import time
import numpy as np
from model.dcrnn_model import DCRNNModel
from lib import load_dataset as ld, utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data = ld.load_dataset(**self._kwargs)
        self.base_lr = kwargs.get('base_lr')
        self.epsilon = kwargs.get('epsilon')
        self.steps = kwargs.get('steps')
        self.lr_decay_ratio = kwargs.get('lr_decay_ratio')
        self._epoch_num = 0
        self.seq_len = kwargs.get('seq_len')
        self.num_nodes = kwargs.get('num_nodes')
        self.input_dim = kwargs.get('input_dim')
        self.output_dim = kwargs.get('output_dim')
        self.horizon = kwargs.get('horizon')

        self.epoch = kwargs.get('epoch')

        dcrnn_model = DCRNNModel(adj_mx, **self._kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model

        
    def train(self, patience=50, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8):
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=self.base_lr, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.steps,
                                                            gamma=self.lr_decay_ratio)
    
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
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=self.base_lr, eps=self.epsilon)

                loss = self._compute_loss(y, output)

                print(loss.item())

                losses.append(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)

                optimizer.step()
            print("epoch complete")
            lr_scheduler.step()
            print("evaluating now!")

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
                print(message)

            if (epoch_num % test_every_n_epochs) == test_every_n_epochs - 1:
                test_loss, _ = self.evaluate(dataset='test', batches_seen=batches_seen)
                message = 'Epoch [{}/{}] ({}) train_mae: {:.4f}, test_mae: {:.4f},  lr: {:.6f}, ' \
                          '{:.1f}s'.format(epoch_num, self.epoch, batches_seen,
                                           np.mean(losses), test_loss, lr_scheduler.get_lr()[0],
                                           (end_time - start_time))
                print(message)

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
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
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
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

