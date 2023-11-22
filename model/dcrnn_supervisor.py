from lib import load_dataset as utils

class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data = utils.load_dataset(**self._kwargs)
        
    def train():
        print("train")


def prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

def get_x_y(self, x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                y shape (horizon, batch_size, num_sensor, input_dim)
    """
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()
    self._logger.debug("X: {}".format(x.size()))
    self._logger.debug("y: {}".format(y.size()))
    x = x.permute(1, 0, 2, 3)
    y = y.permute(1, 0, 2, 3)
    return x, y