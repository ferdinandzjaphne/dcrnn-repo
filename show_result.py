import argparse
import numpy as np
import os
import sys
import yaml

from lib.load_graph import get_adjacency_matrix
from model.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)
        # b = np.load('data/dcrnn_predictions.npz')
        # print(b)

        graph_filename = supervisor_config[args.module].get('data').get('adj_csv_file')
        data_filename =  supervisor_config[args.module].get('data').get('csv_file')
        adj_mx = get_adjacency_matrix(graph_filename, data_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config.get(args.module))
        mean_score, outputs = supervisor.evaluate('test')
        print(outputs)
        np.savez_compressed(args.output_filename, **outputs)
        print("MAE : {}".format(mean_score))
        print('Predictions saved as {}.'.format(args.output_filename))


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='data/model/pretrained/urban-core/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--module', default='urban_core', type=str,
                        help='module name')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()
    run_dcrnn(args)
