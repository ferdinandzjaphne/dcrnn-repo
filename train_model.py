import argparse
import yaml
from model.dcrnn_supervisor import DCRNNSupervisor
import const
from lib.load_graph import get_adjacency_matrix

def main(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.safe_load(f)

        graph_filename = supervisor_config[args.module].get('adj_csv_file')
        data_filename =  supervisor_config[args.module].get('csv_file')
        adj_mx = get_adjacency_matrix(graph_filename, data_filename)

        supervisor = DCRNNSupervisor(
            adj_mx= adj_mx, 
            **supervisor_config.get(args.module))

        supervisor.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default=None, type=str,
                        help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--module', default=None, type=str, help='module name, can be urban-core or urban-mix')
    args = parser.parse_args()
    main(args)
