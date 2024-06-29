import argparse
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def get_params():
    """
    _KI_
    To run the joint training, the arguments that are required are set to default settings
    mainly some common settings for deep model , training and dataset.
    """
    parser = argparse.ArgumentParser(description='Entry Point of the code')
    parser.add_argument('--is_test', type=bool, default=False, help='test the code')
    parser.add_argument('--datatype', default='order', type=str, help='datatype')
    parser.add_argument('--batch-size', type=int, default=512, help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 1e-4)')
    parser.add_argument('--seed', type=int, default=2021, metavar='S', help='random seed (default: 6)')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay (default: 1e-5)')
    parser.add_argument('--early_stop', type=int, default=5, help='early stop at')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    args, _ = parser.parse_known_args()

    return args


def main(params):
    model = params['modelRoute']
    if model == 'graph2route_pd':
        import graph2route.graph2route_pd.train as graph2route_pd
        graph2route_pd.main(params)

def get_params():
    from my_utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':

    params = vars(get_params())
    import graph2route.graph2route_pd.train as graph2route_pd
    graph2route_pd.main(params)
