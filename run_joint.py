import argparse
import os
import time


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


def get_params():
    from utils.utils import get_common_params
    parser = get_common_params()
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    # Get common parameters from utils
    params = vars(get_params())

    # Import the Graph2Route model and run it using the parameters
    from models import graph2route as graph2route_pd

    graph2route_pd.main(params)
