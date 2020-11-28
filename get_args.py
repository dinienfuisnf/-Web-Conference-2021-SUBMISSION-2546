import argparse
import numpy as np

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser()
parser.add_argument("--thread_num", type=int, help='thread_num',default=12)
parser.add_argument("--num_pop", type=int, help='num_pop',default=10000)
parser.add_argument("--q_threshold", type=int, help='q_threshold',default=10000)
parser.add_argument("--i_threshold", type=int, help='i_threshold',default=500)
parser.add_argument("--risk_threshold", type=float, help='risk_threshold',default=0.5)
# Env
parser.add_argument("--period", type=int, help='period',default = 14 * 60)
parser.add_argument("--fixed_no_policy_days", type=int, help='fixed_no_policy_days',default=1)

args = parser.parse_args()