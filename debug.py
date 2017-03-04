import argparse

parser = argparse.ArgumentParser(description="Run Name Entity Recognition powered by Feature Rich Model")
parser.add_argument('--learning-rate', '-l', type=float, default=1,
    help="The learning rate eta of the Perceptron, should be a float.")
parser.add_argument('--epoch', '-e', type=int, default=5,
    help="The epoch time of the Perceptron, should be an int")
parser.add_argument('--current-token', nargs='?', default=False, const=True)

args = parser.parse_args()

import pdb; pdb.set_trace()
print args.sum
