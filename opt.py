import argparse
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):

        # --------------------------  General Training Options
        self.parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
        self.parser.add_argument('--number_of_epoch', type=int, default=16, help='Epoch')
        self.parser.add_argument('--batch_size', type=int, default=8, help='Batch Size')
        self.parser.add_argument('--node_count', type=int, default=1024, help='Hidden Layer Node Count')
        # --------------------------  General Training Options

        self.parser.add_argument('--exp', type=str, default='test/', help='Experiment name')

        # --------------------------
        self.parser.add_argument('--coeff', type=int, default=2, help='Coefficient of bbox size')
        self.parser.add_argument('--threshold', type=int, default=0.21, help='BBOX threshold')
        self.parser.add_argument('--window_size', type=int, default=15, help='Windows size for cropping')
        # --------------------------

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self._print()
        return self.opt
