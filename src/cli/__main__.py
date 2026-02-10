import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from cli import lapsim, tracks

import argparse

# from . import lapsim, tracks


# TODO Write proper readme for converter and update notion with explantions for how it works


parser = argparse.ArgumentParser()

parser.add_argument('module', type=str)
parser.add_argument('function', type=str)

# Common required
parser.add_argument('--src', type=str)
parser.add_argument('--dest', type=str)

# Common optional
parser.add_argument("--cores", type=int)
parser.add_argument("--batch-size", type=int)
parser.add_argument("--portion", type=float)
parser.add_argument("--seed", type=int)

# Optional converter
parser.add_argument("--spacing", type=int)
parser.add_argument("--precision", type=int)

# Optional encoder
parser.add_argument("--partitions", type=int)
parser.add_argument("--flip", nargs='?', const=True)

args = parser.parse_args()

breakpoint()
if args.module == 'lapsim':
    lapsim.parse_cli_args(args)

elif args.module == 'tracks':
    tracks.parse_cli_args(args)

else:
    print(f"Unknown module: {args.module}. Please choose from: 'lapsim'")
