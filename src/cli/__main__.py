import sys
from pathlib import Path

from lapsim import encoder

from toolkit.tracks import splicer

sys.path.append(str(Path(__file__).parent.parent))

# from cli import lapsim

import argparse

# from . import lapsim, tracks


# TODO Write proper readme for converter and update notion with explantions for how it works


parser = argparse.ArgumentParser()

parser.add_argument('function', type=str)

parser.add_argument('--src', type=str)
parser.add_argument('--dest', type=str)

parser.add_argument("--spacing", type=int)
parser.add_argument("--partitions", type=int)
parser.add_argument("--flip", nargs='?', const=True)

args = parser.parse_args()

if args.function == 'splice':
    if not args.src or not args.dest or not args.spacing:
        raise Exception("Incorrect args. `splice --src <src> --dest <dest> --spacing <spacing>` ")

    splicer.from_cli(
        args.src,
        args.dest,
        args.spacing
    )

elif args.function == 'encode':
    if not args.src or not args.dest:
        raise Exception("Incorrect args. `encode --src <src> --dest <dest> (optional) --flip --partitions 10` ")

    encoder.from_cli(
        args.src,
        args.dest,
        n_partitions=args.partitions,
        flip=args.flip,
    )

else:
    print(f"Unknown function: {args.function}. Please choose from: 'splice'")
