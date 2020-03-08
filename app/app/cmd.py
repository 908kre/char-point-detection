import yaml
import argparse
from . import flows


def preprocess() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="input file")
    args = parser.parse_args()
    flows.preprocess(input_path=args.input, output_path=args.output)
