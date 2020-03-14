import yaml
import argparse
from . import flows


def preprocess() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="input file")
    args = parser.parse_args()
    flows.preprocess(input_path=args.input, output_path=args.output)


def kfold() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input file")
    parser.add_argument("--output", type=str, help="output directory")
    args = parser.parse_args()
    flows.kfold(
        input_path=args.input, output_dir=args.output,
    )


def dea() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="test csv")
    parser.add_argument("--test", type=str, help="train csv")
    args = parser.parse_args()
    flows.dea(
        test_path=args.test, train_path=args.train,
    )
