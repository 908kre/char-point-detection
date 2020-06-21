import argparse

from .pipeline import train, submit

parser = argparse.ArgumentParser(description="")
subparsers = parser.add_subparsers()

parser_train = subparsers.add_parser("train")
parser_train.add_argument("--id", type=int, dest="fold_idx")
parser_train.set_defaults(handler=train)

parser_submit = subparsers.add_parser("submit")
parser_submit.add_argument("--id", type=int, dest="fold_idx", default=0)
parser_submit.set_defaults(handler=submit)


def main() -> None:
    args = parser.parse_args()
    if args.handler is not None:
        kwargs = vars(args)
        handler = args.handler
        kwargs.pop("handler")
        handler(**kwargs)
    else:
        parser.print_help()
