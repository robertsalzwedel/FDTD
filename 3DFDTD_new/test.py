from collections import namedtuple
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument(
    "--pulse",
    "-p",
    help="Choice: [THz, Optical]",
    choices=["THz", "Optical"],
    default="Optical",
)
args = parser.parse_args()
print(args.pulse)


def to_namedtuple(classname="argparse_to_namedtuple", **kwargs):
    return namedtuple(classname, tuple(kwargs))(**kwargs)


nt = to_namedtuple(**vars(args))
print(nt.pulse)
