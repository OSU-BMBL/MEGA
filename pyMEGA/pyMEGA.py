"""
Main application file.
"""
import sys

from . import foo
from .lib import greet


def main(args):
    """
    Main command-line function.
    """
    name = "My name"
    print(args)
    print(greet(name))

    print(foo.greet(name))


if __name__ == "__main__":
    print("Hello: " + sys.argv[1:])
    main(sys.argv[1:])
