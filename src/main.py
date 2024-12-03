#!/usr/bin/env python3
"""
Module Docstring
"""

__author__ = "Team names comma separated"
__version__ = "0.1.0"
__license__ = "GPLv3"

import argparse

def main(args):
    """ Main entry point of the app """
    training_file = args.training_file
    testing_file = args.testing_file
    mape = 0.0
    # HERE GOES YOUR CODE TO CALCULATE THE MAPE
    # FEEL FREE TO IMPLEMENT HELPER FUNCTIONS
    print("MAPE: {}".format(mape))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("training_file", help="Training data file")
    parser.add_argument("testing_file", help="Testing data file")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
