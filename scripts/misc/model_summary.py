#!/usr/bin/env python3

import sys

from keras.models import load_model


def main():
    if len(sys.argv) != 2:
        raise ValueError("This script takes the weights file path as its sole argument.")

    weights_file = sys.argv[1]
    print('Reading model from %s' % weights_file)

    model = load_model(weights_file)

    model.build_model()
    model.summary()

    return 0


if __name__ == '__main__':
    sys.exit(main())
