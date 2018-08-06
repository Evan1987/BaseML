
# coding: utf-8

import numpy as np
import sys


def read_input(file):
    for line in file:
        yield line.rstrip()

inputGen = read_input(sys.stdin)
inputData = [float(line) for line in inputGen]

numInputs = len(inputData)
input = np.mat(inputData)
sqInput = np.power(input, 2)
print("%d\t%f\t%f" % (numInputs, np.mean(input), np.mean(sqInput)))
print("report: still alive", file=sys.stderr)
