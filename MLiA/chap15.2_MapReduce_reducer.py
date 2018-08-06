
# coding: utf-8
import numpy as np
import sys
import os

def read_input(file):
    for line in file:
        yield line.rstrip()
inputGen = read_input(sys.stdin)
#inputGen = read_input(open(file))
# 读取mapper的输出，mapper输出到 stdin中 有，numInputs，mean(input)，mean(sqInput)
mapperOut = [line.split('\t') for line in inputGen]

cumVal = 0.0
cumSumSq = 0.0
cumN = 0.0

for instance in mapperOut:
    nj = float(instance[0])
    cumN += nj
    # 各mapper的总和
    cumVal += nj*float(instance[1])
    cumSumSq += nj*float(instance[2])
mean = cumVal/cumN
varSum = (cumSumSq - cumN*mean*mean)/cumN
print("%d\t%f\t%f" % (cumN, mean, varSum))
print("report: still alive", file=sys.stderr)

