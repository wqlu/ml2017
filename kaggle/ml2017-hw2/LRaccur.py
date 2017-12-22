# -*- coding: utf-8
# !/usr/bin/env python

import pandas as pd
import numpy as np
import os

correct_answer = pd.read_csv('correct_answer.csv')
correct_answer = np.array(correct_answer)
# print correct_answer['label']
get_answer = pd.read_csv('answer')
get_answer = np.array(get_answer)
# print get_answer
size = len(correct_answer)
sum = 0.0
for i,j in zip(correct_answer, get_answer):
    if i[1] == j[1]:
        sum += 1
print sum/size # 0.842454394693
