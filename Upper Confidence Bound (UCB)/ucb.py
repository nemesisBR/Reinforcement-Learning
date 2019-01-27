# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv(r"F:\Udemy\Machine Learning\Machine Learning A-Z Template Folder\Part 6 - Reinforcement Learning\Section 32 - Upper Confidence Bound (UCB)\Ads_CTR_Optimisation.csv")

import math
N = 10000
d = 10
ads_selected = []
numbers_of_selection = [0] * d
sums_of_reward = [0] * d
total_reward = 0
for n in range(0,N):
    max_upper_bound = 0
    ad = 0
    for i in range(0,d):
        if (numbers_of_selection[i]>0):
            average_reward = sums_of_reward[i] / numbers_of_selection[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selection[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400 
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selection[ad] = numbers_of_selection[ad] + 1
    sums_of_reward[ad] = sums_of_reward[ad] + dataset.values[n, ad]
    total_reward = total_reward + dataset.values[n, ad]

plt.hist(ads_selected)
plt.show()