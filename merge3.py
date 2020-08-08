import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time

from utils.basic_calcs import *

data = get_data()
revenue_data, qty_data, hits_data = get_data_vals(data)

x = 7
y = 20
desired_revenue, desired_qty = get_desired_vals(data, x, y)

results1 = pd.read_csv('./results_criteria3.csv')['Price']
results1 = np.asarray([price_cols.index(x) for x in results1])

results2 = pd.read_csv('./results_rev_criteria3.csv')['Price']
results2 = np.asarray([price_cols.index(x) for x in results2])

idx = np.arange(300)
same_idx = list(idx[(results1 == results2)])
mismatch_idx = list(idx[~(results1 == results2)])

final_solution = np.zeros(300, dtype = 'int32')
for i in same_idx:
    final_solution[i] = results1[i]

mismatch_elems = zip(mismatch_idx, np.abs(results1[mismatch_idx] - results2[mismatch_idx]))
mismatch_idx = list(sorted(mismatch_elems, key = lambda x: x[1], reverse = True))
mismatch_idx = [x[0] for x in mismatch_idx]

err_cnt = 0
new_mismatch_idx = mismatch_idx.copy()
for i in mismatch_idx:
    new_solution = results2.copy()
    new_solution[i] = results1[i]
    valid1, hits1, revenue1, qty1 = check_solution(data, new_solution, desired_revenue, desired_qty)
    new_solution[i] = results2[i]
    valid2, hits2, revenue2, qty2 = check_solution(data, new_solution, desired_revenue, desired_qty)
    if((valid1 == True) & (valid2 == True)):
        if(hits1 < hits2):
            final_solution[i] = results1[i]
        else:
            final_solution[i] = results2[i]
    elif(valid1 == True):
        final_solution[i] = results1[i]
    elif(valid2 == True):
        final_solution[i] = results2[i]
    else:
        err_cnt += 1
        print("Error!")
    new_mismatch_idx.remove(i)
    same_idx.append(i)

check_solution(data, final_solution, desired_revenue, desired_qty)

best_prices = [price_cols[x] for x in final_solution]
final_df = pd.DataFrame({'Item id': np.arange(1, 301), 'Price': best_prices})
final_df.to_csv('./results_merged_criteria3.csv', index = False)