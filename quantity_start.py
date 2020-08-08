import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time

from utils.basic_calcs import *

data = get_data()
revenue_data, qty_data, hits_data = get_data_vals(data)

initial_solution = np.argmax(qty_data, axis = 1)

x = 10
y = 25
desired_revenue, desired_qty = get_desired_vals(data, x, y)

check_solution(data, initial_solution, desired_revenue, desired_qty)

revenue_max_solution = np.argmax(revenue_data, axis = 1)
solution = np.zeros(revenue_max_solution.shape[0], dtype = 'int32')
for idx in range(300):
    if(initial_solution[idx] > revenue_max_solution[idx]):
        solution[idx] = revenue_max_solution[idx]
    else:
        solution[idx] = initial_solution[idx]

check_solution(data, solution, desired_revenue, desired_qty)

def find_best_params(solution, hits_per_qty_thresh, thresh_diff):
    while(True):
        hits_per_qty_thresh -= 1
        hits_per_qty = {}
        for idx in range(300):
            if(solution[idx] == 0):
                hits_per_qty[idx] = -1
            else:
                delta_hits = hits_data[idx, solution[idx] - 1] - hits_data[idx, solution[idx]]
                delta_qty = qty_data[idx, solution[idx] - 1] - qty_data[idx, solution[idx]]
                hits_per_qty[idx] = delta_hits / delta_qty
        
        broken = False
        sorted_hits_per_qty = list(sorted(hits_per_qty.items(), key = lambda x: x[1], reverse = True))
        for i in range(300):
            item_idx = sorted_hits_per_qty[i][0]
            item_hits_per_qty = sorted_hits_per_qty[i][1]
            if((item_hits_per_qty == -1) | (item_hits_per_qty < hits_per_qty_thresh)):
                break
            new_solution = np.zeros(300, dtype = 'int32')
            for j in range(300):
                new_solution[j] = solution[j]
            new_solution[item_idx] = solution[item_idx] - 1
            valid, hits, revenue, qty = check_solution(data, new_solution, desired_revenue, desired_qty)
            if(valid == False):
                broken = True
                break
            else:
                solution = new_solution
        if(broken == True):
            break
    return solution, check_solution(data, solution, desired_revenue, desired_qty)

def tune_thresh(thresh_vals, diff_vals):
    combinations = product(thresh_vals, diff_vals)
    combination_results = {}
    for combination in combinations:
        start = time.time()
        combination_results[combination] = find_best_params(solution, combination[0], combination[1])
        print(f"Combination: {combination}")
        print(f"Time Taken: {(time.time() - start):.1f}s")
        print(f"Hits: {combination_results[combination][1][1]}")
    return combination_results

results = tune_thresh([20, 30, 40], [0.5, 1, 1.5])
print(results)