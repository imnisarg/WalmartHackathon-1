import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time

from utils.basic_calcs import *

data = get_data()
revenue_data, qty_data, hits_data = get_data_vals(data)

initial_solution = np.argmax(qty_data, axis = 1)

x = 5
y = 15
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

def optimize2(solution):
    iter_num = 0
    use_all = False
    while(True):
        print("ITER NUM", iter_num + 1)
        hits_per_qty = {}
        for idx in range(300):
            if(solution[idx] == 0):
                hits_per_qty[idx] = -1
            else:
                delta_hits = hits_data[idx, solution[idx] - 1] - hits_data[idx, solution[idx]]
                delta_qty = qty_data[idx, solution[idx] - 1] - qty_data[idx, solution[idx]]
                delta_revenue = revenue_data[idx, solution[idx] - 1] - revenue_data[idx, solution[idx]]
                hits_per_qty[idx] = np.abs(delta_hits / delta_qty)
        
        broken = False
        err_cnt = 0
        sorted_hits_per_qty = list(sorted(hits_per_qty.items(), key = lambda x: x[1], reverse = True))
        for i in range(300):
            if(use_all == True):
                if(i == 299):
                    broken = True
                    break
            item_idx = sorted_hits_per_qty[i][0]
            item_hits_per_qty = sorted_hits_per_qty[i][1]
            if(item_hits_per_qty == -1):
                continue
            new_solution = np.zeros(300, dtype = 'int32')
            for j in range(300):
                new_solution[j] = solution[j]
            new_solution[item_idx] = solution[item_idx] - 1
            valid, hits, revenue, qty = check_solution(data, new_solution, desired_revenue, desired_qty)
            if(valid == False):
                err_cnt += 1
                use_all = True
            else:
                print(hits, revenue, qty)
                solution = new_solution
            if(use_all == False):
                if(i == 0):
                    break
        if(broken == True):
            break
        iter_num += 1
    print("DONE!")
    return solution, check_solution(data, solution, desired_revenue, desired_qty)
    
best_solution, results = optimize2(solution)
assert check_solution(data, best_solution, desired_revenue, desired_qty)[0] == True

best_prices = [price_cols[x] for x in best_solution]
final_df = pd.DataFrame({'Item id': np.arange(1, 301), 'Price': best_prices})
final_df.to_csv('./results_criteria2.csv', index = False)
