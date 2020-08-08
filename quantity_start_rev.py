#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
import time

from utils.basic_calcs import *

data = get_data()
revenue_data, qty_data, hits_data = get_data_vals(data)

#%%
#solution = pd.read_csv('./data/result.csv', index_col = 0)['Price']
#solution = [price_cols.index(x) for x in solution]

# %%
initial_solution = np.argmin(qty_data, axis = 1)

# %%
x = 7
y = 20
desired_revenue, desired_qty = get_desired_vals(data, x, y)

#%%
check_solution(data, initial_solution, desired_revenue, desired_qty)

# %%
#revenue_max_solution = np.argmin(revenue_data, axis = 1)
#solution = np.zeros(revenue_max_solution.shape[0], dtype = 'int32')
#for idx in range(300):
#    if(initial_solution[idx] < revenue_max_solution[idx]):
#        solution[idx] = revenue_max_solution[idx]
#    else:
#        solution[idx] = initial_solution[idx]
solution = initial_solution

# %%
check_solution(data, solution, desired_revenue, desired_qty)

# %%
#idx = 101
#price_data = data[price_cols].values
#plt.plot(qty_data[idx, :], revenue_data[idx, :])
#plt.show()

# %%
def optimize(solution, hits_per_qty_thresh, thresh_diff):
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

#%%
def optimize2(solution):
    iter_num = 0
    use_all = False
    while(True):
        print("ITER NUM", iter_num + 1)
        hits_per_qty = {}
        for idx in range(300):
            if(solution[idx] == 4):
                hits_per_qty[idx] = 10000000
            else:
                delta_hits = hits_data[idx, solution[idx] + 1] - hits_data[idx, solution[idx]]
                delta_qty = qty_data[idx, solution[idx] + 1] - qty_data[idx, solution[idx]]
                delta_revenue = revenue_data[idx, solution[idx] + 1] - revenue_data[idx, solution[idx]]
                hits_per_qty[idx] = np.abs(delta_hits / delta_qty)
        
        broken = False
        err_cnt = 0
        sorted_hits_per_qty = list(sorted(hits_per_qty.items(), key = lambda x: x[1], reverse = False))
        for i in range(300):
            item_idx = sorted_hits_per_qty[i][0]
            item_hits_per_qty = sorted_hits_per_qty[i][1]
            if(item_hits_per_qty == 10000000):
                continue
            new_solution = np.zeros(300, dtype = 'int32')
            for j in range(300):
                new_solution[j] = solution[j]
            new_solution[item_idx] = solution[item_idx] + 1
            valid, hits, revenue, qty = check_solution(data, new_solution, desired_revenue, desired_qty)
            print(hits, revenue, qty)
            solution = new_solution
            if(valid == True):
                broken = True
                break
            if(i == 0):
                break
        if(broken == True):
            break
        iter_num += 1
    print("DONE!")
    return solution, check_solution(data, solution, desired_revenue, desired_qty)
    
#%%
best_solution, results = optimize2(solution)
# %%
def tune_thresh(thresh_vals, diff_vals):
    combinations = product(thresh_vals, diff_vals)
    combination_results = {}
    for combination in combinations:
        start = time.time()
        combination_results[combination] = optimize(solution, combination[0], combination[1])
        print(f"Combination: {combination}")
        print(f"Time Taken: {(time.time() - start):.1f}s")
        print(f"Hits: {combination_results[combination][1][1]}")
        print()
    return combination_results
# %%
#results = tune_thresh([25, 30, 35], [1, 2])
#best_solution = list(sorted(results.items(), key = lambda x: x[1][1][2]))[0][1][0]

#%%
assert check_solution(data, best_solution, desired_revenue, desired_qty)[0] == True

# %%
best_prices = [price_cols[x] for x in best_solution]
final_df = pd.DataFrame({'Item id': np.arange(1, 301), 'Price': best_prices})
final_df.to_csv('./results_rev_criteria3.csv', index = False)
