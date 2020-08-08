import numpy as np
import pandas as pd

price_cols = ['Base_Price', 'Price1', 'Price2', 'Price3', 'Price4']
qty_cols = ['Base_Units', 'Units1', 'Units2', 'Units3', 'Units4']
revenue_cols = ['Base_Revenue', 'Revenue1', 'Revenue2', 'Revenue3', 'Revenue4']
hits_cols = ['Base_Hits', 'Hits1', 'Hits2', 'Hits3', 'Hits4']

def get_data():
    data = pd.read_csv('./data/dataset.csv')
    data = data.assign(Base_Revenue = data['Base_Price'] * data['Base_Units'])
    data = data.assign(Revenue1 = data['Price1'] * data['Units1'])
    data = data.assign(Revenue2 = data['Price2'] * data['Units2'])
    data = data.assign(Revenue3 = data['Price3'] * data['Units3'])
    data = data.assign(Revenue4 = data['Price4'] * data['Units4'])
    data = data.assign(Base_Hits = (data['Base_Price'] - data['Base_Price']) * data['Base_Units'])
    data = data.assign(Hits1 = (data['Base_Price'] - data['Price1']) * data['Units1'])
    data = data.assign(Hits2 = (data['Base_Price'] - data['Price2']) * data['Units2'])
    data = data.assign(Hits3 = (data['Base_Price'] - data['Price3']) * data['Units3'])
    data = data.assign(Hits4 = (data['Base_Price'] - data['Price4']) * data['Units4'])
    data = data.set_index('Item_id')
    return data

def get_data_vals(data):
    revenue_data = data[revenue_cols].values
    qty_data = data[qty_cols].values
    hits_data = data[hits_cols].values
    return revenue_data, qty_data, hits_data

def get_hits(data, solution):
    selling_price = np.zeros(len(solution))
    selling_qty = np.zeros(len(solution))
    for item_num in range(data.shape[0]):
        selling_price[item_num] = data.loc[item_num + 1][price_cols[solution[item_num]]]
        selling_qty[item_num] = data.loc[item_num + 1][qty_cols[solution[item_num]]]
    base_price = np.asarray(data['Base_Price'].values)
    return np.sum(np.multiply(base_price - selling_price, selling_qty))

def get_desired_vals(data, x, y):
    desired_revenue = data['Base_Revenue'].sum() * (1 + x / 100.0)
    desired_qty = data['Base_Units'].sum() * (1 + y / 100.0)
    return desired_revenue, desired_qty

def check_solution(data, solution, desired_revenue, desired_qty, verbose = False):
    solution_revenue = 0
    solution_qty = 0
    solution_hits = 0
    for item_num in range(data.shape[0]):
        solution_revenue += data.loc[item_num + 1][revenue_cols[solution[item_num]]]
        solution_qty += data.loc[item_num + 1][qty_cols[solution[item_num]]]
    valid = True
    if(solution_revenue < desired_revenue):
        if(verbose == True):
            print("Revenue criteria not met")
        valid = False
    if(solution_qty < desired_qty):
        if(verbose == True):
            print("Quantity criteria not met")
        valid = False
    solution_hits = get_hits(data, solution)
    return valid, solution_hits, solution_revenue, solution_qty

def softmax(weights):
    exp_weights = np.exp(weights)
    exp_weights = np.divide(exp_weights, np.repeat(np.expand_dims(np.sum(exp_weights, axis = 1), axis = 1), 5, axis = 1))
    return exp_weights