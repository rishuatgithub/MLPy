# this code is for the instacart predict kaggle challenge
# author: rishu shirvastava (rishu.shrivastava@gmail.com)
# date: 16.07.2017

import pandas as pd


#read csv files
aisles_df = pd.read_csv('~/Desktop/dATA/instacart/data/aisles.csv')
department_df = pd.read_csv('~/Desktop/dATA/instacart/data/departments.csv')
order_products_prior_df = pd.read_csv('~/Desktop/dATA/instacart/data/order_products__prior.csv')
order_products_train_df = pd.read_csv('~/Desktop/dATA/instacart/data/order_products__train.csv')
order_df = pd.read_csv('~/Desktop/dATA/instacart/data/orders.csv')
products_df = pd.read_csv('~/Desktop/dATA/instacart/data/products.csv')
