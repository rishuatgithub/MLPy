# hotstar dataset - India Hack

# author: Rishu Shrivastava (rishu.shrivastava@gmail.com)
# last modified: June 17, 2017

# imports
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize

#read JSON dataset

# hotstar_train = pd.read_json('C:/Users/Rishu/Documents/GitHub/MLPy/data/hotstar/train_data.json')
hotstar_file_train = 'C:/Users/Rishu/Desktop/dATA/5f828822-4--4-hotstar_dataset/train_data.json'
with open(hotstar_file_train) as hotstar_json_file:
    dict_train = json.load(hotstar_json_file)
	
# converting json dataset to dataframe
train = pd.DataFrame.from_dict(dict_train, orient='index')
train.reset_index(level=0, inplace=True)
train.rename(columns = {'index':'ID'},inplace=True)
#print(train.shape)
#print(train.head())
print(train.info())

print(train.stack().unique())

#train.to_csv('C:/Users/Rishu/Desktop/dATA/5f828822-4--4-hotstar_dataset/csv_train.csv', sep=',', encoding='utf-8')