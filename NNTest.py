# from numpy import load
# from sklearn.model_selection import train_test_split
#
# a = load('mfcc_file.npy')
#

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_json('mfcc.json', orient='index')
dataset = dataset[sorted(dataset.columns)]

x = dataset.drop(columns=[13])
y = dataset[13].values

x = x.apply(pd.Series)

# Splitting the DF to get 80/20 elements
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1,
                                                                        stratify=y)


print(x_test)
print(y_test)

