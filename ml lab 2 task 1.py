import numpy as np
import pandas as pd 


data=pd.read_csv("Lab2 D1A.csv")
print(data.info())
data1=pd.read_csv("Lab2 D1B.csv")
print(data1.info())

combined=pd.concat([data,data1], axis=1)
combined = combined.loc[:, ~combined.columns.duplicated()]

print(combined.shape)
print(combined.head())
print(combined.info())

data2=pd.read_csv("Lab2 D1C.csv")
print("this is D1C", data2.info())
print("this is D1A", data.info())

common_cols = list(data.columns.intersection(data2.columns))
comboAC = pd.merge(data, data2, on=common_cols, how="inner")
print(comboAC.shape)   
print(comboAC.head())
print(comboAC.columns)