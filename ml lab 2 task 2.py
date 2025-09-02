import numpy as np
import pandas as pd

num=50

data=pd.read_csv("Lab2 D1A.csv")
data1=pd.read_csv("Lab2 D1B.csv")
data2=pd.read_csv("Lab2 D1C.csv")
print(data1.columns)
print(data2.columns)
print(data.columns)
CustomizedData=pd.DataFrame({
    "county": np.random.choice(['BRISTOL', "WORCESTER", "PLYMOUTH", "FRANKLIN", "HAMPDEN", "ALLEGAN", "ALGER"], size=num),
    "Size":np.random.choice(["Small", "Medium", "High"], size=num),
    "Cardinal Direction": np.random.choice(["North", "South", "East", "West"], size=num),
    "Timings":np.random.choice(["Full Time", "Part Time"], size=num),
    "Qualification": np.random.choice(["Bsc", "Msc", "Phd"], size=num),
    "Rating": np.random.randint(0,5,size=num)
})
print(CustomizedData.head())

print(CustomizedData.shape)
print(data.shape)
print(data1.shape)
print(data2.shape)

Mergeddata=(CustomizedData.merge(data, how="inner", on="county").merge(data1, how="inner", on="county").merge(data2, how="inner", on="county"))
print(Mergeddata.head())
print(Mergeddata.shape)