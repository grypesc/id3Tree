import pandas as pd

data = pd.read_csv("data.csv")
data = data.sample(frac=1).reset_index(drop=True)
data.loc[:,'class'].replace({"e":0,"p":1},inplace=True)

data.head()
print(data.describe(include=["O))
