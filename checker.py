import pandas as pd

# read excel file
df = pd.read_csv("data\\raw\\kddcup.data_10_percent.gz", header=None, compression='gzip')   # replace with your file name

print(len(df.columns))
# get unique values from a specific column
unique_values = df[41].unique()   # replace with your column name

print(unique_values)
