import pandas as pd
df = pd.read_csv('data/wheat.csv')
non_zero = df[df['Daily_NSS'] != 0].shape[0]
print(f"Total rows in wheat.csv: {df.shape[0]}")
print(f"Rows with non-zero sentiment: {non_zero}")
