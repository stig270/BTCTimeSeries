import pandas as pd

df = pd.read_csv('metrics.csv')

df["Label"] = (df['Close'].shift(-1) < df['Close']).astype(int)

df.to_csv('data.csv')