import pandas as pd

df = pd.read_csv("data/raw/dataset.csv")
print(df["label"].value_counts().head(20))