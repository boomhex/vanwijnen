import pandas as pd

FILE = "post_dataset(1).csv"

df = pd.read_csv(FILE)

df = df[["text","post","page","has_amount","has_unit"]]
df.to_csv("post_dataset(2).csv")