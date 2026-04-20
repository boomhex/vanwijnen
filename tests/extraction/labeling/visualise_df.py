import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("post_dataset.csv")

labels = ["Amount", "Geen amount"]

def true_percentage(df, colname):
    return (sum(df[colname]) / len(df)) * 100

per = true_percentage(df[df["post"] == False], "has_amount")

counts = [
    per,
    100 - per
]

fig, ax = plt.subplots()

ax.bar(labels, counts, color=['tab:red', 'tab:blue'])
ax.set_ylim(0, 100)
ax.set_ylabel("Percentage (%)")
plt.show()
