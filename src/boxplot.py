from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Create output folder if it does not exist
output_dir = Path("figs")
output_dir.mkdir(exist_ok=True)

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Create boxplot for median house value
plt.figure(figsize=(8, 6))
plt.boxplot(df["MedHouseVal"])
plt.title("Boxplot of Median House Value - California Housing Dataset")
plt.ylabel("Median House Value")
plt.grid(axis="y", alpha=0.3)

# Save figure
plt.savefig("figs/boxplot.png", dpi=300, bbox_inches="tight")
plt.close()

print("Boxplot saved successfully to figs/boxplot.png")
