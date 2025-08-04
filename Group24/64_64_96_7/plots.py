#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file "summary.csv"
df = pd.read_csv("summary_big.csv")

# Get the unique method names and ensure the node values are sorted
methods = df["method used"].unique()
nodes = sorted(df["number of nodes"].unique())

# ---------------- Plot for Time-1 ----------------
plt.figure()
for method in methods:
    data_method = df[df["method used"] == method].sort_values(by="number of nodes")
    plt.plot(data_method["number of nodes"], data_method["time-1"],
             marker='o', linestyle='-', label=method)
plt.xlabel("Number of Nodes")
plt.ylabel("Time 1 (seconds)")
plt.title(" Read Time vs. Number of Nodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time1_plot.png")  # Save plot as a PNG file
plt.show()

# ---------------- Plot for Time-2 ----------------
plt.figure()
for method in methods:
    data_method = df[df["method used"] == method].sort_values(by="number of nodes")
    plt.plot(data_method["number of nodes"], data_method["time-2"],
             marker='o', linestyle='-', label=method)
plt.xlabel("Number of Nodes")
plt.ylabel("Time 2 (seconds)")
plt.title("Main code Time vs. Number of Nodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time2_plot.png")
plt.show()

# ---------------- Plot for Time-3 ----------------
plt.figure()
for method in methods:
    data_method = df[df["method used"] == method].sort_values(by="number of nodes")
    plt.plot(data_method["number of nodes"], data_method["time-3"],
             marker='o', linestyle='-', label=method)
plt.xlabel("Number of Nodes")
plt.ylabel("Time 3 (seconds)")
plt.title("Total time vs. Number of Nodes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("time3_plot.png")
plt.show()
