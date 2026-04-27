# -*- coding: utf-8 -*-
"""
Plots.py (Task 6)
==================
Generates all figures required by Task 6 for the PDF report.
Imports results from Evaluate.py.

Run Evaluate.py first, then run this file.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = "/Users/manostsili/Desktop/dtu/courses/decision making under uncertainty /assignment_DC"
TASK6_DIR = os.path.join(BASE_DIR, "task6")
TASK1_DIR = os.path.join(BASE_DIR, "Task1")

sys.path.insert(0, TASK6_DIR)
sys.path.insert(0, TASK1_DIR)

# ── import results from Evaluate.py ──────────────────────────────────────────
from Evaluate import results


# =============================================================================
# FIGURE 1: Average daily cost per policy (bar chart)
# =============================================================================

policy_names = list(results.keys())
avg_costs    = [np.mean(c) for c in results.values()]
std_costs    = [np.std(c)  for c in results.values()]

plt.figure(figsize=(10, 5))
plt.bar(policy_names, avg_costs, yerr=std_costs,
        capsize=5, color="steelblue", edgecolor="black", alpha=0.85)
plt.title("Average Daily Electricity Cost per Policy", fontsize=13)
plt.ylabel("Average Daily Cost (€)")
plt.xlabel("Policy")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(TASK6_DIR, "fig1_avg_cost.png"), dpi=150)
plt.show()
print("Saved: fig1_avg_cost.png")


# =============================================================================
# FIGURE 2: Histogram of daily costs per policy
# =============================================================================

plt.figure(figsize=(10, 5))
for name, costs in results.items():
    plt.hist(costs, bins=20, alpha=0.6, label=name, edgecolor="black")

plt.title("Distribution of Daily Costs per Policy", fontsize=13)
plt.xlabel("Daily Cost (€)")
plt.ylabel("Frequency (days)")
plt.legend()
plt.grid(linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(TASK6_DIR, "fig2_histogram.png"), dpi=150)
plt.show()
print("Saved: fig2_histogram.png")