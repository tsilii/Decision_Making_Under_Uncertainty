# -*- coding: utf-8 -*-
"""
Evaluate.py (Task 6)
=====================
Runs all policies through the environment for 100 days and collects results.
Run this file first, then run Plots.py to generate figures.
"""

import numpy as np
import sys
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = "/Users/manostsili/Desktop/dtu/courses/decision making under uncertainty /assignment_DC"
TASK6_DIR = os.path.join(BASE_DIR, "task6")
TASK1_DIR = os.path.join(BASE_DIR, "Task1")

sys.path.insert(0, TASK6_DIR)   # Environment, DummyPolicy
sys.path.insert(0, TASK1_DIR)   # HindsightPolicy
# sys.path.insert(0, os.path.join(BASE_DIR, "task3"))  # SPPolicy etc.

# ── import environment ────────────────────────────────────────────────────────
from Environment import run_simulation

# ── import policies ───────────────────────────────────────────────────────────
from DummyPolicy     import DummyPolicy
from HindsightPolicy import HindsightPolicy
from SPPolicy import SPPolicy
# from EVPolicy        import EVPolicy
# from TwoStagePolicy  import TwoStagePolicy
# from ADPPolicy       import ADPPolicy
# from HybridPolicy    import HybridPolicy


# =============================================================================
# RUN ALL POLICIES
# =============================================================================

print("Running evaluations...")

dummy_costs = run_simulation(DummyPolicy(), num_experiments=100)
print(f"Dummy Policy  -> avg cost: {np.mean(dummy_costs):.2f}")

hindsight_costs = run_simulation(HindsightPolicy(), num_experiments=100)
print(f"Hindsight     -> avg cost: {np.mean(hindsight_costs):.2f}")

# Uncomment as you build each policy:
#sp_costs       = run_simulation(SPPolicy(),       num_experiments=100)
#print(f"SP Policy     -> avg cost: {np.mean(sp_costs):.2f}")

sp_costs = run_simulation(SPPolicy(), num_experiments=100, verbose=True)
print(f"SP Policy     -> avg cost: {np.mean(sp_costs):.2f}")

# ev_costs       = run_simulation(EVPolicy(),       num_experiments=100)
# print(f"EV Policy     -> avg cost: {np.mean(ev_costs):.2f}")

# twostage_costs = run_simulation(TwoStagePolicy(), num_experiments=100)
# print(f"Two-Stage     -> avg cost: {np.mean(twostage_costs):.2f}")

# adp_costs      = run_simulation(ADPPolicy(),      num_experiments=100)
# print(f"ADP Policy    -> avg cost: {np.mean(adp_costs):.2f}")

# hybrid_costs   = run_simulation(HybridPolicy(),   num_experiments=100)
# print(f"Hybrid Policy -> avg cost: {np.mean(hybrid_costs):.2f}")


# =============================================================================
# COLLECT RESULTS
# =============================================================================

results = {
    "Dummy":      dummy_costs,
    "Hindsight":  hindsight_costs,
    "SP":         sp_costs,
    # "EV":         ev_costs,
    # "Two-Stage":  twostage_costs,
    # "ADP":        adp_costs,
    # "Hybrid":     hybrid_costs,
}

# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

print("\n" + "="*55)
print(f"{'Policy':<15} {'Avg Cost':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("="*55)
for name, costs in results.items():
    print(f"{name:<15} {np.mean(costs):>10.2f} {np.std(costs):>10.2f} "
          f"{min(costs):>10.2f} {max(costs):>10.2f}")
print("="*55)
print("\nDone. Run Plots.py to generate figures.")