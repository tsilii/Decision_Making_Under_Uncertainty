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
BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
TASK1_DIR = os.path.join(BASE_DIR, "Task1")
TASK3_DIR = os.path.join(BASE_DIR, "Task3")
TASK5_DIR = os.path.join(BASE_DIR, "Task5")
TASK6_DIR = os.path.join(BASE_DIR, "task6")
GIVEN_DIR = os.path.join(BASE_DIR, "given")

sys.path.insert(0, TASK1_DIR)
sys.path.insert(0, TASK3_DIR)
sys.path.insert(0, TASK5_DIR)
sys.path.insert(0, TASK6_DIR)
sys.path.insert(0, GIVEN_DIR)

# ── import environment ────────────────────────────────────────────────────────
import v2_SystemCharacteristics as SC
from Environment import run_simulation

# ── import policies ───────────────────────────────────────────────────────────
from DummyPolicy     import DummyPolicy
from HindsightPolicy import HindsightPolicy
from DLPolicy import DeterministicLookaheadPolicy
import importlib
from TwoSPPolicy import TwoStageSPPolicy
from MSPolicy import MultiStageSPPolicy
from MSPolicy_test import MultiStageSPPolicy as MultiStageSPPolicy_test
from HybridPolicy import HybridPolicy
from ADPPolicy import ADPPolicy  # available in task6/ADPPolicy.py (Rodrigo)


# =============================================================================
# RUN ALL POLICIES
# =============================================================================

print("Running evaluations...")

# np.random.seed(20)
# dummy_costs = run_simulation(DummyPolicy(), num_experiments=100)
# print(f"Dummy Policy  -> avg cost: {np.mean(dummy_costs):.2f}")

#np.random.seed(20)
#hindsight_costs = run_simulation(HindsightPolicy(), num_experiments=100)
#print(f"Hindsight     -> avg cost: {np.mean(hindsight_costs):.2f}")

#np.random.seed(20)
#dl_costs = run_simulation(DeterministicLookaheadPolicy(), num_experiments=100, verbose=True)
#print(f"DL Policy     -> avg cost: {np.mean(dl_costs):.2f}")

np.random.seed(20)
sp2_costs = run_simulation(TwoStageSPPolicy(), num_experiments=100, verbose=True)
print(f"2SP Policy     -> avg cost: {np.mean(sp2_costs):.2f}")

#np.random.seed(20)
#ms_costs_original = run_simulation(MultiStageSPPolicy(L=3, B=[20, 4], S_init=500), num_experiments=100)
#print(f"MS Original    -> avg cost: {np.mean(ms_costs_original):.2f}")

#np.random.seed(20)
#ms_costs_test = run_simulation(MultiStageSPPolicy_test(L=3, B=[20, 4], S_init=500), num_experiments=100)
#print(f"MS Test        -> avg cost: {np.mean(ms_costs_test):.2f}")

#np.random.seed(20)
#hybrid_costs = run_simulation(HybridPolicy(), num_experiments=100)
#print(f"Hybrid Policy  -> avg cost: {np.mean(hybrid_costs):.2f}")

#np.random.seed(20)
#adp_costs = run_simulation(ADPPolicy(), num_experiments=100)
#print(f"ADP Policy     -> avg cost: {np.mean(adp_costs):.2f}")


# =============================================================================
# COLLECT RESULTS
# =============================================================================

results = {
    # "Dummy":      dummy_costs,
    #"Hindsight":  hindsight_costs,
    #"DL":         dl_costs,
    # "2SP":        sp2_costs,
    # "MS-SP":      ms_costs,
    #"Hybrid":     hybrid_costs,
    "ADP":        adp_costs,
    #"MS-Original": ms_costs_original,
    #"MS-Test":     ms_costs_test,
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
