# -*- coding: utf-8 -*-
import sys
import os
import importlib.util
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(BASE_DIR, "task6"))
sys.path.insert(0, os.path.join(BASE_DIR, "Task1"))
sys.path.insert(0, os.path.join(BASE_DIR, "given"))

import v2_SystemCharacteristics as SC
sys.modules['SystemCharacteristics'] = SC

from Environment import run_simulation, apply_overrule_controllers, compute_next_state, compute_cost
from v2_Checks import check_and_sanitize_action
from HindsightPolicy import HindsightPolicy

PARAMS   = SC.get_fixed_data()
P_max    = PARAMS["heating_max_power"]
PowerMax = {1: P_max, 2: P_max}
T        = int(PARAMS["num_timeslots"])
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_policy(filepath, class_name):
    spec = importlib.util.spec_from_file_location("_mod", filepath)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

PolicyTsilis  = load_policy(os.path.join(BASE_DIR, "task6", "DLPolicy_Tsilis_updated.py"), "DLPolicy")
PolicyRodrigo = load_policy(os.path.join(BASE_DIR, "task6", "DLPolicy_Rodrigo.py"),        "DLPolicy")

# =============================================================================
# Step-by-step episode debug — cada política no seu próprio estado
# =============================================================================

def run_one_episode(policy, prices, price_prev, occ1_row, occ2_row, params):
    state = {
        "T1": params["T1"], "T2": params["T2"], "H": params["H"],
        "Occ1": occ1_row[0], "Occ2": occ2_row[0],
        "price_t": prices[0], "price_previous": price_prev,
        "vent_counter": params["vent_counter"],
        "low_override_r1": params["low_override_r1"],
        "low_override_r2": params["low_override_r2"],
        "current_time": 0,
    }
    log = []
    total = 0.0
    for t in range(T):
        action = check_and_sanitize_action(policy, state, PowerMax)
        eff, lor1, lor2 = apply_overrule_controllers(state, action, params)
        cost = compute_cost(eff, state["price_t"], params)
        total += cost
        log.append({
            "t": t,
            "T1": round(state["T1"], 2), "T2": round(state["T2"], 2),
            "H":  round(state["H"], 1),
            "u1": state["low_override_r1"], "u2": state["low_override_r2"],
            "vc": state["vent_counter"],
            "p1": round(action["HeatPowerRoom1"], 2),
            "p2": round(action["HeatPowerRoom2"], 2),
            "v":  action["VentilationON"],
            "p1e": round(eff["HeatPowerRoom1"], 2),
            "p2e": round(eff["HeatPowerRoom2"], 2),
            "ve":  eff["VentilationON"],
            "price": round(state["price_t"], 2),
            "cost": round(cost, 3),
        })
        next_occ1  = occ1_row[t+1] if t+1 < T else occ1_row[-1]
        next_occ2  = occ2_row[t+1] if t+1 < T else occ2_row[-1]
        next_price = prices[t+1]    if t+1 < T else prices[-1]
        state["low_override_r1"] = lor1
        state["low_override_r2"] = lor2
        state = compute_next_state(state, eff, next_occ1, next_occ2,
                                   next_price, state["price_t"], t, params)
    return log, round(total, 3)

price_df = pd.read_csv(os.path.join(DATA_DIR, "v2_PriceData.csv"))
occ1_df  = pd.read_csv(os.path.join(DATA_DIR, "OccupancyRoom1.csv"))
occ2_df  = pd.read_csv(os.path.join(DATA_DIR, "OccupancyRoom2.csv"))

DAY        = 0
price_row  = price_df.iloc[DAY].values
price_prev = price_row[0]
prices     = price_row[1:]
occ1_row   = occ1_df.iloc[DAY].values
occ2_row   = occ2_df.iloc[DAY].values

log_t, cost_t = run_one_episode(PolicyTsilis(),  prices, price_prev, occ1_row, occ2_row, PARAMS)
log_r, cost_r = run_one_episode(PolicyRodrigo(), prices, price_prev, occ1_row, occ2_row, PARAMS)

# cabeçalho
W = 140
print(f"\n{'='*W}")
print(f"  Episódio dia {DAY}   |   Tsilis custo={cost_t}   Rodrigo custo={cost_r}")
print(f"{'='*W}")
print(f"  {'t':>2}  {'price':>5}  "
      f"{'--- TSILIS ---':^42}  "
      f"{'--- RODRIGO ---':^42}  "
      f"DIFF")
print(f"  {'':>2}  {'':>5}  "
      f"{'T1':>6} {'T2':>6} {'H':>5} {'u1':>2} {'u2':>2} {'vc':>2}  {'p1':>5} {'p2':>5} {'v':>2} {'cost':>6}  "
      f"{'T1':>6} {'T2':>6} {'H':>5} {'u1':>2} {'u2':>2} {'vc':>2}  {'p1':>5} {'p2':>5} {'v':>2} {'cost':>6}  ")
print(f"  {'-'*(W-2)}")

for lt, lr in zip(log_t, log_r):
    diffs = []
    if abs(lt["p1"] - lr["p1"]) > 0.01: diffs.append("p1")
    if abs(lt["p2"] - lr["p2"]) > 0.01: diffs.append("p2")
    if lt["v"] != lr["v"]:               diffs.append("v")
    # estado diferente entre as duas políticas
    if abs(lt["T1"] - lr["T1"]) > 0.01 or abs(lt["T2"] - lr["T2"]) > 0.01:
        diffs.append("STATE")
    flag = ("*** " + "+".join(diffs)) if diffs else ""

    print(f"  {lt['t']:>2}  {lt['price']:>5}  "
          f"{lt['T1']:>6.2f} {lt['T2']:>6.2f} {lt['H']:>5.1f} {lt['u1']:>2} {lt['u2']:>2} {lt['vc']:>2}  "
          f"{lt['p1']:>5} {lt['p2']:>5} {lt['v']:>2} {lt['cost']:>6.3f}  "
          f"{lr['T1']:>6.2f} {lr['T2']:>6.2f} {lr['H']:>5.1f} {lr['u1']:>2} {lr['u2']:>2} {lr['vc']:>2}  "
          f"{lr['p1']:>5} {lr['p2']:>5} {lr['v']:>2} {lr['cost']:>6.3f}  "
          f"{flag}")

print(f"  {'-'*(W-2)}")
print(f"  {'TOTAL':>54}  {cost_t:>6.3f}  {'':>42}  {cost_r:>6.3f}")
print(f"{'='*W}")

# =============================================================================
# Full simulation comparison
# =============================================================================

print(f"\n{'='*60}")
np.random.seed(20)
cost_tsilis   = run_simulation(PolicyTsilis(),    num_experiments=10)
np.random.seed(20)
cost_rodrigo  = run_simulation(PolicyRodrigo(),   num_experiments=10)
np.random.seed(20)
cost_hindsight = run_simulation(HindsightPolicy(), num_experiments=10)
print(f"Simulação 10 dias:")
print(f"  Hindsight: avg={np.mean(cost_hindsight):.2f}  std={np.std(cost_hindsight):.2f}")
print(f"  Tsilis:    avg={np.mean(cost_tsilis):.2f}  std={np.std(cost_tsilis):.2f}")
print(f"  Rodrigo:   avg={np.mean(cost_rodrigo):.2f}  std={np.std(cost_rodrigo):.2f}")
print(f"{'='*60}")
