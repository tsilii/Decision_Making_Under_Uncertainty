# -*- coding: utf-8 -*-
"""
debug_trajectory.py
====================
Compares the MILP's planned trajectory against the environment's actual
trajectory for a single day, step by step.

BUG FIXED: the previous version called select_action() twice for day 0
(once before the loop, once inside at t=0), causing the policy to solve
day 1's MILP while simulating day 0's data.  Now we call _solve_milp
directly to extract the plan without touching current_day.
"""

import numpy as np
import pandas as pd
import sys, os

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GIVEN_DIR = os.path.join(BASE_DIR, "given")
TASK1_DIR = os.path.join(BASE_DIR, "Task1")
sys.path.insert(0, GIVEN_DIR)
sys.path.insert(0, TASK1_DIR)

import v2_SystemCharacteristics as SC
from HindsightPolicy import HindsightPolicy
from Environment import apply_overrule_controllers, compute_next_state, compute_cost

DAY = 0  # change to test different days

# ── load data ──────────────────────────────────────────────────────────────────
params   = SC.get_fixed_data()
T        = params["num_timeslots"]
price_df = pd.read_csv(os.path.join(GIVEN_DIR, "PriceData.csv"))
occ1_df  = pd.read_csv(os.path.join(GIVEN_DIR, "OccupancyRoom1.csv"))
occ2_df  = pd.read_csv(os.path.join(GIVEN_DIR, "OccupancyRoom2.csv"))

prices   = price_df.iloc[DAY].values[:T]
occ1_row = occ1_df.iloc[DAY].values
occ2_row = occ2_df.iloc[DAY].values

# ── solve MILP directly for day DAY (no current_day side-effects) ──────────────
policy = HindsightPolicy()
policy._solve_milp(prices, occ1_row, occ2_row)

milp_p1 = list(policy.planned_p1)
milp_p2 = list(policy.planned_p2)
milp_v  = list(policy.planned_v)

# ── reconstruct MILP's expected trajectory via dynamics (no overrule) ──────────
milp_T1 = [params["T1"]]
milp_T2 = [params["T2"]]
milp_H  = [params["H"]]

zeta_exch = params["heat_exchange_coeff"]
zeta_loss = params["thermal_loss_coeff"]
zeta_conv = params["heating_efficiency_coeff"]
zeta_cool = params["heat_vent_coeff"]
zeta_occ  = params["heat_occupancy_coeff"]
eta_occ   = params["humidity_occupancy_coeff"]
eta_vent  = params["humidity_vent_coeff"]
T_out_arr = params["outdoor_temperature"]

for t in range(T - 1):
    T1 = milp_T1[-1]; T2 = milp_T2[-1]; H = milp_H[-1]
    p1 = milp_p1[t];  p2 = milp_p2[t];  v = milp_v[t]
    T_out = T_out_arr[t]
    milp_T1.append(T1 + zeta_exch*(T2-T1) - zeta_loss*(T1-T_out) + zeta_conv*p1 - zeta_cool*v + zeta_occ*occ1_row[t])
    milp_T2.append(T2 + zeta_exch*(T1-T2) - zeta_loss*(T2-T_out) + zeta_conv*p2 - zeta_cool*v + zeta_occ*occ2_row[t])
    milp_H.append(max(H + eta_occ*(occ1_row[t]+occ2_row[t]) - eta_vent*v, 0.0))

# ── simulate environment applying the MILP's planned actions ───────────────────
state = {
    "T1": params["T1"], "T2": params["T2"], "H": params["H"],
    "Occ1": occ1_row[0], "Occ2": occ2_row[0],
    "price_t": prices[0], "price_previous": prices[0],
    "vent_counter": 0, "low_override_r1": 0, "low_override_r2": 0,
    "current_time": 0,
}

total_milp_cost = 0.0
total_env_cost  = 0.0

print(f"\nDay {DAY} — step-by-step comparison")
print(f"{'t':>3}  {'mT1':>7} {'eT1':>7}  {'mT2':>7} {'eT2':>7}  {'mH':>7} {'eH':>7}  "
      f"{'mp1':>5} {'ep1':>5}  {'mp2':>5} {'ep2':>5}  {'mv':>3} {'ev':>3}  "
      f"{'price':>6}  {'mCost':>7} {'eCost':>7}")
print("-" * 115)

for t in range(T):
    raw_action = {
        "HeatPowerRoom1": milp_p1[t],
        "HeatPowerRoom2": milp_p2[t],
        "VentilationON":  milp_v[t],
    }

    eff_action, low_r1, low_r2 = apply_overrule_controllers(state, raw_action, params)
    state["low_override_r1"] = low_r1
    state["low_override_r2"] = low_r2

    ep1 = eff_action["HeatPowerRoom1"]
    ep2 = eff_action["HeatPowerRoom2"]
    ev  = eff_action["VentilationON"]

    cost_env  = compute_cost(eff_action,  prices[t], params)
    cost_milp = prices[t] * (milp_p1[t] + milp_p2[t] + params["ventilation_power"] * milp_v[t])
    total_env_cost  += cost_env
    total_milp_cost += cost_milp

    flag = " <-- DIVERGES" if (abs(milp_p1[t]-ep1)>0.01 or abs(milp_p2[t]-ep2)>0.01 or milp_v[t]!=ev) else ""

    print(f"{t:>3}  "
          f"{milp_T1[t]:>7.3f} {state['T1']:>7.3f}  "
          f"{milp_T2[t]:>7.3f} {state['T2']:>7.3f}  "
          f"{milp_H[t]:>7.3f} {state['H']:>7.3f}  "
          f"{milp_p1[t]:>5.2f} {ep1:>5.2f}  "
          f"{milp_p2[t]:>5.2f} {ep2:>5.2f}  "
          f"{milp_v[t]:>3} {ev:>3}  "
          f"{prices[t]:>6.3f}  "
          f"{cost_milp:>7.4f} {cost_env:>7.4f}{flag}")

    if t < T - 1:
        next_occ1  = occ1_row[t+1]; next_occ2 = occ2_row[t+1]; next_price = prices[t+1]
    else:
        next_occ1  = occ1_row[t];   next_occ2 = occ2_row[t];   next_price = prices[t]

    next_state = compute_next_state(state, eff_action, next_occ1, next_occ2, next_price, prices[t], t, params)
    state = next_state

print("-" * 115)
print(f"\nMILP objective (planned):  {total_milp_cost:.4f}")
print(f"Environment cost (actual): {total_env_cost:.4f}")
print(f"Difference:                {total_env_cost - total_milp_cost:.4f}  "
      f"({100*(total_env_cost-total_milp_cost)/max(total_milp_cost,1e-9):.1f}%)")

# ── print overrule state at each step for extra context ───────────────────────
print(f"\n{'t':>3}  {'T_low':>6}  {'mT1':>7} {'mT2':>7}  override_r1  override_r2  note")
state2 = {
    "T1": params["T1"], "T2": params["T2"], "H": params["H"],
    "Occ1": occ1_row[0], "Occ2": occ2_row[0],
    "price_t": prices[0], "price_previous": prices[0],
    "vent_counter": 0, "low_override_r1": 0, "low_override_r2": 0,
    "current_time": 0,
}
for t in range(T):
    raw_action = {"HeatPowerRoom1": milp_p1[t], "HeatPowerRoom2": milp_p2[t], "VentilationON": milp_v[t]}
    eff_action, low_r1, low_r2 = apply_overrule_controllers(state2, raw_action, params)
    state2["low_override_r1"] = low_r1
    state2["low_override_r2"] = low_r2
    note = ""
    if state2["T1"] < params["temp_min_comfort_threshold"]: note += " T1<Tlow!"
    if state2["T2"] < params["temp_min_comfort_threshold"]: note += " T2<Tlow!"
    if state2["H"]  > params["humidity_threshold"]:         note += " H>Hhigh!"
    print(f"{t:>3}  {params['temp_min_comfort_threshold']:>6.1f}  "
          f"{state2['T1']:>7.3f} {state2['T2']:>7.3f}  "
          f"{low_r1:>11}  {low_r2:>11}{note}")
    if t < T - 1:
        next_occ1  = occ1_row[t+1]; next_occ2 = occ2_row[t+1]; next_price = prices[t+1]
    else:
        next_occ1  = occ1_row[t];   next_occ2 = occ2_row[t];   next_price = prices[t]
    next_state = compute_next_state(state2, eff_action, next_occ1, next_occ2, next_price, prices[t], t, params)
    state2 = next_state
