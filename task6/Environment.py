# -*- coding: utf-8 -*-
"""
Task 6: Simulation Environment
================================
This environment:
  - Loads 100 days of price and occupancy data from CSV files
  - At each hour, gives the policy ONLY the current state
  - After the policy commits to an action, reveals the next values
  - Applies transition dynamics (temperature, humidity, overrule controllers,
    ventilation inertia)
  - Tracks and returns daily costs across all 100 experiments
"""

import numpy as np
import pandas as pd
import sys
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = "/Users/manostsili/Desktop/dtu/courses/decision making under uncertainty /assignment_DC"
GIVEN_DIR   = os.path.join(BASE_DIR, "given")    # professor's files
DATA_DIR    = os.path.join(BASE_DIR, "data")     # CSV files

sys.path.insert(0, GIVEN_DIR)   # so we can import v2_SystemCharacteristics, v2_Checks

import v2_SystemCharacteristics as SC
from v2_Checks import check_and_sanitize_action


# =============================================================================
# TRANSITION DYNAMICS
# =============================================================================

def apply_overrule_controllers(state, action, params):
    """
    Applies the overrule controllers BEFORE the action is used in dynamics.
    Returns the effective (possibly overruled) action and updated override flags.

    Rules:
      Low-temperature overrule (per room):
        - If T < T_low  → activate override (force heater to max)
        - If override active AND T >= T_OK → deactivate override
        - While override active → heater forced to max power
      High-temperature overrule (per room):
        - If T > T_high → force heater to zero for this hour
      Humidity overrule:
        - If H > H_high → force ventilation ON
      Ventilation inertia:
        - If vent_counter in {1, 2} → ventilation must stay ON
    """
    T1 = state["T1"]
    T2 = state["T2"]
    H  = state["H"]
    vent_counter    = state["vent_counter"]
    low_override_r1 = state["low_override_r1"]
    low_override_r2 = state["low_override_r2"]

    T_low  = params["temp_min_comfort_threshold"]
    T_OK   = params["temp_OK_threshold"]
    T_high = params["temp_max_comfort_threshold"]
    H_high = params["humidity_threshold"]
    P_max  = params["heating_max_power"]

    p1 = action["HeatPowerRoom1"]
    p2 = action["HeatPowerRoom2"]
    v  = action["VentilationON"]

    # ── Room 1 low-temperature override ──────────────────────────────────────
    if T1 < T_low:
        low_override_r1 = 1
    if low_override_r1 == 1:
        if T1 >= T_OK:
            low_override_r1 = 0
        else:
            p1 = P_max

    # ── Room 1 high-temperature override ─────────────────────────────────────
    if T1 > T_high:
        p1 = 0

    # ── Room 2 low-temperature override ──────────────────────────────────────
    if T2 < T_low:
        low_override_r2 = 1
    if low_override_r2 == 1:
        if T2 >= T_OK:
            low_override_r2 = 0
        else:
            p2 = P_max

    # ── Room 2 high-temperature override ─────────────────────────────────────
    if T2 > T_high:
        p2 = 0

    # ── Humidity overrule ─────────────────────────────────────────────────────
    if H > H_high:
        v = 1

    # ── Ventilation inertia ───────────────────────────────────────────────────
    if 1 <= vent_counter <= 2:
        v = 1

    effective_action = {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON":  v
    }

    return effective_action, low_override_r1, low_override_r2


def compute_next_state(state, effective_action, next_occ1, next_occ2,
                       next_price, current_price, t, params):
    """
    Applies the physical transition dynamics to compute the next state.

    Temperature dynamics (per room r):
      T_r,t+1 = T_r,t
              + zeta_exch * (T_other,t - T_r,t)
              - zeta_loss * (T_r,t - T_out,t)
              + zeta_conv * p_r,t
              - zeta_cool * v_t
              + zeta_occ  * occ_r,t

    Humidity dynamics:
      H_t+1 = H_t + eta_occ * (occ1_t + occ2_t) - eta_vent * v_t
    """
    T1   = state["T1"]
    T2   = state["T2"]
    H    = state["H"]
    Occ1 = state["Occ1"]
    Occ2 = state["Occ2"]
    vent_counter = state["vent_counter"]

    p1 = effective_action["HeatPowerRoom1"]
    p2 = effective_action["HeatPowerRoom2"]
    v  = effective_action["VentilationON"]

    zeta_exch = params["heat_exchange_coeff"]
    zeta_loss = params["thermal_loss_coeff"]
    zeta_conv = params["heating_efficiency_coeff"]
    zeta_cool = params["heat_vent_coeff"]
    zeta_occ  = params["heat_occupancy_coeff"]
    eta_occ   = params["humidity_occupancy_coeff"]
    eta_vent  = params["humidity_vent_coeff"]
    T_out     = params["outdoor_temperature"][t]

    # ── Temperature updates ───────────────────────────────────────────────────
    T1_next = (T1
               + zeta_exch * (T2 - T1)
               - zeta_loss * (T1 - T_out)
               + zeta_conv * p1
               - zeta_cool * v
               + zeta_occ  * Occ1)

    T2_next = (T2
               + zeta_exch * (T1 - T2)
               - zeta_loss * (T2 - T_out)
               + zeta_conv * p2
               - zeta_cool * v
               + zeta_occ  * Occ2)

    # ── Humidity update ───────────────────────────────────────────────────────
    H_next = H + eta_occ * (Occ1 + Occ2) - eta_vent * v
    H_next = max(H_next, 0.0)

    # ── Ventilation counter ───────────────────────────────────────────────────
    if v == 1:
        vent_counter_next = min(vent_counter + 1, params["vent_min_up_time"])
    else:
        vent_counter_next = 0

    # ── Low-override flags for next step ─────────────────────────────────────
    T_low = params["temp_min_comfort_threshold"]
    T_OK  = params["temp_OK_threshold"]

    low_override_r1_next = state["low_override_r1"]
    if T1_next < T_low:
        low_override_r1_next = 1
    elif T1_next >= T_OK:
        low_override_r1_next = 0

    low_override_r2_next = state["low_override_r2"]
    if T2_next < T_low:
        low_override_r2_next = 1
    elif T2_next >= T_OK:
        low_override_r2_next = 0

    next_state = {
        "T1":              T1_next,
        "T2":              T2_next,
        "H":               H_next,
        "Occ1":            next_occ1,
        "Occ2":            next_occ2,
        "price_t":         next_price,
        "price_previous":  current_price,
        "vent_counter":    vent_counter_next,
        "low_override_r1": low_override_r1_next,
        "low_override_r2": low_override_r2_next,
        "current_time":    t + 1,
    }

    return next_state


def compute_cost(effective_action, current_price, params):
    """
    Cost = price * (p1 + p2 + P_vent * v)
    """
    p1     = effective_action["HeatPowerRoom1"]
    p2     = effective_action["HeatPowerRoom2"]
    v      = effective_action["VentilationON"]
    P_vent = params["ventilation_power"]
    return current_price * (p1 + p2 + P_vent * v)


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================

def run_simulation(policy, num_experiments=100, verbose=False):
    """
    Runs the simulation for num_experiments days.

    Parameters
    ----------
    policy          : object with a .select_action(state) method
    num_experiments : int, number of days to simulate (max 100)
    verbose         : bool, print per-day cost if True

    Returns
    -------
    daily_costs : list of floats, one total cost per day
    """
    params   = SC.get_fixed_data()
    P_max    = params["heating_max_power"]
    PowerMax = {1: P_max, 2: P_max}
    T        = params["num_timeslots"]

    # ── Load CSV data ─────────────────────────────────────────────────────────
    price_df = pd.read_csv(os.path.join(DATA_DIR, "v2_PriceData.csv"))
    occ1_df  = pd.read_csv(os.path.join(DATA_DIR, "OccupancyRoom1.csv"))
    occ2_df  = pd.read_csv(os.path.join(DATA_DIR, "OccupancyRoom2.csv"))

    daily_costs = []

    for day in range(num_experiments):

        # ── Extract this day's sequences ──────────────────────────────────────
        price_row  = price_df.iloc[day].values   # shape (11,)
        price_prev = price_row[0]
        prices     = price_row[1:]               # shape (10,)

        occ1_row = occ1_df.iloc[day].values      # shape (10,)
        occ2_row = occ2_df.iloc[day].values      # shape (10,)

        # ── Initialise state ──────────────────────────────────────────────────
        state = {
            "T1":              params["T1"],
            "T2":              params["T2"],
            "H":               params["H"],
            "Occ1":            occ1_row[0],
            "Occ2":            occ2_row[0],
            "price_t":         prices[0],
            "price_previous":  price_prev,
            "vent_counter":    params["vent_counter"],
            "low_override_r1": params["low_override_r1"],
            "low_override_r2": params["low_override_r2"],
            "current_time":    0,
        }

        daily_cost = 0.0

        for t in range(T):

            # Step 1: policy commits to action
            action = check_and_sanitize_action(policy, state, PowerMax)

            # Step 2: apply overrule controllers
            effective_action, low_r1, low_r2 = apply_overrule_controllers(
                state, action, params
            )
            state["low_override_r1"] = low_r1
            state["low_override_r2"] = low_r2

            # Step 3: compute cost on effective action
            current_price = prices[t]
            cost = compute_cost(effective_action, current_price, params)
            daily_cost += cost

            # Step 4: environment reveals next values
            if t < T - 1:
                next_occ1  = occ1_row[t + 1]
                next_occ2  = occ2_row[t + 1]
                next_price = prices[t + 1]
            else:
                next_occ1  = occ1_row[t]
                next_occ2  = occ2_row[t]
                next_price = prices[t]

            # Step 5: compute next state via dynamics
            next_state = compute_next_state(
                state, effective_action,
                next_occ1, next_occ2, next_price, current_price,
                t, params
            )

            state = next_state

        daily_costs.append(daily_cost)

        if verbose:
            print(f"Day {day+1:3d}: cost = {daily_cost:.4f}")

    return daily_costs


# =============================================================================
# REPORTING HELPER
# =============================================================================

def evaluate_policy(policy, policy_name="Policy", num_experiments=100, verbose=False):
    """Runs simulation and prints a summary report."""
    costs = run_simulation(policy, num_experiments=num_experiments, verbose=verbose)
    avg   = np.mean(costs)
    std   = np.std(costs)
    print(f"\n{'='*50}")
    print(f"  Policy : {policy_name}")
    print(f"  Days   : {num_experiments}")
    print(f"  Avg daily cost : {avg:.4f}")
    print(f"  Std daily cost : {std:.4f}")
    print(f"  Min daily cost : {min(costs):.4f}")
    print(f"  Max daily cost : {max(costs):.4f}")
    print(f"{'='*50}\n")
    return costs

# Environment is complete.
# Import this file in Evaluate.py and plug in any policy via run_simulation(policy)