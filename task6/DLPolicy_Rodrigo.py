# -*- coding: utf-8 -*-
"""
DLPolicy_Rodrigo.py — Deterministic Lookahead Policy (Expected Value Policy)
==============================================================================
A special case of the stochastic programming policy that uses a single
scenario: the expected value scenario (noise = 0 in the professor's models).

Steps at each hour t:
  1. Build the expected forecast for the remaining horizon
  2. Solve a deterministic MILP over that single scenario
  3. Return only the here-and-now action (slot 0)
"""

import numpy as np
import sys
import os
from pyomo.environ import (
    ConcreteModel, RangeSet, Set, Var, Objective, Constraint,
    NonNegativeReals, Reals, Binary, SolverFactory, value, minimize
)

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GIVEN_DIR = os.path.join(BASE_DIR, "given")
sys.path.insert(0, GIVEN_DIR)

import v2_SystemCharacteristics as SC
sys.modules['SystemCharacteristics'] = SC

PARAMS = SC.get_fixed_data()


# =============================================================================
#  SECTION 1 — Deterministic forecast  (expected value, noise = 0)
# =============================================================================
# Directly mirrors the professor's process models with noise zeroed out.
#
# Price (from PriceProcessRestaurant.py, noise=0):
#   next = curr + 0.6*(curr - prev) + 0.12*(4.0 - curr),  clipped [0, 12]
#
# Occupancy (from OccupancyProcessRestaurant.py, noise=0):
#   occ1_next = occ1 + 0.25*(35 - occ1) + 0.1*(occ2 - occ1),  clipped [20, 50]
#   occ2_next = occ2 + 0.25*(25 - occ2) + 0.1*(occ1 - occ2),  clipped [10, 30]
# =============================================================================

def build_forecast(state, horizon):
    """
    Returns (prices, occ1s, occ2s), each a list of length `horizon`.
    Index 0 = current known values (already observed).
    Index t > 0 = deterministic forecast for lookahead slot t.
    """
    prices = [0.0] * horizon
    occ1s  = [0.0] * horizon
    occ2s  = [0.0] * horizon

    # Slot 0: current known values
    prices[0] = float(state['price_t'])
    occ1s[0]  = float(state['Occ1'])
    occ2s[0]  = float(state['Occ2'])

    curr_price = float(state['price_t'])
    prev_price = float(state['price_previous'])
    curr_occ1  = float(state['Occ1'])
    curr_occ2  = float(state['Occ2'])

    for t in range(1, horizon):
        # Price: mean-reversion dynamics, noise = 0
        next_price = curr_price + 0.6 * (curr_price - prev_price) + 0.12 * (4.0 - curr_price)
        next_price = float(np.clip(next_price, 0.0, 12.0))

        # Occupancy: mean-reversion + coupling, noise = 0
        next_occ1 = curr_occ1 + 0.25 * (35.0 - curr_occ1) + 0.1 * (curr_occ2 - curr_occ1)
        next_occ2 = curr_occ2 + 0.25 * (25.0 - curr_occ2) + 0.1 * (curr_occ1 - curr_occ2)
        next_occ1 = float(np.clip(next_occ1, 20.0, 50.0))
        next_occ2 = float(np.clip(next_occ2, 10.0, 30.0))

        prices[t] = next_price
        occ1s[t]  = next_occ1
        occ2s[t]  = next_occ2

        prev_price = curr_price
        curr_price = next_price
        curr_occ1  = next_occ1
        curr_occ2  = next_occ2

    return prices, occ1s, occ2s


# =============================================================================
#  SECTION 2 — Slot-0 overrule resolution
# =============================================================================
# Before building the MILP we check the current state and determine which
# slot-0 actions are already forced by the simulator's overrule controllers.
# This mirrors apply_overrule_controllers() in Environment.py exactly.
#
# Returns (p1_fix, p2_fix, v0_fix):
#   None  → variable is free (MILP decides)
#   float → variable is fixed to this value
# =============================================================================

def resolve_slot0_overrules(state, params):
    T1        = float(state['T1'])
    T2        = float(state['T2'])
    H         = float(state['H'])
    vent_cnt  = int(state['vent_counter'])
    u1        = int(state['low_override_r1'])
    u2        = int(state['low_override_r2'])

    P_max  = params['heating_max_power']
    T_low  = params['temp_min_comfort_threshold']
    T_OK   = params['temp_OK_threshold']
    T_high = params['temp_max_comfort_threshold']
    H_high = params['humidity_threshold']
    U_vent = params['vent_min_up_time']

    # ── Room 1 ────────────────────────────────────────────────────────────────
    p1_fix = None
    if T1 < T_low:
        u1 = 1
    if u1 == 1:
        if T1 < T_OK:
            p1_fix = P_max
    if T1 > T_high:          # high override takes priority over everything
        p1_fix = 0.0

    # ── Room 2 ────────────────────────────────────────────────────────────────
    p2_fix = None
    if T2 < T_low:
        u2 = 1
    if u2 == 1:
        if T2 < T_OK:
            p2_fix = P_max
    if T2 > T_high:
        p2_fix = 0.0

    # ── Ventilation ───────────────────────────────────────────────────────────
    v0_fix = None
    if H > H_high:                    # humidity overrule
        v0_fix = 1
    if 1 <= vent_cnt < U_vent:        # inertia: must stay ON
        v0_fix = 1

    return p1_fix, p2_fix, v0_fix, u1, u2


# =============================================================================
#  SECTION 3 — Deterministic MILP
# =============================================================================

def build_and_solve_milp(state, prices, occ1s, occ2s, params):
    L = len(prices)   # remaining slots (= horizon)

    # ── Parameters ────────────────────────────────────────────────────────────
    P_max     = float(params['heating_max_power'])
    P_vent    = float(params['ventilation_power'])
    zeta_exch = float(params['heat_exchange_coeff'])
    zeta_loss = float(params['thermal_loss_coeff'])
    zeta_conv = float(params['heating_efficiency_coeff'])
    zeta_cool = float(params['heat_vent_coeff'])
    zeta_occ  = float(params['heat_occupancy_coeff'])
    eta_occ   = float(params['humidity_occupancy_coeff'])
    eta_vent  = float(params['humidity_vent_coeff'])
    T_low     = float(params['temp_min_comfort_threshold'])
    T_ok      = float(params['temp_OK_threshold'])
    T_high    = float(params['temp_max_comfort_threshold'])
    H_high    = float(params['humidity_threshold'])
    U_vent    = int(params['vent_min_up_time'])
    T_out     = list(params['outdoor_temperature'])
    M_temp    = 50.0
    M_hum     = 100.0

    # ── Current state scalars ─────────────────────────────────────────────────
    T1_init  = float(state['T1'])
    T2_init  = float(state['T2'])
    H_init   = float(state['H'])
    vent_cnt = int(state['vent_counter'])
    t_now    = int(state['current_time'])
    v_prev   = 1 if vent_cnt > 0 else 0

    # Occupancy dict — same layout as milp_solver.py
    occ = {(1, t): occ1s[t] for t in range(L)}
    occ.update({(2, t): occ2s[t] for t in range(L)})

    # ── Slot-0 overrules ──────────────────────────────────────────────────────
    p1_fix, p2_fix, v0_fix, u1_eff, u2_eff = resolve_slot0_overrules(state, params)

    # ── Model ─────────────────────────────────────────────────────────────────
    model   = ConcreteModel()
    model.T = RangeSet(0, L - 1)
    model.R = Set(initialize=[1, 2])

    model.p      = Var(model.R, model.T, domain=NonNegativeReals, bounds=(0, P_max))
    model.Temp   = Var(model.R, model.T, domain=Reals)
    model.H      = Var(model.T, domain=Reals)
    model.v      = Var(model.T, domain=Binary)
    model.s      = Var(model.T, domain=Binary)
    model.y_low  = Var(model.R, model.T, domain=Binary)
    model.y_ok   = Var(model.R, model.T, domain=Binary)
    model.y_high = Var(model.R, model.T, domain=Binary)
    model.u      = Var(model.R, model.T, domain=Binary)

    # ── Fix slot-0 actions from overrules ─────────────────────────────────────
    if p1_fix is not None: model.p[1, 0].fix(p1_fix)
    if p2_fix is not None: model.p[2, 0].fix(p2_fix)
    if v0_fix is not None: model.v[0].fix(float(v0_fix))

    # Ventilation carry-over: remaining forced-ON slots after slot 0
    if vent_cnt > 0:
        for t in range(1, min(U_vent - vent_cnt, L)):
            model.v[t].fix(1.0)

    # ── Objective ─────────────────────────────────────────────────────────────
    def obj_rule(model):
        return sum(
            prices[t] * (sum(model.p[r, t] for r in [1, 2]) + P_vent * model.v[t])
            for t in model.T
        )
    model.obj = Objective(rule=obj_rule, sense=minimize)

    # ── Initial conditions (same as milp_solver.py) ───────────────────────────
    model.init_T1 = Constraint(expr=model.Temp[1, 0] == T1_init)
    model.init_T2 = Constraint(expr=model.Temp[2, 0] == T2_init)
    model.init_H  = Constraint(expr=model.H[0] == H_init)

    # ── Temperature dynamics ──────────────────────────────────────────────────
    def temp_dynamics(model, r, t):
        if t == 0:
            return Constraint.Skip
        r_other = 2 if r == 1 else 1
        return model.Temp[r, t] == (
            model.Temp[r, t-1]
            + zeta_exch * (model.Temp[r_other, t-1] - model.Temp[r, t-1])
            - zeta_loss * (model.Temp[r, t-1] - T_out[t_now + t - 1])
            + zeta_conv * model.p[r, t-1]
            - zeta_cool * model.v[t-1]
            + zeta_occ  * occ[r, t-1]
        )
    model.temp_dynamics = Constraint(model.R, model.T, rule=temp_dynamics)

    # ── Humidity dynamics ─────────────────────────────────────────────────────
    def hum_dynamics(model, t):
        if t == 0:
            return Constraint.Skip
        return model.H[t] == (
            model.H[t-1]
            + eta_occ * sum(occ[r, t-1] for r in [1, 2])
            - eta_vent * model.v[t-1]
        )
    model.hum_dynamics = Constraint(model.T, rule=hum_dynamics)

    # ── Big-M: temperature zone indicators ───────────────────────────────────
    def y_high_upper(model, r, t):
        return model.Temp[r, t] >= T_high - M_temp * (1 - model.y_high[r, t])
    def y_high_lower(model, r, t):
        return model.Temp[r, t] <= T_high + M_temp * model.y_high[r, t]
    model.y_high_upper = Constraint(model.R, model.T, rule=y_high_upper)
    model.y_high_lower = Constraint(model.R, model.T, rule=y_high_lower)

    def y_low_upper(model, r, t):
        return model.Temp[r, t] <= T_low + M_temp * (1 - model.y_low[r, t])
    def y_low_lower(model, r, t):
        return model.Temp[r, t] >= T_low - M_temp * model.y_low[r, t]
    model.y_low_upper = Constraint(model.R, model.T, rule=y_low_upper)
    model.y_low_lower = Constraint(model.R, model.T, rule=y_low_lower)

    def y_ok_upper(model, r, t):
        return model.Temp[r, t] >= T_ok - M_temp * (1 - model.y_ok[r, t])
    def y_ok_lower(model, r, t):
        return model.Temp[r, t] <= T_ok + M_temp * model.y_ok[r, t]
    model.y_ok_upper = Constraint(model.R, model.T, rule=y_ok_upper)
    model.y_ok_lower = Constraint(model.R, model.T, rule=y_ok_lower)

    # ── Override state u ──────────────────────────────────────────────────────
    # At t=0: fix u to the known initial override state
    # At t>0: model the trigger / memory / persist / release logic
    model.u[1, 0].fix(u1_eff)
    model.u[2, 0].fix(u2_eff)

    def overrule_trigger(model, r, t):
        if t == 0: return Constraint.Skip
        return model.u[r, t] >= model.y_low[r, t]
    model.overrule_trigger = Constraint(model.R, model.T, rule=overrule_trigger)

    def overrule_memory(model, r, t):
        if t == 0: return Constraint.Skip
        return model.u[r, t] <= model.u[r, t-1] + model.y_low[r, t]
    model.overrule_memory = Constraint(model.R, model.T, rule=overrule_memory)

    def overrule_persist(model, r, t):
        if t == 0: return Constraint.Skip
        return model.u[r, t] >= model.u[r, t-1] - model.y_ok[r, t]
    model.overrule_persist = Constraint(model.R, model.T, rule=overrule_persist)

    def overrule_deactivate(model, r, t):
        if t == 0: return Constraint.Skip
        return model.u[r, t] <= 1 - model.y_ok[r, t]
    model.overrule_deactivate = Constraint(model.R, model.T, rule=overrule_deactivate)

    def overrule_max(model, r, t):
        if t == 0: return Constraint.Skip   # handled by p1_fix/p2_fix
        return model.p[r, t] >= P_max * model.u[r, t]
    model.overrule_max = Constraint(model.R, model.T, rule=overrule_max)

    def overrule_zero(model, r, t):
        if t == 0: return Constraint.Skip   # handled by p1_fix/p2_fix
        return model.p[r, t] <= P_max * (1 - model.y_high[r, t])
    model.overrule_zero = Constraint(model.R, model.T, rule=overrule_zero)

    # ── Humidity overrule on ventilation ──────────────────────────────────────
    def hum_vent(model, t):
        if t == 0: return Constraint.Skip   # handled by v0_fix
        return model.H[t] <= H_high + M_hum * model.v[t]
    model.hum_vent = Constraint(model.T, rule=hum_vent)

    # ── Ventilation startup detection ─────────────────────────────────────────
    def startup_detect1(model, t):
        return model.s[t] >= model.v[t] - (model.v[t-1] if t > 0 else v_prev)
    def startup_detect2(model, t):
        return model.s[t] <= model.v[t]
    def startup_detect3(model, t):
        if t == 0: return Constraint.Skip
        return model.s[t] <= 1 - model.v[t-1]
    model.startup_detect1 = Constraint(model.T, rule=startup_detect1)
    model.startup_detect2 = Constraint(model.T, rule=startup_detect2)
    model.startup_detect3 = Constraint(model.T, rule=startup_detect3)

    # ── Ventilation minimum up-time ───────────────────────────────────────────
    def vent_uptime(model, t):
        h_end    = min(t + U_vent, L)
        duration = min(U_vent, L - t)
        return sum(model.v[tau] for tau in range(t, h_end)) >= duration * model.s[t]
    model.vent_uptime = Constraint(model.T, rule=vent_uptime)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit']  = 8
    solver.options['MIPGap']     = 1e-3
    solver.options['OutputFlag'] = 0
    solver.options['Threads']    = 2
    solver.solve(model, tee=False)

    # ── Extract here-and-now action (slot 0) ─────────────────────────────────
    try:
        p1 = float(value(model.p[1, 0]))
        p2 = float(value(model.p[2, 0]))
        v  = int(round(float(value(model.v[0]))))
    except Exception:
        p1, p2, v = 0.0, 0.0, 0

    return {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}


# =============================================================================
#  SECTION 4 — Policy class
# =============================================================================

class DLPolicy:
    """Expected Value (Deterministic Lookahead) Policy."""

    def __init__(self):
        self.params = SC.get_fixed_data()

    def select_action(self, state):
        t_now   = int(state['current_time'])
        horizon = int(self.params['num_timeslots']) - t_now

        if horizon <= 0:
            return {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}

        prices, occ1s, occ2s = build_forecast(state, horizon)

        try:
            return build_and_solve_milp(state, prices, occ1s, occ2s, self.params)
        except Exception as e:
            print(f"[DLPolicy] solver failed at t={t_now}: {e}")
            return {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}


# =============================================================================
#  SECTION 5 — Debug / local test
# =============================================================================
if __name__ == "__main__":

    state_normal = {
        "T1": 21.0, "T2": 20.5, "H": 45.0,
        "Occ1": 30.0, "Occ2": 20.0,
        "price_t": 4.0, "price_previous": 3.5,
        "vent_counter": 0, "low_override_r1": 0, "low_override_r2": 0,
        "current_time": 0,
    }

    # ── Test 1: forecast ──────────────────────────────────────────────────────
    print("=" * 50)
    print("TEST 1 — Forecast (horizon=10)")
    print("=" * 50)
    prices, occ1s, occ2s = build_forecast(state_normal, horizon=10)
    print(f"{'t':>3}  {'price':>8}  {'occ1':>8}  {'occ2':>8}")
    for t in range(10):
        print(f"{t:>3}  {prices[t]:>8.3f}  {occ1s[t]:>8.3f}  {occ2s[t]:>8.3f}")

    # ── Test 2: slot-0 overrules ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST 2 — Slot-0 overrule resolution")
    print("=" * 50)
    params = SC.get_fixed_data()

    cases = {
        "normal (no overrules)":       dict(state_normal),
        "T1 cold (< T_low=18)":        {**state_normal, "T1": 16.0},
        "T1 hot  (> T_high=26)":       {**state_normal, "T1": 27.0},
        "low_override_r1 active":      {**state_normal, "low_override_r1": 1, "T1": 20.0},
        "humidity high (H > 70)":      {**state_normal, "H": 75.0},
        "vent inertia (counter=1)":    {**state_normal, "vent_counter": 1},
    }

    for name, s in cases.items():
        p1, p2, v = resolve_slot0_overrules(s, params)
        print(f"  {name:<35}  p1={str(p1):<6}  p2={str(p2):<6}  v={v}")

    # ── Test 3: full policy — single step ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST 3 — Full policy (single select_action call)")
    print("=" * 50)
    policy = DLPolicy()
    action = policy.select_action(state_normal)
    print(f"  Action: {action}")

    # ── Test 4: mid-day call (t=5) ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST 4 — Mid-day call (current_time=5, horizon=5)")
    print("=" * 50)
    state_mid = {**state_normal, "current_time": 5}
    action_mid = policy.select_action(state_mid)
    print(f"  Action: {action_mid}")

    # ── Test 5: edge case — humidity overrule ─────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST 5 — Humidity overrule (H=75 → v must be 1)")
    print("=" * 50)
    state_humid = {**state_normal, "H": 75.0}
    action_humid = policy.select_action(state_humid)
    print(f"  Action (v should=1): {action_humid}")
    assert action_humid["VentilationON"] == 1, "FAIL: ventilation should be forced ON"
    print("  PASS")

    # ── Test 6: edge case — cold room ────────────────────────────────────────
    print("\n" + "=" * 50)
    print("TEST 6 — Cold room (T1=16 → p1 must be P_max=3)")
    print("=" * 50)
    state_cold = {**state_normal, "T1": 16.0}
    action_cold = policy.select_action(state_cold)
    print(f"  Action (p1 should=3): {action_cold}")
    assert action_cold["HeatPowerRoom1"] == 3.0, "FAIL: p1 should be forced to P_max"
    print("  PASS")

    # ── Test 7: deterministic vs stochastic forecast comparison ───────────────
    print("\n" + "=" * 50)
    print("TEST 7 — Deterministic vs Stochastic forecast (price only)")
    print("=" * 50)
    N_paths = 5
    np.random.seed(42)

    print(f"\n{'t':>3}  {'det':>8}  ", end="")
    for i in range(N_paths):
        print(f"{'path'+str(i+1):>8}", end="  ")
    print()

    # generate stochastic paths
    stoch_prices = []
    for _ in range(N_paths):
        path = [float(state_normal['price_t'])]
        curr, prev = state_normal['price_t'], state_normal['price_previous']
        for _ in range(9):
            nxt = curr + 0.6*(curr - prev) + 0.12*(4.0 - curr) + np.random.normal(0, 0.5)
            if nxt < 0 and np.random.rand() > 0.2:
                nxt = np.random.uniform(0, 1.2)
            nxt = float(np.clip(nxt, 0, 12))
            path.append(nxt)
            prev, curr = curr, nxt
        stoch_prices.append(path)

    for t in range(10):
        print(f"{t:>3}  {prices[t]:>8.3f}  ", end="")
        for path in stoch_prices:
            print(f"{path[t]:>8.3f}", end="  ")
        print()
