# -*- coding: utf-8 -*-
"""
================================================================================
  DLPolicy.py  —  Deterministic Lookahead Policy  (fully corrected)
================================================================================
  All bugs from BOTH analysis rounds are fixed here.

  From analysis round 1:
    BUG #1  [CRITICAL] Temp[r,0] did not depend on actions → slot-0 was free.
            Fixed: state variables start at t=1, T1_0/T2_0 are scalars,
            Temp[r,1] is written as a function of p[r,0] and v[0].

    BUG #3  u[r,0] was a Pyomo variable that could contradict u1_0/u2_0.
            Fixed: u[r,t] starts at t=1. u1_0/u2_0 are scalars used only
            as the "previous" value inside t=1 constraints.

    BUG #6  MC forecast was biased by non-linear clipping and negative-price
            resample rule.
            Fixed: pure deterministic mean-propagation, zero RNG calls.

    BUG #9  Unseeded RNG was polluting the global numpy state.
            Fixed: no RNG calls at all in the forecast.

    BUG #10 Big-M was 100/200 — too large, slows LP relaxation.
            Fixed: M_temp=50, M_hum=100.

  From analysis round 2:
    BUG F   Slot-0 humidity overrule was not enforced: if H_0 > H_high the
            simulator forces v[0]=1 but the MILP could choose v[0]=0 (then
            the H[1] constraint self-corrects but with one-step delay).
            Fixed: explicit v[0].fix(1) when H_0 > H_high, mirroring the
            simulator's apply_overrule_controllers.

    BUG G   Slot-0 low-temp overrule was only handled when u1_0==1, missing
            the case T1_0 <= T_low and u1_0==0 (simulator fires overrule
            immediately; MILP let p[1,0]=0 and under-counted the bill).
            Fixed: check T_r,0 <= T_low directly, same as simulator.

    BUG H   Strict vs non-strict inequality mismatch (simulator uses
            T > T_High strict; code used >=).
            Fixed: T1_0 > T_high (strict) for the high-temp fix.

  NOTE on BUG D (reporting MILP predicted cost vs realised cost):
    This is an environment/evaluator bug, NOT a policy bug. The policy
    returns actions; the environment measures cost. Verify your Environment
    sums:  cost_t = price_t * (applied_p1 + applied_p2 + P_vent * applied_v)
    using the POST-overrule applied values, not the commanded values.
    This policy cannot fix that — check your Environment.py.

  NOTE on BUG K (RNG coupling):
    Removed all np.random calls from this file. The simulator's RNG is no
    longer perturbed by the policy. Confirm DL and Hindsight are tested
    on the SAME pre-generated time series (mode="csv" guarantees this).
================================================================================
"""

import numpy as np
import pyomo.environ as pyo
import sys
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = "/Users/manostsili/Desktop/dtu/courses/decision making under uncertainty /assignment_DC"
GIVEN_DIR = os.path.join(BASE_DIR, "given")
sys.path.insert(0, GIVEN_DIR)

import v2_SystemCharacteristics as SC
sys.modules['SystemCharacteristics'] = SC

PARAMS = SC.get_fixed_data()


# =============================================================================
#  SECTION 1 :  Deterministic forecast  (BUG #6, #9 fixed)
# =============================================================================
# Pure mean-propagation. Zero noise. Zero RNG. No bias from clipping.
#
# Price mean dynamics (from PriceProcessRestaurant, noise=0):
#   next = curr + 0.6*(curr - prev) + 0.12*(4.0 - curr),  clipped [0,12]
#
# Occupancy mean dynamics (from OccupancyProcessRestaurant, noise=0):
#   occ1_next = occ1 + 0.25*(35-occ1) + 0.1*(occ2-occ1),  clipped [20,50]
#   occ2_next = occ2 + 0.25*(25-occ2) + 0.1*(occ1-occ2),  clipped [10,30]
# =============================================================================

def build_forecast(state, horizon):
    """
    Returns (prices, occ1s, occ2s), each a list of length `horizon`.
    Index 0 = current known values (slot 0, already observed).
    Index t = deterministic forecast for lookahead slot t.
    """
    prices = [0.0] * horizon
    occ1s  = [0.0] * horizon
    occ2s  = [0.0] * horizon

    prices[0] = state['price_t']
    occ1s[0]  = state['Occ1']
    occ2s[0]  = state['Occ2']

    curr_price = state['price_t']
    prev_price = state['price_previous']
    curr_occ1  = state['Occ1']
    curr_occ2  = state['Occ2']

    for t in range(1, horizon):
        next_price = (curr_price
                      + 0.6  * (curr_price - prev_price)
                      + 0.12 * (4.0 - curr_price))
        next_price = float(np.clip(next_price, 0.0, 12.0))

        next_occ1  = (curr_occ1
                      + 0.25 * (35.0 - curr_occ1)
                      + 0.1  * (curr_occ2 - curr_occ1))
        next_occ2  = (curr_occ2
                      + 0.25 * (25.0 - curr_occ2)
                      + 0.1  * (curr_occ1 - curr_occ2))
        next_occ1  = float(np.clip(next_occ1, 20.0, 50.0))
        next_occ2  = float(np.clip(next_occ2, 10.0, 30.0))

        prices[t] = next_price
        occ1s[t]  = next_occ1
        occ2s[t]  = next_occ2

        prev_price = curr_price
        curr_price = next_price
        curr_occ1  = next_occ1
        curr_occ2  = next_occ2

    return prices, occ1s, occ2s


# =============================================================================
#  SECTION 2 :  Slot-0 overrule resolution  (BUG G, H, F fixed)
# =============================================================================
# Mirrors apply_overrule_controllers() in the simulator EXACTLY.
# Must be called before building the MILP so p[r,0] and v[0] can be fixed.
#
# Priority order (from the assignment solution PDF):
#   1. T_r,0 > T_high  (strict)  →  p[r,0] = 0
#   2. T_r,0 <= T_low  OR u_r,0=1  →  p[r,0] = P_max   (BUG G fix)
#   3. H_0 > H_high               →  v[0]   = 1         (BUG F fix)
#   4. vent_counter in {1,2}      →  v[0]   = 1
# =============================================================================

def resolve_slot0_overrules(state, params):
    """
    Returns (p1_fix, p2_fix, v0_fix) where:
      None   = free variable (MILP decides)
      float  = fixed to this value
    """
    T1_0     = float(state['T1'])
    T2_0     = float(state['T2'])
    H_0      = float(state['H'])
    u1_0     = int(state['low_override_r1'])
    u2_0     = int(state['low_override_r2'])
    vent_cnt = int(state['vent_counter'])

    P_max   = params['heating_max_power']
    T_high  = params['temp_max_comfort_threshold']
    T_low   = params['temp_min_comfort_threshold']
    H_high  = params['humidity_threshold']
    U_vent  = params['vent_min_up_time']

    # ── Room 1 heating ────────────────────────────────────────────────────────
    if T1_0 > T_high:                      # BUG H fix: strict inequality
        p1_fix = 0.0
    elif T1_0 <= T_low or u1_0 == 1:      # BUG G fix: check temperature too
        p1_fix = P_max
    else:
        p1_fix = None                      # MILP decides

    # ── Room 2 heating ────────────────────────────────────────────────────────
    if T2_0 > T_high:
        p2_fix = 0.0
    elif T2_0 <= T_low or u2_0 == 1:
        p2_fix = P_max
    else:
        p2_fix = None

    # ── Ventilation ───────────────────────────────────────────────────────────
    if H_0 > H_high:                       # BUG F fix: humidity overrule
        v0_fix = 1
    elif 1 <= vent_cnt < U_vent:           # carry-over inertia
        v0_fix = 1
    else:
        v0_fix = None                      # MILP decides

    return p1_fix, p2_fix, v0_fix


# =============================================================================
#  SECTION 3 :  Deterministic lookahead MILP  (BUG #1, #3, #10 fixed)
# =============================================================================

def build_and_solve_milp(state, prices, occ1s, occ2s, params):
    """
    Builds and solves the deterministic lookahead MILP.

    Variable layout after BUG #1 fix:
      Action variables:  p[r,t], v[t], s_vent[t]  for t = 0 .. L-1
      State variables:   Temp[r,t], H[t]           for t = 1 .. L
                         (state AFTER applying action t-1)
      T1_0, T2_0, H_0 are plain Python scalars — NOT Pyomo variables.

    This means Temp[r,1] is written as:
        T1_0 + dynamics_coeffs + zeta_conv*p[r,0] - zeta_cool*v[0] + ...
    so slot-0 actions are genuinely in the objective-constraint system.
    """

    L = len(prices)   # remaining horizon length

    # ── Parameters ────────────────────────────────────────────────────────────
    P_max     = params['heating_max_power']
    P_vent    = params['ventilation_power']
    zeta_exch = params['heat_exchange_coeff']
    zeta_loss = params['thermal_loss_coeff']
    zeta_conv = params['heating_efficiency_coeff']
    zeta_cool = params['heat_vent_coeff']
    zeta_occ  = params['heat_occupancy_coeff']
    eta_occ   = params['humidity_occupancy_coeff']
    eta_vent  = params['humidity_vent_coeff']
    T_low     = params['temp_min_comfort_threshold']
    T_ok      = params['temp_OK_threshold']
    T_high    = params['temp_max_comfort_threshold']
    H_high    = params['humidity_threshold']
    U_vent    = params['vent_min_up_time']
    Tout      = params['outdoor_temperature']

    M_temp = 50.0    # BUG #10 fix
    M_hum  = 100.0   # BUG #10 fix

    # ── Current state scalars ──────────────────────────────────────────────────
    T1_0     = float(state['T1'])
    T2_0     = float(state['T2'])
    H_0      = float(state['H'])
    u1_0     = int(state['low_override_r1'])
    u2_0     = int(state['low_override_r2'])
    vent_cnt = int(state['vent_counter'])
    t_now    = int(state['current_time'])
    v_prev   = 1 if vent_cnt > 0 else 0

    # ── Resolve slot-0 overrules BEFORE building the model ────────────────────
    p1_fix, p2_fix, v0_fix = resolve_slot0_overrules(state, params)

    # ── Pyomo model ────────────────────────────────────────────────────────────
    m = pyo.ConcreteModel()

    m.R    = pyo.Set(initialize=[1, 2])
    m.Tact = pyo.Set(initialize=range(L),      ordered=True)  # 0 .. L-1
    m.Tsta = pyo.Set(initialize=range(1, L+1), ordered=True)  # 1 .. L

    # ── Action variables ───────────────────────────────────────────────────────
    m.p      = pyo.Var(m.R, m.Tact, domain=pyo.NonNegativeReals, bounds=(0, P_max))
    m.v      = pyo.Var(m.Tact, domain=pyo.Binary)
    m.s_vent = pyo.Var(m.Tact, domain=pyo.Binary)

    # ── State variables: start at t=1 (BUG #1 fix) ───────────────────────────
    m.Temp  = pyo.Var(m.R, m.Tsta, domain=pyo.Reals)
    m.H     = pyo.Var(m.Tsta,      domain=pyo.Reals)

    # ── Binary indicators for t=1..L ──────────────────────────────────────────
    m.yLow  = pyo.Var(m.R, m.Tsta, domain=pyo.Binary)
    m.yOK   = pyo.Var(m.R, m.Tsta, domain=pyo.Binary)
    m.yHigh = pyo.Var(m.R, m.Tsta, domain=pyo.Binary)
    # u[r,t] for t=1..L only — u at t=0 is the scalar u1_0/u2_0 (BUG #3 fix)
    m.u     = pyo.Var(m.R, m.Tsta, domain=pyo.Binary)

    # ── Fix slot-0 actions where overrules apply ───────────────────────────────
    if p1_fix is not None:
        m.p[1, 0].fix(p1_fix)
    if p2_fix is not None:
        m.p[2, 0].fix(p2_fix)
    if v0_fix is not None:
        m.v[0].fix(float(v0_fix))

    # ── Objective: minimise total cost over remaining horizon ──────────────────
    def obj_rule(m):
        return sum(
            prices[t] * (P_vent * m.v[t] + m.p[1, t] + m.p[2, t])
            for t in m.Tact
        )
    m.Obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ── Temperature dynamics (BUG #1 fix) ─────────────────────────────────────
    # Temp[r, t] = temperature at the START of hour t_now+t (after action t-1).
    # At t=1: previous temperature is the known scalar T1_0/T2_0.
    # At t>1: previous temperature is the variable Temp[r, t-1].
    def temp_dyn(m, r, t):
        r_other  = 2 if r == 1 else 1
        t_abs    = t_now + t - 1    # index into outdoor_temperature array

        if t == 1:
            T_prev       = T1_0 if r == 1 else T2_0
            T_other_prev = T2_0 if r == 1 else T1_0
            occ_prev     = occ1s[0] if r == 1 else occ2s[0]
        else:
            T_prev       = m.Temp[r,       t - 1]
            T_other_prev = m.Temp[r_other, t - 1]
            occ_prev     = occ1s[t - 1] if r == 1 else occ2s[t - 1]

        return m.Temp[r, t] == (
            T_prev
            + zeta_exch * (T_other_prev - T_prev)
            - zeta_loss * (T_prev - Tout[t_abs])
            + zeta_conv * m.p[r, t - 1]
            - zeta_cool * m.v[t - 1]
            + zeta_occ  * occ_prev
        )
    m.TempDyn = pyo.Constraint(m.R, m.Tsta, rule=temp_dyn)

    # ── Humidity dynamics ──────────────────────────────────────────────────────
    def hum_dyn(m, t):
        H_prev    = H_0      if t == 1 else m.H[t - 1]
        occ1_prev = occ1s[0] if t == 1 else occ1s[t - 1]
        occ2_prev = occ2s[0] if t == 1 else occ2s[t - 1]
        return m.H[t] == (
            H_prev
            + eta_occ  * (occ1_prev + occ2_prev)
            - eta_vent * m.v[t - 1]
        )
    m.HumDyn = pyo.Constraint(m.Tsta, rule=hum_dyn)

    # ── Big-M indicators (BUG #10: smaller M) ─────────────────────────────────

    def yHigh_lb(m, r, t):
        return m.Temp[r, t] >= T_high - M_temp * (1 - m.yHigh[r, t])
    def yHigh_ub(m, r, t):
        return m.Temp[r, t] <= T_high + M_temp * m.yHigh[r, t]
    m.YHighLB = pyo.Constraint(m.R, m.Tsta, rule=yHigh_lb)
    m.YHighUB = pyo.Constraint(m.R, m.Tsta, rule=yHigh_ub)

    def yLow_ub(m, r, t):
        return m.Temp[r, t] <= T_low + M_temp * (1 - m.yLow[r, t])
    def yLow_lb(m, r, t):
        return m.Temp[r, t] >= T_low - M_temp * m.yLow[r, t]
    m.YLowUB = pyo.Constraint(m.R, m.Tsta, rule=yLow_ub)
    m.YLowLB = pyo.Constraint(m.R, m.Tsta, rule=yLow_lb)

    def yOK_lb(m, r, t):
        return m.Temp[r, t] >= T_ok - M_temp * (1 - m.yOK[r, t])
    def yOK_ub(m, r, t):
        return m.Temp[r, t] <= T_ok + M_temp * m.yOK[r, t]
    m.YOKLB = pyo.Constraint(m.R, m.Tsta, rule=yOK_lb)
    m.YOKUB = pyo.Constraint(m.R, m.Tsta, rule=yOK_ub)

    # ── Overrule controller for t = 1..L (BUG #3 fix) ─────────────────────────
    # u[r,t] is the overrule state AFTER applying action t-1.
    # u at slot 0 = scalar u1_0/u2_0, used only inside the t=1 rules.

    def u_trigger(m, r, t):
        return m.u[r, t] >= m.yLow[r, t]
    m.UTrigger = pyo.Constraint(m.R, m.Tsta, rule=u_trigger)

    def u_persist(m, r, t):
        u_prev = (u1_0 if r == 1 else u2_0) if t == 1 else m.u[r, t - 1]
        return m.u[r, t] <= u_prev + m.yLow[r, t]
    m.UPersist = pyo.Constraint(m.R, m.Tsta, rule=u_persist)

    def u_rel_lb(m, r, t):
        u_prev = (u1_0 if r == 1 else u2_0) if t == 1 else m.u[r, t - 1]
        return m.u[r, t] >= u_prev - m.yOK[r, t]
    m.URelLB = pyo.Constraint(m.R, m.Tsta, rule=u_rel_lb)

    # Skip t=1 so the known u1_0/u2_0 is never contradicted (BUG #3 fix)
    def u_rel_ub(m, r, t):
        if t == 1:
            return pyo.Constraint.Skip
        return m.u[r, t] <= 1 - m.yOK[r, t]
    m.URelUB = pyo.Constraint(m.R, m.Tsta, rule=u_rel_ub)

    # ── Overrule forces on actions t=1..L-1 ───────────────────────────────────
    # Action t is driven by overrule state at slot t (= state after action t-1).
    # Slot-0 actions are already handled by the fixes above.
    def high_force_off(m, r, t):
        if t == 0:
            return pyo.Constraint.Skip   # handled by p1_fix/p2_fix
        return m.p[r, t] <= P_max * (1 - m.yHigh[r, t])
    m.HighForceOff = pyo.Constraint(m.R, m.Tact, rule=high_force_off)

    def low_force_max(m, r, t):
        if t == 0:
            return pyo.Constraint.Skip   # handled by p1_fix/p2_fix
        return m.p[r, t] >= P_max * m.u[r, t]
    m.LowForceMax = pyo.Constraint(m.R, m.Tact, rule=low_force_max)

    # ── Humidity-triggered ventilation for t=1..L (BUG F fix) ─────────────────
    # H[t] > H_high forces v[t-1]=1.
    # H[1] > H_high forces v[0]=1 — but v[0] is already fixed via v0_fix
    # if H_0 > H_high, so this constraint is consistent.
    def hum_force_vent(m, t):
        return m.H[t] <= H_high + M_hum * m.v[t - 1]
    m.HumForceVent = pyo.Constraint(m.Tsta, rule=hum_force_vent)

    # ── Ventilation startup detection ──────────────────────────────────────────
    def s_ge_diff(m, t):
        v_prev_t = float(v_prev) if t == 0 else m.v[t - 1]
        return m.s_vent[t] >= m.v[t] - v_prev_t
    def s_le_v(m, t):
        return m.s_vent[t] <= m.v[t]
    def s_le_notvprev(m, t):
        # If v[0] is fixed to 1 (carry-over), s_vent[0]=0 is enforced here
        v_prev_t = float(v_prev) if t == 0 else m.v[t - 1]
        return m.s_vent[t] <= 1 - v_prev_t
    m.SDiff       = pyo.Constraint(m.Tact, rule=s_ge_diff)
    m.SLeV        = pyo.Constraint(m.Tact, rule=s_le_v)
    m.SLeNotVPrev = pyo.Constraint(m.Tact, rule=s_le_notvprev)

    # ── Ventilation minimum up-time ────────────────────────────────────────────
    def vent_uptime(m, t):
        h_end    = min(t + U_vent, L)
        duration = h_end - t
        return sum(m.v[tau] for tau in range(t, h_end)) >= duration * m.s_vent[t]
    m.VentUptime = pyo.Constraint(m.Tact, rule=vent_uptime)

    # ── Ventilation carry-over (constraint-based, no fix()) ───────────────────
    # If vent_cnt > 0 but v0_fix has not already fixed v[0]=1 (it will have),
    # enforce remaining forced-ON hours via constraints.
    # We always use constraints for t>0 carry-over hours.
    if 0 < vent_cnt < U_vent:
        remaining_forced = U_vent - vent_cnt
        def carry_rule(m, t):
            if t < remaining_forced:
                return m.v[t] == 1
            return pyo.Constraint.Skip
        m.VentCarry = pyo.Constraint(m.Tact, rule=carry_rule)

    # ── Solve ──────────────────────────────────────────────────────────────────
    solver = pyo.SolverFactory('gurobi')
    solver.options['TimeLimit']  = 8
    solver.options['MIPGap']     = 1e-3
    solver.options['OutputFlag'] = 0
    solver.options['Threads']    = 2
    solver.solve(m, tee=False)

    # ── Extract here-and-now action ────────────────────────────────────────────
    try:
        p1 = float(pyo.value(m.p[1, 0]))
        p2 = float(pyo.value(m.p[2, 0]))
        v  = int(round(float(pyo.value(m.v[0]))))
    except Exception:
        p1, p2, v = 0.0, 0.0, 0

    return {
        "HeatPowerRoom1": p1,
        "HeatPowerRoom2": p2,
        "VentilationON":  v,
    }


# =============================================================================
#  SECTION 4 :  Policy class
# =============================================================================

class DLPolicy:
    """
    Deterministic Lookahead Policy.
    Implements the select_action(state) interface required by the environment.
    """

    def __init__(self):
        self.params = SC.get_fixed_data()

    def select_action(self, state):
        params  = self.params
        t_now   = int(state['current_time'])
        horizon = int(params['num_timeslots']) - t_now

        if horizon <= 0:
            return {"HeatPowerRoom1": 0.0,
                    "HeatPowerRoom2": 0.0,
                    "VentilationON":  0}

        # Step 1: deterministic forecast — no RNG (BUG #6, #9 fix)
        prices, occ1s, occ2s = build_forecast(state, horizon)

        # Step 2: solve MILP and return here-and-now action
        try:
            return build_and_solve_milp(state, prices, occ1s, occ2s, params)
        except Exception as e:
            print(f"[DLPolicy] solver failed at t={t_now}: {e}. Returning dummy.")
            return {"HeatPowerRoom1": 0.0,
                    "HeatPowerRoom2": 0.0,
                    "VentilationON":  0}


# standalone function interface (compatible with function-based callers)
def select_action(state):
    return DLPolicy().select_action(state)


# =============================================================================
#  SECTION 5 :  Quick local test
# =============================================================================
if __name__ == "__main__":
    test_state = {
        "T1": 21.0, "T2": 20.5, "H": 45.0,
        "Occ1": 30.0, "Occ2": 20.0,
        "price_t": 4.0, "price_previous": 3.5,
        "vent_counter": 0,
        "low_override_r1": 0, "low_override_r2": 0,
        "current_time": 0,
    }
    print("Action:", select_action(test_state))

    # Edge case: humidity overrule should fire
    humid_state = dict(test_state)
    humid_state["H"] = 75.0
    print("Humid action (v should=1):", select_action(humid_state))

    # Edge case: low-temp overrule with u=0 (BUG G case)
    cold_state = dict(test_state)
    cold_state["T1"] = 17.5
    cold_state["low_override_r1"] = 0
    print("Cold action (p1 should=3):", select_action(cold_state))