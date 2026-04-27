# -*- coding: utf-8 -*-
"""
SPPolicy.py (Task 3)
=====================
Multi-stage Stochastic Programming policy.

HOW IT WORKS:
  At each hour t:
    1. Observe current state
    2. Generate N_init=100 scenarios for remaining hours using
       price_model() and next_occupancy_levels()
    3. Reduce to N_reduced=30 scenarios using Fast Forward Selection
    4. Solve a multi-stage stochastic MILP over all scenarios
       - Here-and-now variables (t=current): SAME across all scenarios
       - Wait-and-see variables (t>current): DIFFERENT per scenario
       - Objective: minimize EXPECTED cost = (1/S) * sum of scenario costs
    5. Extract and return ONLY the here-and-now action
    6. Discard the rest of the plan
    7. Repeat next hour with new observed state

KEY DESIGN CHOICES:
  - Lookahead horizon : full remaining horizon (T - current_time)
  - Initial scenarios : 100
  - Reduced scenarios : 30
  - Reduction method  : Fast Forward Selection
"""

import numpy as np
import sys
import os
from pyomo.environ import *

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = "/Users/manostsili/Desktop/dtu/courses/decision making under uncertainty /assignment_DC"
GIVEN_DIR = os.path.join(BASE_DIR, "given")
sys.path.insert(0, GIVEN_DIR)

import v2_SystemCharacteristics as SC

# Create alias so PriceProcessRestaurant.py can find SystemCharacteristics
# (professor's file uses old name without v2_ prefix)
sys.modules['SystemCharacteristics'] = SC

# Import the professor's process models
from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels


# =============================================================================
# SCENARIO GENERATION
# =============================================================================

def generate_scenarios(state, horizon, N_init=100):
    """
    Generates N_init scenarios for the remaining `horizon` hours.

    Each scenario is a sample path of:
      - prices[t]   for t = current_time+1 ... current_time+horizon
      - occ1[t]     for t = current_time+1 ... current_time+horizon
      - occ2[t]     for t = current_time+1 ... current_time+horizon

    The current values (t=0 of the lookahead) are known from the state
    and are the same across all scenarios.

    Returns
    -------
    scenarios : list of N_init dicts, each with keys:
                'prices' : array of shape (horizon,)
                'occ1'   : array of shape (horizon,)
                'occ2'   : array of shape (horizon,)
    """
    scenarios = []

    for _ in range(N_init):
        prices = np.zeros(horizon)
        occ1   = np.zeros(horizon)
        occ2   = np.zeros(horizon)

        # First step uses current state values
        curr_price = state["price_t"]
        prev_price = state["price_previous"]
        curr_occ1  = state["Occ1"]
        curr_occ2  = state["Occ2"]

        for h in range(horizon):
            # Sample next price
            next_price = price_model(curr_price, prev_price)

            # Sample next occupancies
            next_occ1, next_occ2 = next_occupancy_levels(curr_occ1, curr_occ2)

            prices[h] = next_price
            occ1[h]   = next_occ1
            occ2[h]   = next_occ2

            # Update for next step
            prev_price = curr_price
            curr_price = next_price
            curr_occ1  = next_occ1
            curr_occ2  = next_occ2

        scenarios.append({
            'prices': prices,
            'occ1':   occ1,
            'occ2':   occ2
        })

    return scenarios


# =============================================================================
# FAST FORWARD SELECTION (SCENARIO REDUCTION)
# =============================================================================

def fast_forward_selection(scenarios, N_reduced):
    """
    Reduces N_init scenarios to N_reduced using Fast Forward Selection.

    Algorithm:
      1. Start with all scenarios in the "deleted" set D
      2. Iteratively find the scenario in D closest to any scenario
         already in the "kept" set J
      3. Move that scenario from D to J
      4. Repeat until |J| = N_reduced

    Distance metric: Euclidean distance between scenario price vectors
    (prices drive cost most directly so we use them as the distance metric)

    Returns
    -------
    reduced_scenarios : list of N_reduced selected scenarios
    probabilities     : list of N_reduced equal weights (1/N_reduced)
    """
    N = len(scenarios)

    # Build distance matrix using price vectors
    price_matrix = np.array([s['prices'] for s in scenarios])  # (N, horizon)

    # Compute pairwise distances
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_matrix[i, j] = np.linalg.norm(price_matrix[i] - price_matrix[j])

    # Fast Forward Selection
    kept    = []       # indices of kept scenarios
    deleted = list(range(N))  # start with all in deleted set

    # Pick the first scenario: the one with minimum total distance to all others
    total_dist = dist_matrix.sum(axis=1)
    first = int(np.argmin(total_dist))
    kept.append(first)
    deleted.remove(first)

    # Iteratively add scenarios
    while len(kept) < N_reduced:
        best_candidate = None
        best_dist      = np.inf

        for candidate in deleted:
            # Distance from candidate to nearest kept scenario
            min_dist_to_kept = min(dist_matrix[candidate, k] for k in kept)
            if min_dist_to_kept < best_dist:
                best_dist      = min_dist_to_kept
                best_candidate = candidate

        kept.append(best_candidate)
        deleted.remove(best_candidate)

    reduced_scenarios = [scenarios[i] for i in kept]
    probabilities     = [1.0 / N_reduced] * N_reduced  # equal weights

    return reduced_scenarios, probabilities


# =============================================================================
# STOCHASTIC MILP
# =============================================================================

def solve_stochastic_milp(state, scenarios, probabilities, params):
    """
    Solves the multi-stage stochastic MILP.

    Structure:
      - Here-and-now variables (t=0 of lookahead):
          p1_now, p2_now, v_now → SAME across all scenarios
          These are the decisions we actually implement

      - Wait-and-see variables (t>0 of lookahead):
          p1[s,t], p2[s,t], v[s,t] → DIFFERENT per scenario s
          These are future decisions that adapt to each scenario

    Objective:
      minimize sum_s prob[s] * sum_t price[s,t] * (p1[s,t] + p2[s,t] + P_vent*v[s,t])

    The here-and-now constraint enforces non-anticipativity:
      all scenarios must agree on the t=0 decision.
    """
    # ── Parameters ────────────────────────────────────────────────────────────
    T_total   = params['num_timeslots']
    t_current = state['current_time']
    remaining = T_total - t_current
    horizon   = len(scenarios[0]['prices'])  # use actual scenario length       
    S         = len(scenarios)

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
    T_out     = params['outdoor_temperature']
    M_temp    = 100
    M_hum     = 200

    # Current state values (same for all scenarios at t=0)
    T1_init        = state['T1']
    T2_init        = state['T2']
    H_init         = state['H']
    occ1_init      = state['Occ1']
    occ2_init      = state['Occ2']
    vent_counter   = state['vent_counter']
    low_override_r1 = state['low_override_r1']
    low_override_r2 = state['low_override_r2']

    # ── Build scenario data arrays ─────────────────────────────────────────────
    # Index 0 = current hour (known), index 1..horizon-1 = future (scenario-dependent)
    # prices_sc[s][h] = price at lookahead step h in scenario s
    prices_sc = []
    occ1_sc   = []
    occ2_sc   = []

    for s in range(S):
        # Step 0: current known values
        p_row   = [state['price_t']] + list(scenarios[s]['prices'])
        o1_row  = [occ1_init]        + list(scenarios[s]['occ1'])
        o2_row  = [occ2_init]        + list(scenarios[s]['occ2'])
        prices_sc.append(p_row)
        occ1_sc.append(o1_row)
        occ2_sc.append(o2_row)

    # ── Pyomo model ────────────────────────────────────────────────────────────
    model   = ConcreteModel()
    Tset    = RangeSet(0, horizon - 1)   # lookahead time steps
    Sset    = RangeSet(0, S - 1)         # scenarios
    Rset    = Set(initialize=[1, 2])     # rooms

    # ── Decision variables ─────────────────────────────────────────────────────
    # Heating power per scenario per room per time
    model.p = Var(Sset, Rset, Tset, domain=NonNegativeReals, bounds=(0, P_max))

    # Ventilation per scenario per time
    model.v = Var(Sset, Tset, domain=Binary)

    # Ventilation startup per scenario per time
    model.s_vent = Var(Sset, Tset, domain=Binary)

    # Temperature per scenario per room per time
    model.Temp = Var(Sset, Rset, Tset, domain=Reals)

    # Humidity per scenario per time
    model.H = Var(Sset, Tset, domain=Reals)

    # Binary helpers for overrule logic
    model.y_low  = Var(Sset, Rset, Tset, domain=Binary)
    model.y_ok   = Var(Sset, Rset, Tset, domain=Binary)
    model.y_high = Var(Sset, Rset, Tset, domain=Binary)
    model.u      = Var(Sset, Rset, Tset, domain=Binary)

    # ── Non-anticipativity constraints (here-and-now) ──────────────────────────
    # All scenarios must agree on t=0 decisions
    # We enforce this by fixing all scenarios to equal scenario 0 at t=0
    def na_p1(model, s):
        if s == 0:
            return Constraint.Skip
        return model.p[s, 1, 0] == model.p[0, 1, 0]
    model.na_p1 = Constraint(Sset, rule=na_p1)

    def na_p2(model, s):
        if s == 0:
            return Constraint.Skip
        return model.p[s, 2, 0] == model.p[0, 2, 0]
    model.na_p2 = Constraint(Sset, rule=na_p2)

    def na_v(model, s):
        if s == 0:
            return Constraint.Skip
        return model.v[s, 0] == model.v[0, 0]
    model.na_v = Constraint(Sset, rule=na_v)

    # ── Objective: minimize expected cost ──────────────────────────────────────
    def obj_rule(model):
        return sum(
            probabilities[s] * sum(
                prices_sc[s][t] * (
                    sum(model.p[s, r, t] for r in [1, 2]) + P_vent * model.v[s, t]
                )
                for t in range(horizon)
            )
            for s in range(S)
        )
    model.obj = Objective(rule=obj_rule, sense=minimize)

    # ── Initial conditions (same for all scenarios) ────────────────────────────
    def init_T1(model, s):
        return model.Temp[s, 1, 0] == T1_init
    model.init_T1 = Constraint(Sset, rule=init_T1)

    def init_T2(model, s):
        return model.Temp[s, 2, 0] == T2_init
    model.init_T2 = Constraint(Sset, rule=init_T2)

    def init_H(model, s):
        return model.H[s, 0] == H_init
    model.init_H = Constraint(Sset, rule=init_H)

    # ── Temperature dynamics ───────────────────────────────────────────────────
    def temp_dynamics(model, s, r, t):
        if t == 0:
            return Constraint.Skip
        r_other = 2 if r == 1 else 1
        t_abs   = t_current + t   # absolute time index for T_out
        return model.Temp[s, r, t] == (
            model.Temp[s, r, t-1]
            + zeta_exch * (model.Temp[s, r_other, t-1] - model.Temp[s, r, t-1])
            - zeta_loss * (model.Temp[s, r, t-1] - T_out[t_abs - 1])
            + zeta_conv * model.p[s, r, t-1]
            - zeta_cool * model.v[s, t-1]
            + zeta_occ  * (occ1_sc[s][t-1] if r == 1 else occ2_sc[s][t-1])
        )
    model.temp_dynamics = Constraint(Sset, Rset, Tset, rule=temp_dynamics)

    # ── Humidity dynamics ──────────────────────────────────────────────────────
    def hum_dynamics(model, s, t):
        if t == 0:
            return Constraint.Skip
        return model.H[s, t] == (
            model.H[s, t-1]
            + eta_occ * (occ1_sc[s][t-1] + occ2_sc[s][t-1])
            - eta_vent * model.v[s, t-1]
        )
    model.hum_dynamics = Constraint(Sset, Tset, rule=hum_dynamics)

    # ── Overrule logic (same as Task 1 MILP) ──────────────────────────────────
    def y_high_upper(model, s, r, t):
        return model.Temp[s, r, t] >= T_high - M_temp * (1 - model.y_high[s, r, t])
    model.y_high_upper = Constraint(Sset, Rset, Tset, rule=y_high_upper)

    def y_high_lower(model, s, r, t):
        return model.Temp[s, r, t] <= T_high + M_temp * model.y_high[s, r, t]
    model.y_high_lower = Constraint(Sset, Rset, Tset, rule=y_high_lower)

    def y_low_upper(model, s, r, t):
        return model.Temp[s, r, t] <= T_low + M_temp * (1 - model.y_low[s, r, t])
    model.y_low_upper = Constraint(Sset, Rset, Tset, rule=y_low_upper)

    def y_low_lower(model, s, r, t):
        return model.Temp[s, r, t] >= T_low - M_temp * model.y_low[s, r, t]
    model.y_low_lower = Constraint(Sset, Rset, Tset, rule=y_low_lower)

    def y_ok_upper(model, s, r, t):
        return model.Temp[s, r, t] >= T_ok - M_temp * (1 - model.y_ok[s, r, t])
    model.y_ok_upper = Constraint(Sset, Rset, Tset, rule=y_ok_upper)

    def y_ok_lower(model, s, r, t):
        return model.Temp[s, r, t] <= T_ok + M_temp * model.y_ok[s, r, t]
    model.y_ok_lower = Constraint(Sset, Rset, Tset, rule=y_ok_lower)

    def overrule_trigger(model, s, r, t):
        return model.u[s, r, t] >= model.y_low[s, r, t]
    model.overrule_trigger = Constraint(Sset, Rset, Tset, rule=overrule_trigger)

    def overrule_memory(model, s, r, t):
        if t == 0:
            return Constraint.Skip
        return model.u[s, r, t] <= model.u[s, r, t-1] + model.y_low[s, r, t]
    model.overrule_memory = Constraint(Sset, Rset, Tset, rule=overrule_memory)

    def overrule_persist(model, s, r, t):
        if t == 0:
            return Constraint.Skip
        return model.u[s, r, t] >= model.u[s, r, t-1] - model.y_ok[s, r, t]
    model.overrule_persist = Constraint(Sset, Rset, Tset, rule=overrule_persist)

    def overrule_deactivate(model, s, r, t):
        if t == 0:
            return Constraint.Skip
        return model.u[s, r, t] <= 1 - model.y_ok[s, r, t]
    model.overrule_deactivate = Constraint(Sset, Rset, Tset, rule=overrule_deactivate)

    def overrule_max(model, s, r, t):
        return model.p[s, r, t] >= P_max * model.u[s, r, t]
    model.overrule_max = Constraint(Sset, Rset, Tset, rule=overrule_max)

    def overrule_zero(model, s, r, t):
        return model.p[s, r, t] <= P_max * (1 - model.y_high[s, r, t])
    model.overrule_zero = Constraint(Sset, Rset, Tset, rule=overrule_zero)

    # ── Ventilation startup and minimum up-time ────────────────────────────────
    def startup_detect1(model, s, t):
        return model.s_vent[s, t] >= model.v[s, t] - (model.v[s, t-1] if t > 0 else 0)
    model.startup_detect1 = Constraint(Sset, Tset, rule=startup_detect1)

    def startup_detect2(model, s, t):
        return model.s_vent[s, t] <= model.v[s, t]
    model.startup_detect2 = Constraint(Sset, Tset, rule=startup_detect2)

    def startup_detect3(model, s, t):
        if t == 0:
            return Constraint.Skip
        return model.s_vent[s, t] <= 1 - model.v[s, t-1]
    model.startup_detect3 = Constraint(Sset, Tset, rule=startup_detect3)

    def vent_uptime(model, s, t):
        h_end    = min(t + U_vent, horizon)
        duration = min(U_vent, horizon - t)
        return sum(model.v[s, tau] for tau in range(t, h_end)) >= duration * model.s_vent[s, t]
    model.vent_uptime = Constraint(Sset, Tset, rule=vent_uptime)

    # ── Ventilation inertia from current state ─────────────────────────────────
    # If vent_counter > 0, ventilation must stay ON for remaining inertia hours
    if vent_counter > 0:
        remaining_forced = U_vent - vent_counter
        for s in range(S):
            for t in range(min(remaining_forced, horizon)):
                model.v[s, t].fix(1)

    # ── Humidity triggered ventilation ────────────────────────────────────────
    def hum_vent(model, s, t):
        return model.H[s, t] <= H_high + M_hum * model.v[s, t]
    model.hum_vent = Constraint(Sset, Tset, rule=hum_vent)

    # ── Handle initial override states ────────────────────────────────────────
    # If override is active at t=0, force heater to max for first step
    if low_override_r1 == 1:
        for s in range(S):
            model.p[s, 1, 0].fix(P_max)
    if low_override_r2 == 1:
        for s in range(S):
            model.p[s, 2, 0].fix(P_max)

    # ── Solve ──────────────────────────────────────────────────────────────────
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 12
    solver.solve(model, tee=False)
    # ── Extract here-and-now action (t=0, scenario 0) ─────────────────────────
    p1_now = value(model.p[0, 1, 0])
    p2_now = value(model.p[0, 2, 0])
    v_now  = round(value(model.v[0, 0]))

    return p1_now, p2_now, v_now


# =============================================================================
# SP POLICY CLASS
# =============================================================================

class SPPolicy:
    """
    Multi-stage Stochastic Programming policy.
    Implements the select_action(state) interface required by the environment.
    """

    def __init__(self, N_init=300, N_reduced=100):
   
        """
        Parameters
        ----------
        N_init    : number of scenarios to generate before reduction
        N_reduced : number of scenarios to keep after reduction
        """
        self.N_init    = N_init
        self.N_reduced = N_reduced
        self.params    = SC.get_fixed_data()

    def select_action(self, state):
        """
        Called by the environment at each hour.

        1. Generate N_init scenarios for remaining horizon
        2. Reduce to N_reduced via Fast Forward Selection
        3. Solve stochastic MILP
        4. Return here-and-now action
        """

        #import time
        #t_start = time.time()

        t_current = state['current_time']
        remaining = self.params['num_timeslots'] - t_current
        horizon   = min(remaining, 5)   # cap at 5 hours lookahead

        # If last hour, just use deterministic action
        if horizon <= 0:
            return {
                "HeatPowerRoom1": 0.0,
                "HeatPowerRoom2": 0.0,
                "VentilationON":  0
            }

        # Step 1: Generate scenarios
        scenarios = generate_scenarios(state, horizon, N_init=self.N_init)

        # Step 2: Reduce scenarios
        reduced_scenarios, probabilities = fast_forward_selection(
            scenarios, N_reduced=self.N_reduced
        )

        # Step 3: Solve stochastic MILP
        p1, p2, v = solve_stochastic_milp(
            state, reduced_scenarios, probabilities, self.params
        )

        # Step 4: Return here-and-now action
        #elapsed = time.time() - t_start
        #print(f"    t={t_current}: {elapsed:.2f}s  (horizon={horizon})")

        return {
            "HeatPowerRoom1": p1 if p1 is not None else 0.0,
            "HeatPowerRoom2": p2 if p2 is not None else 0.0,
            "VentilationON":  v  if v  is not None else 0
        }