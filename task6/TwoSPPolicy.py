# 2-stage SP policy for the restaurant HVAC problem.
# Stage τ decisions (p1_now, p2_now, v_now) are fixed across all scenarios;
# stage τ+1 recourse (p1[ω], p2[ω], v[ω]) adapts per scenario.
# Uncertainty is over joint (price, Occ1, Occ2) at τ+1 — 10k MC draws
# clustered down to 100 scenarios via k-means.

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyomo.environ import *
import sys, os

import v2_SystemCharacteristics as SC

from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels


def generate_scenarios(state, N_init=10000):
    """Draw N_init joint samples of next-step price and occupancy."""
    scenarios_raw = np.empty((N_init, 3))
    for i in range(N_init):
        scenarios_raw[i, 0] = price_model(
            state['price_t'], state['price_previous']
        )
        scenarios_raw[i, 1], scenarios_raw[i, 2] = next_occupancy_levels(
            state['Occ1'], state['Occ2']
        )
    return scenarios_raw


def reduce_scenarios(scenarios_raw, N_reduced=100):
    """Cluster raw scenarios down to N_reduced via k-means.

    Must scale first — price [0,12] vs occupancy [10,50] would otherwise
    let occupancy dominate the distance metric.
    """
    N_init = len(scenarios_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(scenarios_raw)

    km = KMeans(n_clusters=N_reduced, random_state=42, n_init=10)
    km.fit(X_scaled)

    centroids_scaled = km.cluster_centers_
    centroids        = scaler.inverse_transform(centroids_scaled)

    labels        = km.labels_
    cluster_sizes = np.bincount(labels, minlength=N_reduced)
    probabilities = cluster_sizes / N_init

    assert abs(probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"

    return centroids, probabilities


def compute_max_scenarios(binary_budget=2000):
    """Cap N so we don't blow the binary variable budget.
    Each scenario adds 10 binaries at τ+1; 2 are shared at τ.
    """
    N_max = int((binary_budget - 2) / 10)
    return max(N_max, 1)


def solve_2stage_milp(state, centroids, probabilities, params):
    """Build and solve the 2-stage MILP over {τ, τ+1}.

    T_{τ+1} is deterministic given here-and-now decisions, but we index it
    by scenario anyway so Pyomo can link it cleanly to the per-ω p constraints.
    """

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
    M_temp    = 50.0
    M_hum     = 200.0

    T1_now          = float(state['T1'])
    T2_now          = float(state['T2'])
    H_now           = float(state['H'])
    price_now       = float(state['price_t'])
    vent_counter    = int(state['vent_counter'])
    low_override_r1 = int(state['low_override_r1'])
    low_override_r2 = int(state['low_override_r2'])
    t_current       = int(state['current_time'])
    occ1_now        = float(state['Occ1'])
    occ2_now        = float(state['Occ2'])

    T_out_now = T_out[min(t_current, len(T_out) - 1)]

    N  = len(centroids)
    Ω  = range(N)

    lambda_s = centroids[:, 0]   # price at τ+1 per scenario

    m = ConcreteModel()

    # here-and-now decisions
    m.p1_now = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.p2_now = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.v_now  = Var(domain=Binary)
    m.s_now  = Var(domain=Binary)

    # recourse decisions, one copy per scenario
    m.p1 = Var(Ω, domain=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(Ω, domain=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(Ω, domain=Binary)
    m.s  = Var(Ω, domain=Binary)

    m.T1_next = Var(Ω, domain=Reals)
    m.T2_next = Var(Ω, domain=Reals)
    m.H_next  = Var(Ω, domain=Reals)

    # temperature zone indicators at τ+1
    m.yLow1  = Var(Ω, domain=Binary)   # T1_next <= T_low
    m.yLow2  = Var(Ω, domain=Binary)   # T2_next <= T_low
    m.yOK1   = Var(Ω, domain=Binary)   # T1_next >= T_ok
    m.yOK2   = Var(Ω, domain=Binary)   # T2_next >= T_ok
    m.yHigh1 = Var(Ω, domain=Binary)   # T1_next >= T_high
    m.yHigh2 = Var(Ω, domain=Binary)   # T2_next >= T_high

    # low-overrule flags at τ+1
    m.u1 = Var(Ω, domain=Binary)
    m.u2 = Var(Ω, domain=Binary)

    # minimise energy cost now + expected cost at τ+1
    def obj_rule(m):
        cost_now    = price_now * (m.p1_now + m.p2_now + P_vent * m.v_now)
        cost_future = sum(
            probabilities[w] * lambda_s[w] * (m.p1[w] + m.p2[w] + P_vent * m.v[w])
            for w in Ω
        )
        return cost_now + cost_future
    m.obj = Objective(rule=obj_rule, sense=minimize)

    def T1_dyn_rule(m, w):
        return m.T1_next[w] == (
            T1_now
            + zeta_exch * (T2_now - T1_now)
            - zeta_loss * (T1_now - T_out_now)
            + zeta_conv * m.p1_now
            - zeta_cool * m.v_now
            + zeta_occ  * occ1_now
        )
    m.T1_dyn = Constraint(Ω, rule=T1_dyn_rule)

    def T2_dyn_rule(m, w):
        return m.T2_next[w] == (
            T2_now
            + zeta_exch * (T1_now - T2_now)
            - zeta_loss * (T2_now - T_out_now)
            + zeta_conv * m.p2_now
            - zeta_cool * m.v_now
            + zeta_occ  * occ2_now
        )
    m.T2_dyn = Constraint(Ω, rule=T2_dyn_rule)

    def H_dyn_rule(m, w):
        return m.H_next[w] == (
            H_now
            + eta_occ  * (occ1_now + occ2_now)
            - eta_vent * m.v_now
        )
    m.H_dyn = Constraint(Ω, rule=H_dyn_rule)

    # current state is known, so fix directly instead of Big-M
    if T1_now < T_low or (low_override_r1 == 1 and T1_now < T_ok):
        m.p1_now.fix(P_max)
    if T1_now > T_high:
        m.p1_now.fix(0.0)
    if T2_now < T_low or (low_override_r2 == 1 and T2_now < T_ok):
        m.p2_now.fix(P_max)
    if T2_now > T_high:
        m.p2_now.fix(0.0)

    # yLow1 = 1 when T1_next <= T_low
    def c_yLow1_upper(m, w):
        return m.T1_next[w] <= T_low + M_temp * (1 - m.yLow1[w])
    def c_yLow1_lower(m, w):
        return m.T1_next[w] >= T_low - M_temp * m.yLow1[w]
    m.c_yLow1_upper = Constraint(Ω, rule=c_yLow1_upper)
    m.c_yLow1_lower = Constraint(Ω, rule=c_yLow1_lower)

    # yLow2 = 1 when T2_next <= T_low
    def c_yLow2_upper(m, w):
        return m.T2_next[w] <= T_low + M_temp * (1 - m.yLow2[w])
    def c_yLow2_lower(m, w):
        return m.T2_next[w] >= T_low - M_temp * m.yLow2[w]
    m.c_yLow2_upper = Constraint(Ω, rule=c_yLow2_upper)
    m.c_yLow2_lower = Constraint(Ω, rule=c_yLow2_lower)

    # yOK1 = 1 when T1_next >= T_ok
    def c_yOK1_lower(m, w):
        return m.T1_next[w] >= T_ok - M_temp * (1 - m.yOK1[w])
    def c_yOK1_upper(m, w):
        return m.T1_next[w] <= T_ok + M_temp * m.yOK1[w]
    m.c_yOK1_lower = Constraint(Ω, rule=c_yOK1_lower)
    m.c_yOK1_upper = Constraint(Ω, rule=c_yOK1_upper)

    # yOK2 = 1 when T2_next >= T_ok
    def c_yOK2_lower(m, w):
        return m.T2_next[w] >= T_ok - M_temp * (1 - m.yOK2[w])
    def c_yOK2_upper(m, w):
        return m.T2_next[w] <= T_ok + M_temp * m.yOK2[w]
    m.c_yOK2_lower = Constraint(Ω, rule=c_yOK2_lower)
    m.c_yOK2_upper = Constraint(Ω, rule=c_yOK2_upper)

    # yHigh1 = 1 when T1_next >= T_high
    def c_yHigh1_lower(m, w):
        return m.T1_next[w] >= T_high - M_temp * (1 - m.yHigh1[w])
    def c_yHigh1_upper(m, w):
        return m.T1_next[w] <= T_high + M_temp * m.yHigh1[w]
    m.c_yHigh1_lower = Constraint(Ω, rule=c_yHigh1_lower)
    m.c_yHigh1_upper = Constraint(Ω, rule=c_yHigh1_upper)

    # yHigh2 = 1 when T2_next >= T_high
    def c_yHigh2_lower(m, w):
        return m.T2_next[w] >= T_high - M_temp * (1 - m.yHigh2[w])
    def c_yHigh2_upper(m, w):
        return m.T2_next[w] <= T_high + M_temp * m.yHigh2[w]
    m.c_yHigh2_lower = Constraint(Ω, rule=c_yHigh2_lower)
    m.c_yHigh2_upper = Constraint(Ω, rule=c_yHigh2_upper)

    # overrule logic for room 1: u1=1 forces heater to P_max
    def c_u1_trigger(m, w):
        return m.u1[w] >= m.yLow1[w]
    m.c_u1_trigger = Constraint(Ω, rule=c_u1_trigger)

    def c_u1_persist(m, w):
        return m.u1[w] >= low_override_r1 * (1 - m.yOK1[w])
    m.c_u1_persist = Constraint(Ω, rule=c_u1_persist)

    def c_u1_deact(m, w):
        return m.u1[w] <= 1 - m.yOK1[w]
    m.c_u1_deact = Constraint(Ω, rule=c_u1_deact)

    # Room 2 — same logic
    def c_u2_trigger(m, w):
        return m.u2[w] >= m.yLow2[w]
    m.c_u2_trigger = Constraint(Ω, rule=c_u2_trigger)

    def c_u2_persist(m, w):
        return m.u2[w] >= low_override_r2 * (1 - m.yOK2[w])
    m.c_u2_persist = Constraint(Ω, rule=c_u2_persist)

    def c_u2_deact(m, w):
        return m.u2[w] <= 1 - m.yOK2[w]
    m.c_u2_deact = Constraint(Ω, rule=c_u2_deact)

    def c_p1_low(m, w):
        return m.p1[w] >= P_max * m.u1[w]
    m.c_p1_low = Constraint(Ω, rule=c_p1_low)

    def c_p2_low(m, w):
        return m.p2[w] >= P_max * m.u2[w]
    m.c_p2_low = Constraint(Ω, rule=c_p2_low)

    # too hot — heater off
    def c_p1_high(m, w):
        return m.p1[w] <= P_max * (1 - m.yHigh1[w])
    m.c_p1_high = Constraint(Ω, rule=c_p1_high)

    def c_p2_high(m, w):
        return m.p2[w] <= P_max * (1 - m.yHigh2[w])
    m.c_p2_high = Constraint(Ω, rule=c_p2_high)

    # s_now = 1 if vent just turned on this step
    v_prev = 1 if vent_counter > 0 else 0
    m.c_s_now_1 = Constraint(expr= m.s_now >= m.v_now - v_prev)
    m.c_s_now_2 = Constraint(expr= m.s_now <= m.v_now)
    m.c_s_now_3 = Constraint(expr= m.s_now <= 1 - v_prev)

    # s[ω] = 1 if vent starts at τ+1
    def c_s1(m, w):
        return m.s[w] >= m.v[w] - m.v_now
    def c_s2(m, w):
        return m.s[w] <= m.v[w]
    def c_s3(m, w):
        return m.s[w] <= 1 - m.v_now
    m.c_s1 = Constraint(Ω, rule=c_s1)
    m.c_s2 = Constraint(Ω, rule=c_s2)
    m.c_s3 = Constraint(Ω, rule=c_s3)

    # if vent starts now and min uptime > 1, it must stay on next step too
    if U_vent >= 2:
        def c_uptime_now(m, w):
            return m.v_now + m.v[w] >= 2 * m.s_now
        m.c_uptime_now = Constraint(Ω, rule=c_uptime_now)

    # carry over vent inertia from before this step
    if vent_counter > 0:
        remaining = U_vent - vent_counter
        if remaining > 0:
            m.v_now.fix(1)
        if remaining > 1:
            for w in Ω:
                m.v[w].fix(1)

    if H_now > H_high:
        m.v_now.fix(1)

    # force vent on next step if humidity would be too high
    def c_hum_next(m, w):
        return m.H_next[w] <= H_high + M_hum * m.v[w]
    m.c_hum_next = Constraint(Ω, rule=c_hum_next)

    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 10
    solver.options['MIPGap']    = 0.01
    result = solver.solve(m, tee=False)

    try:
        p1_val = float(value(m.p1_now))
        p2_val = float(value(m.p2_now))
        v_val  = int(round(value(m.v_now)))
    except Exception:
        p1_val, p2_val, v_val = 0.0, 0.0, 0

    return p1_val, p2_val, v_val


def solve_1stage(state, params):
    """Deterministic single-step MILP — used on the last time slot."""
    P_max     = params['heating_max_power']
    P_vent    = params['ventilation_power']
    zeta_exch = params['heat_exchange_coeff']
    zeta_loss = params['thermal_loss_coeff']
    zeta_conv = params['heating_efficiency_coeff']
    zeta_cool = params['heat_vent_coeff']
    zeta_occ  = params['heat_occupancy_coeff']
    T_high    = params['temp_max_comfort_threshold']
    T_low     = params['temp_min_comfort_threshold']
    T_ok      = params['temp_OK_threshold']
    H_high    = params['humidity_threshold']
    U_vent    = params['vent_min_up_time']
    T_out     = params['outdoor_temperature']
    M_temp    = 50.0
    M_hum     = 200.0

    T1_now          = float(state['T1'])
    T2_now          = float(state['T2'])
    H_now           = float(state['H'])
    occ1_now        = float(state['Occ1'])
    occ2_now        = float(state['Occ2'])
    price_now       = float(state['price_t'])
    vent_counter    = int(state['vent_counter'])
    low_override_r1 = int(state['low_override_r1'])
    low_override_r2 = int(state['low_override_r2'])

    m = ConcreteModel()
    m.p1 = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(domain=Binary)
    m.s  = Var(domain=Binary)

    m.obj = Objective(
        expr=price_now * (m.p1 + m.p2 + P_vent * m.v),
        sense=minimize
    )
    if T1_now < T_low or (low_override_r1 == 1 and T1_now < T_ok):
        m.p1.fix(P_max)
    if T1_now > T_high:
        m.p1.fix(0.0)
    if T2_now < T_low or (low_override_r2 == 1 and T2_now < T_ok):
        m.p2.fix(P_max)
    if T2_now > T_high:
        m.p2.fix(0.0)
    if H_now > H_high:
        m.v.fix(1)

    v_prev = 1 if vent_counter > 0 else 0
    if vent_counter > 0 and (U_vent - vent_counter) > 0:
        m.v.fix(1)

    m.c_s1 = Constraint(expr= m.s >= m.v - v_prev)
    m.c_s2 = Constraint(expr= m.s <= m.v)
    m.c_s3 = Constraint(expr= m.s <= 1 - v_prev)

    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 5
    result = solver.solve(m, tee=False)

    try:
        return float(value(m.p1)), float(value(m.p2)), int(round(value(m.v)))
    except Exception:
        return 0.0, 0.0, 0


class TwoStageSPPolicy:

    def __init__(self, N_init=10000, N_reduced=100):
        self.N_init    = N_init
        self.N_reduced = N_reduced
        self.params    = SC.get_fixed_data()

    def select_action(self, state):

        t_now   = int(state['current_time'])
        horizon = self.params['num_timeslots'] - t_now

        if horizon <= 1:
            p1, p2, v = solve_1stage(state, self.params)
            return {
                "HeatPowerRoom1": p1,
                "HeatPowerRoom2": p2,
                "VentilationON":  v
            }

        N_use = min(self.N_reduced, compute_max_scenarios())

        scenarios_raw = generate_scenarios(state, N_init=self.N_init)
        centroids, probabilities = reduce_scenarios(scenarios_raw, N_reduced=N_use)

        try:
            p1, p2, v = solve_2stage_milp(
                state, centroids, probabilities, self.params
            )
        except Exception as e:
            print(f"[SP2Policy] solver failed at t={t_now}: {e}")
            p1, p2, v = 0.0, 0.0, 0

        return {
            "HeatPowerRoom1": p1 if p1 is not None else 0.0,
            "HeatPowerRoom2": p2 if p2 is not None else 0.0,
            "VentilationON":  v  if v  is not None else 0
        }


_policy = TwoStageSPPolicy(N_init=10000, N_reduced=100)

def select_action(state):
    return _policy.select_action(state)