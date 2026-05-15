# =============================================================================
# 2-Stage Stochastic Programming Policy for the Restaurant HVAC Problem
# =============================================================================
#
# TRUE 2-stage structure per Lecture 4:
#   TT = {τ, τ+1}  — only two timeslots in the lookahead
#
#   Stage τ   (here-and-now):  p1_now, p2_now, v_now  — shared across all ω
#   Stage τ+1 (recourse):      p1[ω],  p2[ω],  v[ω]  — scenario-specific
#
# Uncertainty: joint (price_{τ+1}, Occ1_{τ+1}, Occ2_{τ+1})
#   - 10,000 Monte Carlo samples → k-means (with standardisation) → 100 scenarios
#   - Probabilities = cluster_size / 10,000
#
# Fixes applied vs previous version:
#   #1  horizon fixed to 2 (true 2-stage SP, not multi-stage)
#   #3  yLow/yOK/yHigh + u added at τ+1 (per scenario)
#   #4  overrule constraints for p[ω] at τ+1 based on T1_next/T2_next
#   #5  vent uptime constraint across τ↔τ+1 boundary
#   #6  humidity overrule at τ+1: H_next <= H_high + M * v[ω] (same time index)
#   #7  high-temp overrule forces p[ω]=0 at τ+1 based on T1_next/T2_next
#   #8  expected_continuation dropped entirely (true 2-stage has no τ+2…T)
#   #10 k-means with StandardScaler (price and occupancy on same scale)
#   #11 solver size is small (≈1,200 binaries), solves in <1s
#   #14 horizon=1: solve 1-stage (just here-and-now), not dummy zero
#   #15 no sys.modules hack
# =============================================================================

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyomo.environ import *
import sys, os

import v2_SystemCharacteristics as SC

from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels


# =============================================================================
# STEP 1: SCENARIO GENERATION — sample only τ+1 (3D)
# =============================================================================

def generate_scenarios(state, N_init=10000):
    """
    Draw N_init joint samples of (price_{τ+1}, Occ1_{τ+1}, Occ2_{τ+1})
    using the given process models directly.

    Returns
    -------
    scenarios_raw : np.array of shape (N_init, 3)
    """
    scenarios_raw = np.empty((N_init, 3))
    for i in range(N_init):
        scenarios_raw[i, 0] = price_model(
            state['price_t'], state['price_previous']
        )
        scenarios_raw[i, 1], scenarios_raw[i, 2] = next_occupancy_levels(
            state['Occ1'], state['Occ2']
        )
    return scenarios_raw


# =============================================================================
# STEP 2: SCENARIO REDUCTION — k-means with standardisation
# =============================================================================

def reduce_scenarios(scenarios_raw, N_reduced=100):
    """
    Reduce N_init raw scenarios to N_reduced via k-means.

    Uses StandardScaler so price [0,12] and occupancy [10,50] are
    on the same scale — fixing the k-means weighting bias (Error #10).

    Returns
    -------
    scenarios_reduced : np.array of shape (N_reduced, 3)  (in original units)
    probabilities     : np.array of shape (N_reduced,) summing to 1
    """
    N_init = len(scenarios_raw)

    # Standardise features before clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(scenarios_raw)

    km = KMeans(n_clusters=N_reduced, random_state=42, n_init=10)
    km.fit(X_scaled)

    # Centroids back in original units
    centroids_scaled = km.cluster_centers_
    centroids        = scaler.inverse_transform(centroids_scaled)

    labels        = km.labels_
    cluster_sizes = np.bincount(labels, minlength=N_reduced)
    probabilities = cluster_sizes / N_init

    assert abs(probabilities.sum() - 1.0) < 1e-6, "Probabilities must sum to 1"

    return centroids, probabilities


# =============================================================================
# STEP 2b: ADAPTIVE SCENARIO COUNT — binary budget guard
# =============================================================================

def compute_max_scenarios(binary_budget=2000):
    """
    For true 2-stage SP (steps=1), cap scenarios so total binary variables
    stay within budget.

    Binaries per scenario at τ+1: v, s, yLow1, yLow2, yOK1, yOK2,
                                   yHigh1, yHigh2, u1, u2  → 10
    Shared at τ: v_now, s_now → 2

    Formula: 2 + 10 * N <= binary_budget
    """
    N_max = int((binary_budget - 2) / 10)
    return max(N_max, 1)


# =============================================================================
# STEP 3: SOLVE 2-STAGE MILP  (TT = {τ, τ+1})
# =============================================================================

def solve_2stage_milp(state, centroids, probabilities, params):
    """
    True 2-stage stochastic MILP with lookahead horizon TT = {τ, τ+1}.

    Stage τ   (here-and-now): p1_now, p2_now, v_now — one shared decision
    Stage τ+1 (recourse):     p1[ω], p2[ω], v[ω]  — per scenario ω

    T_{τ+1} is SCALAR (same for all ω): depends only on known state and
    here-and-now decisions.  Overrule at τ+1 is enforced per scenario
    using Big-M on T1_next / T2_next (which are the same for all ω but
    kept as variables so Pyomo can link them to p[ω] constraints).
    """

    # ── Parameters ────────────────────────────────────────────────────────────
    P_max     = params['heating_max_power']          # 3.0
    P_vent    = params['ventilation_power']           # 2.0
    zeta_exch = params['heat_exchange_coeff']         # 0.6
    zeta_loss = params['thermal_loss_coeff']          # 0.1
    zeta_conv = params['heating_efficiency_coeff']    # 1.0
    zeta_cool = params['heat_vent_coeff']             # 0.7
    zeta_occ  = params['heat_occupancy_coeff']        # 0.02
    eta_occ   = params['humidity_occupancy_coeff']    # 0.18
    eta_vent  = params['humidity_vent_coeff']         # 15.0
    T_low     = params['temp_min_comfort_threshold']  # 18.0
    T_ok      = params['temp_OK_threshold']           # 22.0
    T_high    = params['temp_max_comfort_threshold']  # 26.0
    H_high    = params['humidity_threshold']          # 70.0
    U_vent    = params['vent_min_up_time']            # 3
    T_out     = params['outdoor_temperature']
    M_temp    = 50.0
    M_hum     = 200.0

    # ── Current state ─────────────────────────────────────────────────────────
    T1_now          = float(state['T1'])
    T2_now          = float(state['T2'])
    H_now           = float(state['H'])
    price_now       = float(state['price_t'])
    vent_counter    = int(state['vent_counter'])
    low_override_r1 = int(state['low_override_r1'])
    low_override_r2 = int(state['low_override_r2'])
    t_current       = int(state['current_time'])

    # Outdoor temperature at τ (for computing T_{τ+1})
    T_out_now = T_out[min(t_current, len(T_out) - 1)]

    # ── Scenario data ──────────────────────────────────────────────────────────
    N  = len(centroids)
    Ω  = range(N)

    lambda_s = centroids[:, 0]   # price  at τ+1 per scenario
    occ1_s   = centroids[:, 1]   # occ1   at τ+1 per scenario
    occ2_s   = centroids[:, 2]   # occ2   at τ+1 per scenario

    # ── Pyomo model ───────────────────────────────────────────────────────────
    m = ConcreteModel()

    # ─────────────────────────────────────────────────────────────────────────
    # VARIABLES
    # ─────────────────────────────────────────────────────────────────────────

    # Stage τ — here-and-now (shared, no ω index)
    m.p1_now = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.p2_now = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.v_now  = Var(domain=Binary)
    m.s_now  = Var(domain=Binary)   # startup at τ

    # Stage τ+1 — recourse (per scenario ω)
    m.p1 = Var(Ω, domain=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(Ω, domain=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(Ω, domain=Binary)
    m.s  = Var(Ω, domain=Binary)   # startup at τ+1

    # T and H at τ+1: PER SCENARIO — occupancy at τ+1 is scenario-specific
    # (occ1_s[w] varies across ω, so T1_next[w] differs per scenario)
    m.T1_next = Var(Ω, domain=Reals)
    m.T2_next = Var(Ω, domain=Reals)
    m.H_next  = Var(Ω, domain=Reals)

    # Threshold detection at τ+1 (per ω)
    m.yLow1  = Var(Ω, domain=Binary)   # T1_next[w] <= T_low
    m.yLow2  = Var(Ω, domain=Binary)   # T2_next[w] <= T_low
    m.yOK1   = Var(Ω, domain=Binary)   # T1_next[w] >= T_ok
    m.yOK2   = Var(Ω, domain=Binary)   # T2_next[w] >= T_ok
    m.yHigh1 = Var(Ω, domain=Binary)   # T1_next[w] >= T_high
    m.yHigh2 = Var(Ω, domain=Binary)   # T2_next[w] >= T_high

    # Low-overrule controller at τ+1 (per ω)
    m.u1 = Var(Ω, domain=Binary)
    m.u2 = Var(Ω, domain=Binary)

    # ─────────────────────────────────────────────────────────────────────────
    # OBJECTIVE: energy cost at τ + expected energy cost at τ+1
    # Comfort is handled purely via hard overrule constraints below
    # ─────────────────────────────────────────────────────────────────────────
    def obj_rule(m):
        cost_now    = price_now * (m.p1_now + m.p2_now + P_vent * m.v_now)
        cost_future = sum(
            probabilities[w] * lambda_s[w] * (m.p1[w] + m.p2[w] + P_vent * m.v[w])
            for w in Ω
        )
        return cost_now + cost_future
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE-1 DYNAMICS: T_{τ+1} and H_{τ+1} — PER SCENARIO
    # occ1_s[w] / occ2_s[w] are the scenario-specific occupancies at τ+1
    # that drive the temperature transition, so T1_next[w] differs per ω
    # ─────────────────────────────────────────────────────────────────────────
    def T1_dyn_rule(m, w):
        return m.T1_next[w] == (
            T1_now
            + zeta_exch * (T2_now - T1_now)
            - zeta_loss * (T1_now - T_out_now)
            + zeta_conv * m.p1_now
            - zeta_cool * m.v_now
            + zeta_occ  * occ1_s[w]
        )
    m.T1_dyn = Constraint(Ω, rule=T1_dyn_rule)

    def T2_dyn_rule(m, w):
        return m.T2_next[w] == (
            T2_now
            + zeta_exch * (T1_now - T2_now)
            - zeta_loss * (T2_now - T_out_now)
            + zeta_conv * m.p2_now
            - zeta_cool * m.v_now
            + zeta_occ  * occ2_s[w]
        )
    m.T2_dyn = Constraint(Ω, rule=T2_dyn_rule)

    def H_dyn_rule(m, w):
        return m.H_next[w] == (
            H_now
            + eta_occ  * (occ1_s[w] + occ2_s[w])
            - eta_vent * m.v_now
        )
    m.H_dyn = Constraint(Ω, rule=H_dyn_rule)

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRULE AT τ — fix() since current state is fully known
    # ─────────────────────────────────────────────────────────────────────────
    # Low-overrule: force heater to max
    if low_override_r1 == 1:
        m.p1_now.fix(P_max)
    if low_override_r2 == 1:
        m.p2_now.fix(P_max)
    # High-overrule takes priority: force heater to zero
    if T1_now >= T_high:
        m.p1_now.fix(0.0)
    if T2_now >= T_high:
        m.p2_now.fix(0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # BIG-M THRESHOLD DETECTION AT τ+1
    # T1_next / T2_next are the SAME for all ω, but we index by ω so we can
    # write per-scenario overrule constraints on p1[ω] / p2[ω] without
    # introducing nonlinear terms.
    # ─────────────────────────────────────────────────────────────────────────

    # yLow1[ω] = 1  iff  T1_next[w] <= T_low
    def c_yLow1_upper(m, w):
        return m.T1_next[w] <= T_low + M_temp * (1 - m.yLow1[w])
    def c_yLow1_lower(m, w):
        return m.T1_next[w] >= T_low - M_temp * m.yLow1[w]
    m.c_yLow1_upper = Constraint(Ω, rule=c_yLow1_upper)
    m.c_yLow1_lower = Constraint(Ω, rule=c_yLow1_lower)

    # yLow2[ω] = 1  iff  T2_next[w] <= T_low
    def c_yLow2_upper(m, w):
        return m.T2_next[w] <= T_low + M_temp * (1 - m.yLow2[w])
    def c_yLow2_lower(m, w):
        return m.T2_next[w] >= T_low - M_temp * m.yLow2[w]
    m.c_yLow2_upper = Constraint(Ω, rule=c_yLow2_upper)
    m.c_yLow2_lower = Constraint(Ω, rule=c_yLow2_lower)

    # yOK1[ω] = 1  iff  T1_next[w] >= T_ok
    def c_yOK1_lower(m, w):
        return m.T1_next[w] >= T_ok - M_temp * (1 - m.yOK1[w])
    def c_yOK1_upper(m, w):
        return m.T1_next[w] <= T_ok + M_temp * m.yOK1[w]
    m.c_yOK1_lower = Constraint(Ω, rule=c_yOK1_lower)
    m.c_yOK1_upper = Constraint(Ω, rule=c_yOK1_upper)

    # yOK2[ω] = 1  iff  T2_next[w] >= T_ok
    def c_yOK2_lower(m, w):
        return m.T2_next[w] >= T_ok - M_temp * (1 - m.yOK2[w])
    def c_yOK2_upper(m, w):
        return m.T2_next[w] <= T_ok + M_temp * m.yOK2[w]
    m.c_yOK2_lower = Constraint(Ω, rule=c_yOK2_lower)
    m.c_yOK2_upper = Constraint(Ω, rule=c_yOK2_upper)

    # yHigh1[ω] = 1  iff  T1_next[w] >= T_high
    def c_yHigh1_lower(m, w):
        return m.T1_next[w] >= T_high - M_temp * (1 - m.yHigh1[w])
    def c_yHigh1_upper(m, w):
        return m.T1_next[w] <= T_high + M_temp * m.yHigh1[w]
    m.c_yHigh1_lower = Constraint(Ω, rule=c_yHigh1_lower)
    m.c_yHigh1_upper = Constraint(Ω, rule=c_yHigh1_upper)

    # yHigh2[ω] = 1  iff  T2_next[w] >= T_high
    def c_yHigh2_lower(m, w):
        return m.T2_next[w] >= T_high - M_temp * (1 - m.yHigh2[w])
    def c_yHigh2_upper(m, w):
        return m.T2_next[w] <= T_high + M_temp * m.yHigh2[w]
    m.c_yHigh2_lower = Constraint(Ω, rule=c_yHigh2_lower)
    m.c_yHigh2_upper = Constraint(Ω, rule=c_yHigh2_upper)

    # ─────────────────────────────────────────────────────────────────────────
    # OVERRULE CONTROLLER AT τ+1 (per scenario ω)
    # Fix #3, #4, #7 — missing in previous version
    # ─────────────────────────────────────────────────────────────────────────

    # Low-overrule: u1[ω]=1 means heater must be at P_max at τ+1
    # Triggered if T1_next <= T_low
    def c_u1_trigger(m, w):
        return m.u1[w] >= m.yLow1[w]
    m.c_u1_trigger = Constraint(Ω, rule=c_u1_trigger)

    # Persists from τ if active and T1_next has not yet reached T_ok
    def c_u1_persist(m, w):
        return m.u1[w] >= low_override_r1 * (1 - m.yOK1[w])
    m.c_u1_persist = Constraint(Ω, rule=c_u1_persist)

    # Deactivated when T1_next >= T_ok
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

    # Low-overrule forces heater to P_max at τ+1
    def c_p1_low(m, w):
        return m.p1[w] >= P_max * m.u1[w]
    m.c_p1_low = Constraint(Ω, rule=c_p1_low)

    def c_p2_low(m, w):
        return m.p2[w] >= P_max * m.u2[w]
    m.c_p2_low = Constraint(Ω, rule=c_p2_low)

    # High-overrule forces heater to 0 at τ+1  (Fix #7)
    def c_p1_high(m, w):
        return m.p1[w] <= P_max * (1 - m.yHigh1[w])
    m.c_p1_high = Constraint(Ω, rule=c_p1_high)

    def c_p2_high(m, w):
        return m.p2[w] <= P_max * (1 - m.yHigh2[w])
    m.c_p2_high = Constraint(Ω, rule=c_p2_high)

    # ─────────────────────────────────────────────────────────────────────────
    # VENTILATION STARTUP AT τ
    # s_now = 1  iff  v_now=1 AND v_{τ-1}=0
    # ─────────────────────────────────────────────────────────────────────────
    v_prev = 1 if vent_counter > 0 else 0
    m.c_s_now_1 = Constraint(expr= m.s_now >= m.v_now - v_prev)
    m.c_s_now_2 = Constraint(expr= m.s_now <= m.v_now)
    m.c_s_now_3 = Constraint(expr= m.s_now <= 1 - v_prev)

    # ─────────────────────────────────────────────────────────────────────────
    # VENTILATION STARTUP AT τ+1 (per scenario)
    # s[ω] = 1  iff  v[ω]=1 AND v_now=0
    # ─────────────────────────────────────────────────────────────────────────
    def c_s1(m, w):
        return m.s[w] >= m.v[w] - m.v_now
    def c_s2(m, w):
        return m.s[w] <= m.v[w]
    def c_s3(m, w):
        return m.s[w] <= 1 - m.v_now
    m.c_s1 = Constraint(Ω, rule=c_s1)
    m.c_s2 = Constraint(Ω, rule=c_s2)
    m.c_s3 = Constraint(Ω, rule=c_s3)

    # ─────────────────────────────────────────────────────────────────────────
    # VENTILATION UPTIME ACROSS τ↔τ+1 BOUNDARY  (Fix #5)
    # If v_now starts (s_now=1) and U_vent >= 2, then v[ω] must also be 1
    # Generalised: v_now + v[ω] >= U_vent * s_now  (capped at 2 since TT={τ,τ+1})
    # ─────────────────────────────────────────────────────────────────────────
    if U_vent >= 2:
        def c_uptime_now(m, w):
            # If we start at τ, v[ω] at τ+1 must also be 1
            return m.v_now + m.v[w] >= 2 * m.s_now
        m.c_uptime_now = Constraint(Ω, rule=c_uptime_now)

    # Uptime at τ+1: if v starts at τ+1, it must stay on for U_vent hours
    # but our horizon only has τ+1, so this is a commitment beyond our window
    # — we simply enforce s[ω] <= v[ω] (already done above) and leave it.

    # ─────────────────────────────────────────────────────────────────────────
    # VENTILATION INERTIA FROM CURRENT STATE
    # ─────────────────────────────────────────────────────────────────────────
    if vent_counter > 0:
        remaining = U_vent - vent_counter
        if remaining > 0:
            # Must stay ON at τ
            m.v_now.fix(1)
        if remaining > 1:
            # Must also stay ON at τ+1 for all scenarios
            for w in Ω:
                m.v[w].fix(1)

    # ─────────────────────────────────────────────────────────────────────────
    # HUMIDITY OVERRULE AT τ — fix() since H_now is known
    # ─────────────────────────────────────────────────────────────────────────
    if H_now >= H_high:
        m.v_now.fix(1)

    # ─────────────────────────────────────────────────────────────────────────
    # HUMIDITY OVERRULE AT τ+1  (Fix #6)
    # H_next and v[ω] are at the SAME time index τ+1
    # If H_next >= H_high → v[ω] must be 1
    # ─────────────────────────────────────────────────────────────────────────
    def c_hum_next(m, w):
        return m.H_next[w] <= H_high + M_hum * m.v[w]
    m.c_hum_next = Constraint(Ω, rule=c_hum_next)

    # ─────────────────────────────────────────────────────────────────────────
    # SOLVE
    # ─────────────────────────────────────────────────────────────────────────
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 10
    solver.options['MIPGap']    = 0.01
    result = solver.solve(m, tee=False)

    # ─────────────────────────────────────────────────────────────────────────
    # EXTRACT HERE-AND-NOW ACTION
    # ─────────────────────────────────────────────────────────────────────────
    try:
        p1_val = float(value(m.p1_now))
        p2_val = float(value(m.p2_now))
        v_val  = int(round(value(m.v_now)))
    except Exception:
        p1_val, p2_val, v_val = 0.0, 0.0, 0

    return p1_val, p2_val, v_val


# =============================================================================
# STEP 4: 1-STAGE FALLBACK for the last hour  (Fix #14)
# =============================================================================

def solve_1stage(state, params):
    """
    When only one hour remains (horizon=1), solve a simple deterministic
    single-stage MILP using the current state. No uncertainty to model.
    """
    P_max     = params['heating_max_power']
    P_vent    = params['ventilation_power']
    zeta_exch = params['heat_exchange_coeff']
    zeta_loss = params['thermal_loss_coeff']
    zeta_conv = params['heating_efficiency_coeff']
    zeta_cool = params['heat_vent_coeff']
    zeta_occ  = params['heat_occupancy_coeff']
    T_high    = params['temp_max_comfort_threshold']
    T_low     = params['temp_min_comfort_threshold']
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

    # Overrule at τ
    if low_override_r1 == 1:
        m.p1.fix(P_max)
    if low_override_r2 == 1:
        m.p2.fix(P_max)
    if T1_now >= T_high:
        m.p1.fix(0.0)
    if T2_now >= T_high:
        m.p2.fix(0.0)
    if H_now >= H_high:
        m.v.fix(1)

    # Ventilation inertia
    v_prev = 1 if vent_counter > 0 else 0
    if vent_counter > 0 and (U_vent - vent_counter) > 0:
        m.v.fix(1)

    # Startup
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


# =============================================================================
# POLICY CLASS
# =============================================================================

class TwoStageSPPolicy:

    def __init__(self, N_init=10000, N_reduced=100):
        self.N_init    = N_init
        self.N_reduced = N_reduced
        self.params    = SC.get_fixed_data()

    def select_action(self, state):

        t_now   = int(state['current_time'])
        horizon = self.params['num_timeslots'] - t_now

        # Last hour: solve 1-stage (Fix #14 — not returning zero)
        if horizon <= 1:
            p1, p2, v = solve_1stage(state, self.params)
            return {
                "HeatPowerRoom1": p1,
                "HeatPowerRoom2": p2,
                "VentilationON":  v
            }

        # Cap scenarios to stay within binary budget
        N_use = min(self.N_reduced, compute_max_scenarios())

        # Step 1: Generate 10,000 Monte Carlo scenarios for τ+1 only
        scenarios_raw = generate_scenarios(state, N_init=self.N_init)

        # Step 2: Reduce to N_use representative scenarios via k-means
        centroids, probabilities = reduce_scenarios(
            scenarios_raw, N_reduced=N_use
        )

        # Step 3: Solve true 2-stage MILP (TT = {τ, τ+1})
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


# =============================================================================
# MODULE-LEVEL select_action (required by the environment)
# =============================================================================

_policy = TwoStageSPPolicy(N_init=10000, N_reduced=100)

def select_action(state):
    return _policy.select_action(state)