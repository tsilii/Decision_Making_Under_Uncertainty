# -*- coding: utf-8 -*-
"""
MSPolicy.py  —  Multi-stage Stochastic Programming Policy (Task 6)
===================================================================
Professor's 5-step structure (Lec05):

  Step 1: Horizon length L (including current stage 0)
  Step 2: Branching factor B per non-leaf node
  Step 3: Build scenario tree via Iterative Branch & Cluster
  Step 4: Non-anticipativity is implicit in the node-based formulation
  Step 5: Solve L-stage SP as a single node-indexed MILP

With L=3, B=5:
  - 31 tree nodes  (root + 5 stage-1 + 25 stage-2)
  - ~302 binary variables  (well within budget)
  - ~2 s per timestep on Gurobi
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyomo.environ import *
from pyomo.opt import TerminationCondition
import sys, os

import v2_SystemCharacteristics as SC
from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels


# =============================================================================
# STEP 3 — Iterative Branch & Cluster: build the scenario tree
# =============================================================================

def build_scenario_tree(state, t_now, L, B, S_init):
    """
    Iterative Branch & Cluster (Lec05 slide 41).

    Starting from the root (fully known current state), expands the tree
    L-1 times. At each non-leaf node, S_init Monte Carlo samples are
    generated, standardised, and clustered into B_k representative children.

    B can be an int (same branching factor at every stage) or a list of
    length L-1 specifying a per-stage branching factor [B_1, B_2, ...].
    Example: B=[20, 4] with L=3 gives 20 stage-1 nodes and 80 stage-2 nodes.

    Node fields: id, stage, parent, children,
                 price, price_prev, occ1, occ2, cond_prob, prob
    """
    # Normalise B to a list of length L-1
    if isinstance(B, int):
        B_per_stage = [B] * (L - 1)
    else:
        B_per_stage = list(B)

    root = {
        'id':         0,
        'stage':      0,
        'parent':     None,
        'children':   [],
        'price':      float(state['price_t']),
        'price_prev': float(state['price_previous']),
        'occ1':       float(state['Occ1']),
        'occ2':       float(state['Occ2']),
        'cond_prob':  1.0,
        'prob':       1.0,
    }
    nodes   = [root]
    next_id = 1
    queue   = [(0, 0)]   # (node_id, stage)

    while queue:
        parent_id, stage = queue.pop(0)
        if stage >= L - 1:
            continue  # leaf — no expansion

        parent  = nodes[parent_id]
        B_k     = B_per_stage[stage]   # branching factor at this stage

        # Sample S_init realisations for the next uncertain stage
        samples = np.empty((S_init, 3))
        for i in range(S_init):
            samples[i, 0] = price_model(parent['price'], parent['price_prev'])
            samples[i, 1], samples[i, 2] = next_occupancy_levels(
                parent['occ1'], parent['occ2']
            )

        # Cluster to B_k representatives (k-means with standardisation)
        scaler     = StandardScaler()
        X_sc       = scaler.fit_transform(samples)
        km         = KMeans(n_clusters=B_k, random_state=42, n_init=10)
        km.fit(X_sc)
        centroids  = scaler.inverse_transform(km.cluster_centers_)
        cond_probs = np.bincount(km.labels_, minlength=B_k) / S_init

        for k in range(B_k):
            child = {
                'id':         next_id,
                'stage':      stage + 1,
                'parent':     parent_id,
                'children':   [],
                'price':      float(centroids[k, 0]),
                'price_prev': parent['price'],   # parent's price becomes prev
                'occ1':       float(centroids[k, 1]),
                'occ2':       float(centroids[k, 2]),
                'cond_prob':  float(cond_probs[k]),
                'prob':       parent['prob'] * float(cond_probs[k]),
            }
            nodes.append(child)
            nodes[parent_id]['children'].append(next_id)
            queue.append((next_id, stage + 1))
            next_id += 1

    return nodes


# =============================================================================
# HELPER
# =============================================================================

def _descendants_within(node_by_id, start_id, max_depth):
    """All descendant IDs reachable from start_id within max_depth steps."""
    result = []
    queue  = [(start_id, 0)]
    while queue:
        nid, depth = queue.pop(0)
        if depth > 0:
            result.append(nid)
        if depth < max_depth:
            for cid in node_by_id[nid]['children']:
                queue.append((cid, depth + 1))
    return result


# =============================================================================
# STEPS 4-5 — Node-indexed MILP (non-anticipativity implicit)
# =============================================================================

def solve_mssp(state, nodes, t_now, params):
    """
    Multi-stage MILP in node-space (Lec05 slide 56).

    Variables are indexed by node n; decisions at a node are shared by
    all scenarios that pass through it — non-anticipativity holds by
    construction (Step 4).

    Objective: sum_{n} prob_n * price_n * (p1_n + p2_n + P_vent * v_n)

    Returns the here-and-now action (root node's decision).
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
    #T_low     = params['temp_min_comfort_threshold']
    T_ok      = params['temp_OK_threshold']
    #T_high    = params['temp_max_comfort_threshold']

    # REPLACEd WITH:
    T_low  = params['temp_min_comfort_threshold']
    T_high = params['temp_max_comfort_threshold']

    H_high    = params['humidity_threshold']
    U_vent    = params['vent_min_up_time']
    T_out_arr = params['outdoor_temperature']
    M_temp    = 50.0
    M_hum     = 200.0

    T1_now       = float(state['T1'])
    T2_now       = float(state['T2'])
    H_now        = float(state['H'])
    price_now    = float(state['price_t'])
    vent_counter = int(state['vent_counter'])
    low_or1      = int(state['low_override_r1'])
    low_or2      = int(state['low_override_r2'])

    node_by_id   = {n['id']: n for n in nodes}
    all_ids      = [n['id'] for n in nodes]
    non_root_ids = [nid for nid in all_ids if nid != 0]

    m = ConcreteModel()

    # ── Decisions at every node ───────────────────────────────────────────────
    m.p1 = Var(all_ids, domain=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(all_ids, domain=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(all_ids, domain=Binary)
    m.s  = Var(all_ids, domain=Binary)

    # ── State at non-root nodes (root state is fully known) ───────────────────
    m.T1 = Var(non_root_ids, domain=Reals)
    m.T2 = Var(non_root_ids, domain=Reals)
    # CURRENT — changed this line:
    # m.H  = Var(non_root_ids, domain=Reals)
    # TO THIS:
    m.H  = Var(non_root_ids, domain=NonNegativeReals) 

    # ── Threshold detectors at non-root nodes ─────────────────────────────────
    m.yLow1  = Var(non_root_ids, domain=Binary)
    m.yLow2  = Var(non_root_ids, domain=Binary)
    m.yOK1   = Var(non_root_ids, domain=Binary)
    m.yOK2   = Var(non_root_ids, domain=Binary)
    m.yHigh1 = Var(non_root_ids, domain=Binary)
    m.yHigh2 = Var(non_root_ids, domain=Binary)
    m.u1     = Var(non_root_ids, domain=Binary)
    m.u2     = Var(non_root_ids, domain=Binary)

    # ── Objective: expected cost across all nodes ─────────────────────────────
    def obj_rule(m):
        c = price_now * (m.p1[0] + m.p2[0] + P_vent * m.v[0])
        for nid in non_root_ids:
            nd = node_by_id[nid]
            c += nd['prob'] * nd['price'] * (m.p1[nid] + m.p2[nid] + P_vent * m.v[nid])
        return c
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # ── State dynamics: T1, T2, H at each non-root node ──────────────────────
    # At node n (stage k), state is driven by parent's decision and node's occ.
    # T_out index: transition from stage k-1 to k uses T_out at t_now + k - 1.
    def T1_dyn(m, nid):
        nd    = node_by_id[nid]
        pid   = nd['parent']
        T_out = T_out_arr[min(t_now + nd['stage'] - 1, len(T_out_arr) - 1)]
        T1p   = T1_now if pid == 0 else m.T1[pid]
        T2p   = T2_now if pid == 0 else m.T2[pid]
        return m.T1[nid] == (
            T1p + zeta_exch*(T2p - T1p) - zeta_loss*(T1p - T_out)
            + zeta_conv*m.p1[pid] - zeta_cool*m.v[pid] + zeta_occ*nd['occ1']
        )
    m.c_T1 = Constraint(non_root_ids, rule=T1_dyn)

    def T2_dyn(m, nid):
        nd    = node_by_id[nid]
        pid   = nd['parent']
        T_out = T_out_arr[min(t_now + nd['stage'] - 1, len(T_out_arr) - 1)]
        T1p   = T1_now if pid == 0 else m.T1[pid]
        T2p   = T2_now if pid == 0 else m.T2[pid]
        return m.T2[nid] == (
            T2p + zeta_exch*(T1p - T2p) - zeta_loss*(T2p - T_out)
            + zeta_conv*m.p2[pid] - zeta_cool*m.v[pid] + zeta_occ*nd['occ2']
        )
    m.c_T2 = Constraint(non_root_ids, rule=T2_dyn)

    def H_dyn(m, nid):
        nd  = node_by_id[nid]
        pid = nd['parent']
        Hp  = H_now if pid == 0 else m.H[pid]
        return m.H[nid] == Hp + eta_occ*(nd['occ1'] + nd['occ2']) - eta_vent*m.v[pid]
    m.c_H = Constraint(non_root_ids, rule=H_dyn)

    # ── Threshold detection at non-root nodes ─────────────────────────────────
    def c_yL1_u(m, n): return m.T1[n] <= T_low + M_temp*(1 - m.yLow1[n])
    def c_yL1_l(m, n): return m.T1[n] >= T_low - M_temp*m.yLow1[n]
    m.c_yL1_u = Constraint(non_root_ids, rule=c_yL1_u)
    m.c_yL1_l = Constraint(non_root_ids, rule=c_yL1_l)

    def c_yL2_u(m, n): return m.T2[n] <= T_low + M_temp*(1 - m.yLow2[n])
    def c_yL2_l(m, n): return m.T2[n] >= T_low - M_temp*m.yLow2[n]
    m.c_yL2_u = Constraint(non_root_ids, rule=c_yL2_u)
    m.c_yL2_l = Constraint(non_root_ids, rule=c_yL2_l)

    def c_yO1_l(m, n): return m.T1[n] >= T_ok - M_temp*(1 - m.yOK1[n])
    def c_yO1_u(m, n): return m.T1[n] <= T_ok + M_temp*m.yOK1[n]
    m.c_yO1_l = Constraint(non_root_ids, rule=c_yO1_l)
    m.c_yO1_u = Constraint(non_root_ids, rule=c_yO1_u)

    def c_yO2_l(m, n): return m.T2[n] >= T_ok - M_temp*(1 - m.yOK2[n])
    def c_yO2_u(m, n): return m.T2[n] <= T_ok + M_temp*m.yOK2[n]
    m.c_yO2_l = Constraint(non_root_ids, rule=c_yO2_l)
    m.c_yO2_u = Constraint(non_root_ids, rule=c_yO2_u)

    def c_yH1_l(m, n): return m.T1[n] >= T_high - M_temp*(1 - m.yHigh1[n])
    def c_yH1_u(m, n): return m.T1[n] <= T_high + M_temp*m.yHigh1[n]
    m.c_yH1_l = Constraint(non_root_ids, rule=c_yH1_l)
    m.c_yH1_u = Constraint(non_root_ids, rule=c_yH1_u)

    def c_yH2_l(m, n): return m.T2[n] >= T_high - M_temp*(1 - m.yHigh2[n])
    def c_yH2_u(m, n): return m.T2[n] <= T_high + M_temp*m.yHigh2[n]
    m.c_yH2_l = Constraint(non_root_ids, rule=c_yH2_l)
    m.c_yH2_u = Constraint(non_root_ids, rule=c_yH2_u)

    # ── Overrule controller at non-root nodes ─────────────────────────────────
    # u1[n]=1 → heater forced to P_max at node n
    # Triggered when T1[n] < T_low; persists from parent until T1 >= T_ok
    def c_u1_trig(m, n): return m.u1[n] >= m.yLow1[n]
    m.c_u1_trig = Constraint(non_root_ids, rule=c_u1_trig)

    def c_u1_pers(m, n):
        pid  = node_by_id[n]['parent']
        u1_p = low_or1 if pid == 0 else m.u1[pid]
        return m.u1[n] >= u1_p - m.yOK1[n]
    m.c_u1_pers = Constraint(non_root_ids, rule=c_u1_pers)

    def c_u1_dact(m, n): return m.u1[n] <= 1 - m.yOK1[n]
    m.c_u1_dact = Constraint(non_root_ids, rule=c_u1_dact)

    def c_u2_trig(m, n): return m.u2[n] >= m.yLow2[n]
    m.c_u2_trig = Constraint(non_root_ids, rule=c_u2_trig)

    def c_u2_pers(m, n):
        pid  = node_by_id[n]['parent']
        u2_p = low_or2 if pid == 0 else m.u2[pid]
        return m.u2[n] >= u2_p - m.yOK2[n]
    m.c_u2_pers = Constraint(non_root_ids, rule=c_u2_pers)

    def c_u2_dact(m, n): return m.u2[n] <= 1 - m.yOK2[n]
    m.c_u2_dact = Constraint(non_root_ids, rule=c_u2_dact)

    def c_p1_low(m, n):  return m.p1[n] >= P_max * m.u1[n]
    def c_p2_low(m, n):  return m.p2[n] >= P_max * m.u2[n]
    def c_p1_high(m, n): return m.p1[n] <= P_max * (1 - m.yHigh1[n])
    def c_p2_high(m, n): return m.p2[n] <= P_max * (1 - m.yHigh2[n])
    m.c_p1_low  = Constraint(non_root_ids, rule=c_p1_low)
    m.c_p2_low  = Constraint(non_root_ids, rule=c_p2_low)
    m.c_p1_high = Constraint(non_root_ids, rule=c_p1_high)
    m.c_p2_high = Constraint(non_root_ids, rule=c_p2_high)

    # ── Overrule at root (current state fully known) ──────────────────────────
    #if low_or1 == 1:      m.p1[0].fix(P_max)
    #if low_or2 == 1:      m.p2[0].fix(P_max)
    #if T1_now >= T_high:  m.p1[0].fix(0.0)
    #if T2_now >= T_high:  m.p2[0].fix(0.0)

    # ── Overrule at root (current state fully known) ──────────────────────────
    # Room 1
    if T1_now < params['temp_min_comfort_threshold']:
        low_or1 = 1
    if low_or1 == 1:
        if T1_now > params['temp_OK_threshold']:
            low_or1 = 0
        else:
            m.p1[0].fix(P_max)
    if T1_now > params['temp_max_comfort_threshold']:
        m.p1[0].fix(0.0)

    # Room 2
    if T2_now < params['temp_min_comfort_threshold']:
        low_or2 = 1
    if low_or2 == 1:
        if T2_now > params['temp_OK_threshold']:
            low_or2 = 0
        else:
            m.p2[0].fix(P_max)
    if T2_now > params['temp_max_comfort_threshold']:
        m.p2[0].fix(0.0)

    # ── Ventilation startup at root ───────────────────────────────────────────
    v_prev = 1 if vent_counter > 0 else 0
    m.c_sr1 = Constraint(expr= m.s[0] >= m.v[0] - v_prev)
    m.c_sr2 = Constraint(expr= m.s[0] <= m.v[0])
    m.c_sr3 = Constraint(expr= m.s[0] <= 1 - v_prev)

    # ── Ventilation startup at non-root nodes ─────────────────────────────────
    # s[n]=1  iff  v[n]=1  AND  v[parent(n)]=0
    def c_s1(m, n): return m.s[n] >= m.v[n] - m.v[node_by_id[n]['parent']]
    def c_s2(m, n): return m.s[n] <= m.v[n]
    def c_s3(m, n): return m.s[n] <= 1 - m.v[node_by_id[n]['parent']]
    m.c_s1 = Constraint(non_root_ids, rule=c_s1)
    m.c_s2 = Constraint(non_root_ids, rule=c_s2)
    m.c_s3 = Constraint(non_root_ids, rule=c_s3)

    # ── Ventilation uptime: startup forces v=1 on all descendants within U_vent-1 steps
    uptime_pairs = []
    for nd in nodes:
        for did in _descendants_within(node_by_id, nd['id'], U_vent - 1):
            uptime_pairs.append((nd['id'], did))

    if uptime_pairs:
        m.up_idx = RangeSet(0, len(uptime_pairs) - 1)
        def c_uptime(m, i):
            a_id, d_id = uptime_pairs[i]
            return m.v[d_id] >= m.s[a_id]
        m.c_uptime = Constraint(m.up_idx, rule=c_uptime)


    # ── Ventilation inertia from current vent_counter ─────────────────────────
    if vent_counter > 0:
        remaining_vent = U_vent - vent_counter
        if remaining_vent > 0:
            m.v[0].fix(1)
        for stage_k in range(1, remaining_vent):
            for nid in [n['id'] for n in nodes if n['stage'] == stage_k]:
                m.v[nid].fix(1)

    # ── Humidity overrule at root ─────────────────────────────────────────────
    #if H_now >= H_high:
    #    m.v[0].fix(1)

    if H_now > H_high:    # strict > to match environment
        m.v[0].fix(1)

    # ── Humidity overrule at non-root nodes ───────────────────────────────────
    def c_hum(m, n): return m.H[n] <= H_high + M_hum * m.v[n]
    m.c_hum = Constraint(non_root_ids, rule=c_hum)

    # ── Solve ─────────────────────────────────────────────────────────────────
    # TimeLimit=12 leaves ~3s buffer vs the environment's 15s wall-clock cutoff
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 12
    solver.options['MIPGap']    = 0.01
    result = solver.solve(m, tee=False)

    if result.solver.termination_condition == TerminationCondition.maxTimeLimit:
        print(f"[MSPolicy] WARNING: Gurobi hit 12s TimeLimit — returning best incumbent found")

    try:
        return float(value(m.p1[0])), float(value(m.p2[0])), int(round(value(m.v[0])))
    except Exception:
        return 0.0, 0.0, 0


# =============================================================================
# FALLBACK: 1-stage MILP for last hour
# =============================================================================

def solve_1stage(state, params):
    P_max  = params['heating_max_power']
    P_vent = params['ventilation_power']
    T_high = params['temp_max_comfort_threshold']
    H_high = params['humidity_threshold']
    U_vent = params['vent_min_up_time']

    T1_now       = float(state['T1'])
    T2_now       = float(state['T2'])
    H_now        = float(state['H'])
    price_now    = float(state['price_t'])
    vent_counter = int(state['vent_counter'])
    low_or1      = int(state['low_override_r1'])
    low_or2      = int(state['low_override_r2'])

    m    = ConcreteModel()
    m.p1 = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(domain=Binary)
    m.s  = Var(domain=Binary)
    m.obj = Objective(expr=price_now*(m.p1 + m.p2 + P_vent*m.v), sense=minimize)

    v_prev = 1 if vent_counter > 0 else 0
    m.c_s1 = Constraint(expr= m.s >= m.v - v_prev)
    m.c_s2 = Constraint(expr= m.s <= m.v)
    m.c_s3 = Constraint(expr= m.s <= 1 - v_prev)

    # if low_or1 == 1:      m.p1.fix(P_max)
    # if low_or2 == 1:      m.p2.fix(P_max)
    # if T1_now >= T_high:  m.p1.fix(0.0)
    # if T2_now >= T_high:  m.p2.fix(0.0)
    # if H_now  >= H_high:  m.v.fix(1)
    # if vent_counter > 0 and (U_vent - vent_counter) > 0:
    #     m.v.fix(1)

    # Room 1
    if T1_now < params['temp_min_comfort_threshold']:
        low_or1 = 1
    if low_or1 == 1:
        if T1_now > params['temp_OK_threshold']:
            low_or1 = 0
        else:
            m.p1.fix(P_max)
    if T1_now > params['temp_max_comfort_threshold']:
        m.p1.fix(0.0)

    # Room 2
    if T2_now < params['temp_min_comfort_threshold']:
        low_or2 = 1
    if low_or2 == 1:
        if T2_now > params['temp_OK_threshold']:
            low_or2 = 0
        else:
            m.p2.fix(P_max)
    if T2_now > params['temp_max_comfort_threshold']:
        m.p2.fix(0.0)

    # Ventilation
    if H_now > params['humidity_threshold']:
        m.v.fix(1)
    if vent_counter > 0 and (U_vent - vent_counter) > 0:
        m.v.fix(1)

    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 5
    solver.solve(m, tee=False)

    try:
        return float(value(m.p1)), float(value(m.p2)), int(round(value(m.v)))
    except Exception:
        return 0.0, 0.0, 0


# =============================================================================
# POLICY CLASS
# =============================================================================

class MultiStageSPPolicy:
    """
    Multi-stage SP policy with rolling horizon (Lec05).

    At each hour τ, builds an L-stage scenario tree with branching factor B
    via Iterative Branch & Cluster, then solves the node-indexed MILP.
    Only the root decision is executed; the tree is rebuilt at τ+1.
    """

    def __init__(self, L=3, B=None, S_init=500):
        """
        L:      number of lookahead stages (root = stage 0, leaves = stage L-1)
        B:      branching factor — int (uniform) or list per stage, e.g. [20, 4].
                Default [20, 4]: 20 stage-1 nodes + 80 stage-2 nodes = 101 nodes, ~1002 binaries.
        S_init: MC samples per node before clustering
        """
        self.L      = L
        self.B      = B if B is not None else [20, 4]
        self.S_init = S_init
        self.params = SC.get_fixed_data()

    def select_action(self, state):
        t_now     = int(state['current_time'])
        T_total   = self.params['num_timeslots']
        remaining = T_total - t_now

        if remaining <= 1:
            p1, p2, v = solve_1stage(state, self.params)
            return {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}

        L_eff = min(self.L, remaining)

        # Step 3: Iterative Branch & Cluster
        nodes = build_scenario_tree(state, t_now, L_eff, self.B, self.S_init)

        # Steps 4-5: solve node-indexed MILP (NA implicit)
        try:
            p1, p2, v = solve_mssp(state, nodes, t_now, self.params)
        except Exception as e:
            print(f"[MSPolicy] solver failed at t={t_now}: {e}")
            p1, p2, v = 0.0, 0.0, 0

        return {
            "HeatPowerRoom1": p1 if p1 is not None else 0.0,
            "HeatPowerRoom2": p2 if p2 is not None else 0.0,
            "VentilationON":  v  if v  is not None else 0,
        }


# =============================================================================
# MODULE-LEVEL select_action (required by environment)
# =============================================================================

_policy = MultiStageSPPolicy(L=3, B=[20, 4], S_init=500)

def select_action(state):
    return _policy.select_action(state)
