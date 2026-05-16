# -*- coding: utf-8 -*-
"""
HybridPolicy.py  —  Hybrid Policy (Task 5)
==========================================
Combines multi-stage stochastic programming with a deterministic
expected-value tail (Cost Function Approximation).

Structure:
  Stage 0          : root (current state, here-and-now decision)
  Stages 1-2       : stochastic scenario tree, B=[20,3] branching
                     (Iterative Branch & Cluster, same as MSPolicy)
  Stages 3-4       : deterministic tail — one expected-value node per
                     stage-2 leaf, computed as the mean of S_tail samples
                     from the stochastic process models

This gives a 5-stage effective lookahead (3 stochastic + 2 deterministic)
while keeping binary variables ≈2002 and solve time within budget.

Non-anticipativity holds by construction (node-indexed MILP).
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pyomo.environ import *
import sys, os

import v2_SystemCharacteristics as SC
from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels


# =============================================================================
# STEP 3a — Stochastic tree: Iterative Branch & Cluster
# =============================================================================

def build_scenario_tree(state, t_now, L, B, S_init):
    """
    Iterative Branch & Cluster for stages 0 … L-1.
    B can be int (uniform) or list [B_1, B_2, …] per stage.
    """
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
        'is_tail':    False,
    }
    nodes   = [root]
    next_id = 1
    queue   = [(0, 0)]

    while queue:
        parent_id, stage = queue.pop(0)
        if stage >= L - 1:
            continue

        parent = nodes[parent_id]
        B_k    = B_per_stage[stage]

        samples = np.empty((S_init, 3))
        for i in range(S_init):
            samples[i, 0] = price_model(parent['price'], parent['price_prev'])
            samples[i, 1], samples[i, 2] = next_occupancy_levels(
                parent['occ1'], parent['occ2']
            )

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
                'price_prev': parent['price'],
                'occ1':       float(centroids[k, 1]),
                'occ2':       float(centroids[k, 2]),
                'cond_prob':  float(cond_probs[k]),
                'prob':       parent['prob'] * float(cond_probs[k]),
                'is_tail':    False,
            }
            nodes.append(child)
            nodes[parent_id]['children'].append(next_id)
            queue.append((next_id, stage + 1))
            next_id += 1

    return nodes


# =============================================================================
# STEP 3b — Deterministic tail: extend each leaf with D expected-value nodes
# =============================================================================

def extend_with_deterministic_tail(nodes, D, S_tail, remaining, L_stoch):
    """
    For each leaf of the stochastic tree (stage = L_stoch - 1), append D
    deterministic nodes whose price and occupancy are the expected values
    of the stochastic process models conditioned on the leaf's state.

    Tail node probability = leaf probability (deterministic continuation).
    Only appends tail stages that fall within the remaining horizon.
    """
    if D <= 0:
        return nodes

    node_by_id = {n['id']: n for n in nodes}
    next_id    = max(n['id'] for n in nodes) + 1

    # Current leaves are nodes at stage L_stoch - 1
    current_leaves = [n for n in nodes if n['stage'] == L_stoch - 1]

    for _ in range(D):
        # Don't extend beyond the available horizon
        if current_leaves[0]['stage'] + 1 >= remaining:
            break

        new_leaves = []
        for leaf in current_leaves:
            # Compute expected next values from this leaf's state
            samples = np.empty((S_tail, 3))
            for i in range(S_tail):
                samples[i, 0] = price_model(leaf['price'], leaf['price_prev'])
                samples[i, 1], samples[i, 2] = next_occupancy_levels(
                    leaf['occ1'], leaf['occ2']
                )
            exp_price = float(np.mean(samples[:, 0]))
            exp_occ1  = float(np.mean(samples[:, 1]))
            exp_occ2  = float(np.mean(samples[:, 2]))

            tail_node = {
                'id':         next_id,
                'stage':      leaf['stage'] + 1,
                'parent':     leaf['id'],
                'children':   [],
                'price':      exp_price,
                'price_prev': leaf['price'],
                'occ1':       exp_occ1,
                'occ2':       exp_occ2,
                'cond_prob':  1.0,
                'prob':       leaf['prob'],   # same prob — deterministic step
                'is_tail':    True,
            }
            nodes.append(tail_node)
            node_by_id[next_id] = tail_node
            node_by_id[leaf['id']]['children'].append(next_id)
            new_leaves.append(tail_node)
            next_id += 1

        current_leaves = new_leaves

    return nodes


# =============================================================================
# HELPER
# =============================================================================

def _descendants_within(node_by_id, start_id, max_depth):
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
# STEP 5 — Node-indexed MILP over stochastic tree + deterministic tail
# =============================================================================

def solve_hybrid_milp(state, nodes, t_now, params):
    """
    Single node-indexed MILP covering both the stochastic tree and the
    deterministic tail. Non-anticipativity is implicit by construction.
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
    T_low     = params['temp_min_comfort_threshold']
    T_ok      = params['temp_OK_threshold']
    T_high    = params['temp_max_comfort_threshold']
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

    # ── State at non-root nodes ───────────────────────────────────────────────
    m.T1 = Var(non_root_ids, domain=Reals)
    m.T2 = Var(non_root_ids, domain=Reals)
    m.H  = Var(non_root_ids, domain=Reals)

    # ── Threshold detectors at non-root nodes ─────────────────────────────────
    m.yLow1  = Var(non_root_ids, domain=Binary)
    m.yLow2  = Var(non_root_ids, domain=Binary)
    m.yOK1   = Var(non_root_ids, domain=Binary)
    m.yOK2   = Var(non_root_ids, domain=Binary)
    m.yHigh1 = Var(non_root_ids, domain=Binary)
    m.yHigh2 = Var(non_root_ids, domain=Binary)
    m.u1     = Var(non_root_ids, domain=Binary)
    m.u2     = Var(non_root_ids, domain=Binary)

    # ── Objective ─────────────────────────────────────────────────────────────
    def obj_rule(m):
        c = price_now * (m.p1[0] + m.p2[0] + P_vent * m.v[0])
        for nid in non_root_ids:
            nd = node_by_id[nid]
            c += nd['prob'] * nd['price'] * (m.p1[nid] + m.p2[nid] + P_vent * m.v[nid])
        return c
    m.obj = Objective(rule=obj_rule, sense=minimize)

    # ── State dynamics ────────────────────────────────────────────────────────
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

    # ── Threshold detection ────────────────────────────────────────────────────
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

    # ── Overrule controller ───────────────────────────────────────────────────
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

    # ── Overrule at root ──────────────────────────────────────────────────────
    if low_or1 == 1:     m.p1[0].fix(P_max)
    if low_or2 == 1:     m.p2[0].fix(P_max)
    if T1_now >= T_high: m.p1[0].fix(0.0)
    if T2_now >= T_high: m.p2[0].fix(0.0)

    # ── Ventilation startup at root ───────────────────────────────────────────
    v_prev = 1 if vent_counter > 0 else 0
    m.c_sr1 = Constraint(expr= m.s[0] >= m.v[0] - v_prev)
    m.c_sr2 = Constraint(expr= m.s[0] <= m.v[0])
    m.c_sr3 = Constraint(expr= m.s[0] <= 1 - v_prev)

    # ── Ventilation startup at non-root nodes ─────────────────────────────────
    def c_s1(m, n): return m.s[n] >= m.v[n] - m.v[node_by_id[n]['parent']]
    def c_s2(m, n): return m.s[n] <= m.v[n]
    def c_s3(m, n): return m.s[n] <= 1 - m.v[node_by_id[n]['parent']]
    m.c_s1 = Constraint(non_root_ids, rule=c_s1)
    m.c_s2 = Constraint(non_root_ids, rule=c_s2)
    m.c_s3 = Constraint(non_root_ids, rule=c_s3)

    # ── Ventilation uptime ────────────────────────────────────────────────────
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

    # ── Ventilation inertia from current state ────────────────────────────────
    if vent_counter > 0:
        remaining_vent = U_vent - vent_counter
        if remaining_vent > 0:
            m.v[0].fix(1)
        for stage_k in range(1, remaining_vent):
            for nid in [n['id'] for n in nodes if n['stage'] == stage_k]:
                m.v[nid].fix(1)

    # ── Humidity overrule ─────────────────────────────────────────────────────
    if H_now >= H_high:
        m.v[0].fix(1)

    def c_hum(m, n): return m.H[n] <= H_high + M_hum * m.v[n]
    m.c_hum = Constraint(non_root_ids, rule=c_hum)

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 30
    solver.options['MIPGap']    = 0.01
    solver.solve(m, tee=False)

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

    if low_or1 == 1:     m.p1.fix(P_max)
    if low_or2 == 1:     m.p2.fix(P_max)
    if T1_now >= T_high: m.p1.fix(0.0)
    if T2_now >= T_high: m.p2.fix(0.0)
    if H_now  >= H_high: m.v.fix(1)
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

class HybridPolicy:
    """
    Hybrid: Multi-stage SP (stochastic tree, L=3, B=[20,3])
            + Deterministic expected-value tail (D=2 stages).

    Total effective lookahead: 5 stages.
    Binary variables: ~2002.
    """

    def __init__(self, L=3, B=None, D=2, S_init=500, S_tail=50):
        """
        L:      stochastic tree depth (stages 0 … L-1)
        B:      per-stage branching factor for the stochastic tree
        D:      number of deterministic tail stages appended to each leaf
        S_init: MC samples per node for the stochastic tree clustering
        S_tail: MC samples per leaf for computing expected tail values
        """
        self.L      = L
        self.B      = B if B is not None else [20, 3]
        self.D      = D
        self.S_init = S_init
        self.S_tail = S_tail
        self.params = SC.get_fixed_data()

    def select_action(self, state):
        t_now     = int(state['current_time'])
        T_total   = self.params['num_timeslots']
        remaining = T_total - t_now

        if remaining <= 1:
            p1, p2, v = solve_1stage(state, self.params)
            return {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}

        L_eff = min(self.L, remaining)

        # Step 3a: build stochastic scenario tree
        nodes = build_scenario_tree(state, t_now, L_eff, self.B, self.S_init)

        # Step 3b: extend leaves with deterministic expected-value tail
        D_eff = min(self.D, remaining - L_eff)
        if D_eff > 0:
            nodes = extend_with_deterministic_tail(
                nodes, D_eff, self.S_tail, remaining, L_eff
            )

        # Steps 4-5: solve node-indexed MILP (NA implicit)
        try:
            p1, p2, v = solve_hybrid_milp(state, nodes, t_now, self.params)
        except Exception as e:
            print(f"[HybridPolicy] solver failed at t={t_now}: {e}")
            p1, p2, v = 0.0, 0.0, 0

        return {
            "HeatPowerRoom1": p1 if p1 is not None else 0.0,
            "HeatPowerRoom2": p2 if p2 is not None else 0.0,
            "VentilationON":  v  if v  is not None else 0,
        }


# =============================================================================
# MODULE-LEVEL select_action (required by environment)
# =============================================================================

_policy = HybridPolicy(L=3, B=[20, 3], D=2, S_init=500, S_tail=50)

def select_action(state):
    return _policy.select_action(state)
