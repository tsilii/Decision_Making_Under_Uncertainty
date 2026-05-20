import numpy as np
from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint, RangeSet,
    NonNegativeReals, Reals, Binary, minimize, value, SolverFactory
)
from pyomo.opt import TerminationCondition

import v2_SystemCharacteristics as SC
from PriceProcessRestaurant     import price_model
from OccupancyProcessRestaurant import next_occupancy_levels


def build_deterministic_path(state, t_now, L, S_init=500):
    # B=1 special case of build_scenario_tree:
    # sample S_init points, take the mean (= single centroid, no KMeans needed)
    root = {
        'id':         0,
        'stage':      0,
        'parent':     None,
        'children':   [1] if L > 1 else [],
        'price':      float(state['price_t']),
        'price_prev': float(state['price_previous']),
        'occ1':       float(state['Occ1']),
        'occ2':       float(state['Occ2']),
        'prob':       1.0,
        'cond_prob':  1.0,
    }
    path = [root]

    for k in range(1, L):
        prev = path[k - 1]

        prices = np.empty(S_init)
        occ1s  = np.empty(S_init)
        occ2s  = np.empty(S_init)
        for i in range(S_init):
            prices[i]          = price_model(prev['price'], prev['price_prev'])
            occ1s[i], occ2s[i] = next_occupancy_levels(prev['occ1'], prev['occ2'])

        node = {
            'id':         k,
            'stage':      k,
            'parent':     k - 1,
            'children':   [k + 1] if k < L - 1 else [],
            'price':      float(np.mean(prices)),
            'price_prev': prev['price'],
            'occ1':       float(np.mean(occ1s)),
            'occ2':       float(np.mean(occ2s)),
            'prob':       1.0,
            'cond_prob':  1.0,
        }
        path.append(node)

    return path


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


def solve_milp(state, nodes, t_now, params):
    # Identical to solve_mssp in MSPolicy.
    # L nodes instead of 1+B+B^2, so solves in milliseconds.
    P_max     = params['heating_max_power']
    P_vent    = params['ventilation_power']
    zeta_exch = params['heat_exchange_coeff']
    zeta_loss = params['thermal_loss_coeff']
    zeta_conv = params['heating_efficiency_coeff']
    zeta_cool = params['heat_vent_coeff']
    zeta_occ  = params['heat_occupancy_coeff']
    eta_occ   = params['humidity_occupancy_coeff']
    eta_vent  = params['humidity_vent_coeff']
    T_low     = params['temp_min_comfort_threshold'] - 0.001
    T_high    = params['temp_max_comfort_threshold'] + 0.001
    T_ok      = params['temp_OK_threshold']
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

    m.p1 = Var(all_ids, domain=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(all_ids, domain=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(all_ids, domain=Binary)
    m.s  = Var(all_ids, domain=Binary)

    m.T1 = Var(non_root_ids, domain=Reals)
    m.T2 = Var(non_root_ids, domain=Reals)
    m.H  = Var(non_root_ids, domain=NonNegativeReals)

    m.yLow1  = Var(non_root_ids, domain=Binary)
    m.yLow2  = Var(non_root_ids, domain=Binary)
    m.yOK1   = Var(non_root_ids, domain=Binary)
    m.yOK2   = Var(non_root_ids, domain=Binary)
    m.yHigh1 = Var(non_root_ids, domain=Binary)
    m.yHigh2 = Var(non_root_ids, domain=Binary)
    m.u1     = Var(non_root_ids, domain=Binary)
    m.u2     = Var(non_root_ids, domain=Binary)

    def obj_rule(m):
        c = price_now * (m.p1[0] + m.p2[0] + P_vent * m.v[0])
        for nid in non_root_ids:
            nd = node_by_id[nid]
            c += nd['price'] * (m.p1[nid] + m.p2[nid] + P_vent * m.v[nid])
        return c
    m.obj = Objective(rule=obj_rule, sense=minimize)

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

    if T1_now < params['temp_min_comfort_threshold']:
        low_or1 = 1
    if low_or1 == 1:
        if T1_now > params['temp_OK_threshold']:
            low_or1 = 0
        else:
            m.p1[0].fix(P_max)
    if T1_now > params['temp_max_comfort_threshold']:
        m.p1[0].fix(0.0)

    if T2_now < params['temp_min_comfort_threshold']:
        low_or2 = 1
    if low_or2 == 1:
        if T2_now > params['temp_OK_threshold']:
            low_or2 = 0
        else:
            m.p2[0].fix(P_max)
    if T2_now > params['temp_max_comfort_threshold']:
        m.p2[0].fix(0.0)

    v_prev = 1 if vent_counter > 0 else 0
    m.c_sr1 = Constraint(expr= m.s[0] >= m.v[0] - v_prev)
    m.c_sr2 = Constraint(expr= m.s[0] <= m.v[0])
    m.c_sr3 = Constraint(expr= m.s[0] <= 1 - v_prev)

    def c_s1(m, n): return m.s[n] >= m.v[n] - m.v[node_by_id[n]['parent']]
    def c_s2(m, n): return m.s[n] <= m.v[n]
    def c_s3(m, n): return m.s[n] <= 1 - m.v[node_by_id[n]['parent']]
    m.c_s1 = Constraint(non_root_ids, rule=c_s1)
    m.c_s2 = Constraint(non_root_ids, rule=c_s2)
    m.c_s3 = Constraint(non_root_ids, rule=c_s3)

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

    if vent_counter > 0:
        remaining_vent = U_vent - vent_counter
        if remaining_vent > 0:
            m.v[0].fix(1)
        for stage_k in range(1, remaining_vent):
            for nid in [n['id'] for n in nodes if n['stage'] == stage_k]:
                m.v[nid].fix(1)

    if H_now > H_high:
        m.v[0].fix(1)

    def c_hum(m, n): return m.H[n] <= H_high + M_hum * m.v[n]
    m.c_hum = Constraint(non_root_ids, rule=c_hum)

    solver = SolverFactory('gurobi')
    solver.options['TimeLimit'] = 10
    solver.options['MIPGap']    = 0.001
    result = solver.solve(m, tee=False)

    if result.solver.termination_condition == TerminationCondition.maxTimeLimit:
        print('[DetPolicy] WARNING: Gurobi hit TimeLimit — returning best incumbent')

    try:
        return float(value(m.p1[0])), float(value(m.p2[0])), int(round(value(m.v[0])))
    except Exception:
        return 0.0, 0.0, 0


def solve_1stage(state, params):
    P_max  = params['heating_max_power']
    P_vent = params['ventilation_power']
    U_vent = params['vent_min_up_time']

    T1_now       = float(state['T1'])
    T2_now       = float(state['T2'])
    H_now        = float(state['H'])
    price_now    = float(state['price_t'])
    vent_counter = int(state['vent_counter'])
    low_or1      = int(state['low_override_r1'])
    low_or2      = int(state['low_override_r2'])

    m     = ConcreteModel()
    m.p1  = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.p2  = Var(domain=NonNegativeReals, bounds=(0, P_max))
    m.v   = Var(domain=Binary)
    m.s   = Var(domain=Binary)
    m.obj = Objective(
        expr=price_now * (m.p1 + m.p2 + P_vent * m.v),
        sense=minimize
    )

    v_prev = 1 if vent_counter > 0 else 0
    m.c_s1 = Constraint(expr= m.s >= m.v - v_prev)
    m.c_s2 = Constraint(expr= m.s <= m.v)
    m.c_s3 = Constraint(expr= m.s <= 1 - v_prev)

    if T1_now < params['temp_min_comfort_threshold']:
        low_or1 = 1
    if low_or1 == 1:
        if T1_now > params['temp_OK_threshold']:
            low_or1 = 0
        else:
            m.p1.fix(P_max)
    if T1_now > params['temp_max_comfort_threshold']:
        m.p1.fix(0.0)

    if T2_now < params['temp_min_comfort_threshold']:
        low_or2 = 1
    if low_or2 == 1:
        if T2_now > params['temp_OK_threshold']:
            low_or2 = 0
        else:
            m.p2.fix(P_max)
    if T2_now > params['temp_max_comfort_threshold']:
        m.p2.fix(0.0)

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


class DeterministicLookaheadPolicy:

    def __init__(self, L=6, S_init=500):
        self.L      = L
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
        nodes = build_deterministic_path(state, t_now, L_eff, self.S_init)

        try:
            p1, p2, v = solve_milp(state, nodes, t_now, self.params)
        except Exception as e:
            print(f'[DetPolicy] solver failed at t={t_now}: {e}')
            p1, p2, v = 0.0, 0.0, 0

        return {
            "HeatPowerRoom1": p1 if p1 is not None else 0.0,
            "HeatPowerRoom2": p2 if p2 is not None else 0.0,
            "VentilationON":  v  if v  is not None else 0,
        }


_policy = None

def select_action(state):
    global _policy
    if _policy is None:
        _policy = DeterministicLookaheadPolicy(L=6, S_init=500)
    return _policy.select_action(state)