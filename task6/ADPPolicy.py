# =============================================================================
# Task 4 - ADP/VFA Policy
# =============================================================================
# Clean policy wrapper for task6/Environment.py.
#
# This is the same one-step ADP decision logic used in policy_test_task4,
# without the out-of-sample loop, MILP benchmark comparison, or printing.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Reals,
    RangeSet,
    Set,
    SolverFactory,
    Var,
    minimize,
    value,
)

# =============================================================================
# 0. Paths – assume this script lives in the 'task4' folder
# =============================================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GIVEN_DIR = os.path.join(BASE_DIR, "given")
OUT_OF_SAMPLE_DIR = os.path.join(BASE_DIR, "out_of_sample")
sys.path.append(GIVEN_DIR)
sys.path.append(BASE_DIR)
sys.path.append(OUT_OF_SAMPLE_DIR)

import v2_SystemCharacteristics as sc


class ADPPolicy:
    """
    ADP policy using the trained VFA weights from task4/eta.npy.

    The environment calls select_action(state) at each timeslot. This policy
    solves the same one-step lookahead problem from policy_test_task4:

        min immediate cost - eta[t]' phi(s_{t+1})
    """

    def __init__(self):
        self.params = sc.get_fixed_data()
        self.eta = np.load(os.path.join(BASE_DIR, "task4", "eta.npy"))

    def select_action(self, state):
        """
        Return the action dictionary expected by task6/Environment.py.
        """
        t = int(state["current_time"])

        p1, p2, v = self.solve_adp_step(
            t=t,
            T_r1=state["T1"],
            T_r2=state["T2"],
            H=state["H"],
            price=state["price_t"],
            occ1=state["Occ1"],
            occ2=state["Occ2"],
            vent_counter=state["vent_counter"],
            low_r1=state["low_override_r1"],
            low_r2=state["low_override_r2"],
        )

        return {
            "HeatPowerRoom1": p1,
            "HeatPowerRoom2": p2,
            "VentilationON": v,
        }

    def solve_adp_step(self, t, T_r1, T_r2, H, price, occ1, occ2,
                       vent_counter, low_r1, low_r2):
        """
        Same one-step ADP subproblem as policy_test_task4.
        Returns only the action, since Environment.py handles state updates.
        """
        params = self.params

        P_max = float(params["heating_max_power"])
        P_vent = float(params["ventilation_power"])
        zeta_exch = float(params["heat_exchange_coeff"])
        zeta_loss = float(params["thermal_loss_coeff"])
        zeta_conv = float(params["heating_efficiency_coeff"])
        zeta_cool = float(params["heat_vent_coeff"])
        zeta_occ = float(params["heat_occupancy_coeff"])
        eta_occ = float(params["humidity_occupancy_coeff"])
        eta_vent = float(params["humidity_vent_coeff"])
        T_out = list(params["outdoor_temperature"])

        model = ConcreteModel()
        model.p1 = Var(domain=NonNegativeReals, bounds=(0, P_max))
        model.p2 = Var(domain=NonNegativeReals, bounds=(0, P_max))
        model.v = Var(domain=Binary)

        T_low = float(params["temp_min_comfort_threshold"])
        T_ok = float(params["temp_OK_threshold"])
        T_high = float(params["temp_max_comfort_threshold"])
        H_high = float(params["humidity_threshold"])
        U_vent = int(params["vent_min_up_time"])

        if H > H_high or (1 <= vent_counter <= U_vent - 1):
            model.v.fix(1)

        if T_r1 < T_low:
            low_r1 = 1
        if low_r1 == 1 and T_r1 > T_ok:
            low_r1 = 0
        if low_r1 == 1 and T_r1 < T_ok:
            model.p1.fix(P_max)
        if T_r1 > T_high:
            model.p1.fix(0)

        if T_r2 < T_low:
            low_r2 = 1
        if low_r2 == 1 and T_r2 > T_ok:
            low_r2 = 0
        if low_r2 == 1 and T_r2 < T_ok:
            model.p2.fix(P_max)
        if T_r2 > T_high:
            model.p2.fix(0)

        def next_T1(m):
            return (
                T_r1
                + zeta_exch * (T_r2 - T_r1)
                - zeta_loss * (T_r1 - T_out[t])
                + zeta_conv * m.p1
                - zeta_cool * m.v
                + zeta_occ * occ1
            )

        def next_T2(m):
            return (
                T_r2
                + zeta_exch * (T_r1 - T_r2)
                - zeta_loss * (T_r2 - T_out[t])
                + zeta_conv * m.p2
                - zeta_cool * m.v
                + zeta_occ * occ2
            )

        def next_H(m):
            return H + eta_occ * (occ1 + occ2) - eta_vent * m.v

        w = self.eta[t]

        def obj_rule(m):
            return (
                price * (m.p1 + m.p2 + P_vent * m.v)
                - w[0]
                - w[1] * next_T1(m)
                - w[2] * next_T2(m)
                - w[3] * next_H(m)
                - w[4] * price
                - w[5] * occ1
                - w[6] * occ2
            )

        model.obj = Objective(rule=obj_rule, sense=minimize)

        solver = SolverFactory("gurobi")
        solver.solve(model, tee=False)

        return value(model.p1), value(model.p2), round(value(model.v))


# =============================================================================
# 2. MILP solver for training (hindsight‑optimal trajectory)
# =============================================================================
def solve_milp_with_results(prices, occ1, occ2, params):
    """
    Solves the full‑horizon MILP (Task 1) for one day.
    Returns a dict with all state and action trajectories.
    """
    T = int(params['num_timeslots'])
    R = [1, 2]
    times = range(T)

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
    T1_init   = float(params['T1'])
    T2_init   = float(params['T2'])
    H_init    = float(params['H'])
    M_temp    = 100
    M_hum     = 200

    occ = {(1, t): occ1[t] for t in times}
    occ.update({(2, t): occ2[t] for t in times})

    model = ConcreteModel()
    model.T = RangeSet(0, T - 1)
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

    def obj_rule(model):
        return sum(prices[t] * (sum(model.p[r, t] for r in R) + P_vent * model.v[t])
                   for t in model.T)
    model.obj = Objective(rule=obj_rule, sense=minimize)

    model.init_T1 = Constraint(expr=model.Temp[1, 0] == T1_init)
    model.init_T2 = Constraint(expr=model.Temp[2, 0] == T2_init)
    model.init_H  = Constraint(expr=model.H[0] == H_init)

    def temp_dynamics(model, r, t):
        if t == 0:
            return Constraint.Skip
        r_other = 2 if r == 1 else 1
        return model.Temp[r, t] == (
            model.Temp[r, t-1]
            + zeta_exch * (model.Temp[r_other, t-1] - model.Temp[r, t-1])
            - zeta_loss * (model.Temp[r, t-1] - T_out[t-1])
            + zeta_conv * model.p[r, t-1]
            - zeta_cool * model.v[t-1]
            + zeta_occ  * occ[r, t-1]
        )
    model.temp_dynamics = Constraint(model.R, model.T, rule=temp_dynamics)

    def hum_dynamics(model, t):
        if t == 0:
            return Constraint.Skip
        return model.H[t] == (
            model.H[t-1]
            + eta_occ * sum(occ[r, t-1] for r in R)
            - eta_vent * model.v[t-1]
        )
    model.hum_dynamics = Constraint(model.T, rule=hum_dynamics)

    # Temperature zone classification (y_low, y_ok, y_high)
    def y_high_upper(model, r, t):
        return model.Temp[r, t] >= T_high - M_temp * (1 - model.y_high[r, t])
    model.y_high_upper = Constraint(model.R, model.T, rule=y_high_upper)

    def y_high_lower(model, r, t):
        return model.Temp[r, t] <= T_high + M_temp * model.y_high[r, t]
    model.y_high_lower = Constraint(model.R, model.T, rule=y_high_lower)

    def y_low_upper(model, r, t):
        return model.Temp[r, t] <= T_low + M_temp * (1 - model.y_low[r, t])
    model.y_low_upper = Constraint(model.R, model.T, rule=y_low_upper)

    def y_low_lower(model, r, t):
        return model.Temp[r, t] >= T_low - M_temp * model.y_low[r, t]
    model.y_low_lower = Constraint(model.R, model.T, rule=y_low_lower)

    def y_ok_upper(model, r, t):
        return model.Temp[r, t] >= T_ok - M_temp * (1 - model.y_ok[r, t])
    model.y_ok_upper = Constraint(model.R, model.T, rule=y_ok_upper)

    def y_ok_lower(model, r, t):
        return model.Temp[r, t] <= T_ok + M_temp * model.y_ok[r, t]
    model.y_ok_lower = Constraint(model.R, model.T, rule=y_ok_lower)

    # Override logic (u[r, t])
    def overrule_trigger(model, r, t):
        return model.u[r, t] >= model.y_low[r, t]
    model.overrule_trigger = Constraint(model.R, model.T, rule=overrule_trigger)

    def overrule_memory(model, r, t):
        if t == 0:
            return Constraint.Skip
        return model.u[r, t] <= model.u[r, t-1] + model.y_low[r, t]
    model.overrule_memory = Constraint(model.R, model.T, rule=overrule_memory)

    def overrule_persist(model, r, t):
        if t == 0:
            return Constraint.Skip
        return model.u[r, t] >= model.u[r, t-1] - model.y_ok[r, t]
    model.overrule_persist = Constraint(model.R, model.T, rule=overrule_persist)

    def overrule_deactivate(model, r, t):
        if t == 0:
            return Constraint.Skip
        return model.u[r, t] <= 1 - model.y_ok[r, t]
    model.overrule_deactivate = Constraint(model.R, model.T, rule=overrule_deactivate)

    def overrule_max(model, r, t):
        return model.p[r, t] >= P_max * model.u[r, t]
    model.overrule_max = Constraint(model.R, model.T, rule=overrule_max)

    def overrule_zero(model, r, t):
        return model.p[r, t] <= P_max * (1 - model.y_high[r, t])
    model.overrule_zero = Constraint(model.R, model.T, rule=overrule_zero)

    # Ventilation startup and minimum up‑time
    def startup_detect1(model, t):
        return model.s[t] >= model.v[t] - (model.v[t-1] if t > 0 else 0)
    model.startup_detect1 = Constraint(model.T, rule=startup_detect1)

    def startup_detect2(model, t):
        return model.s[t] <= model.v[t]
    model.startup_detect2 = Constraint(model.T, rule=startup_detect2)

    def startup_detect3(model, t):
        if t == 0:
            return Constraint.Skip
        return model.s[t] <= 1 - model.v[t-1]
    model.startup_detect3 = Constraint(model.T, rule=startup_detect3)

    def vent_uptime(model, t):
        horizon  = min(t + U_vent, T)
        duration = min(U_vent, T - t)
        return sum(model.v[tau] for tau in range(t, horizon)) >= duration * model.s[t]
    model.vent_uptime = Constraint(model.T, rule=vent_uptime)

    def hum_vent(model, t):
        return model.H[t] <= H_high + M_hum * model.v[t]
    model.hum_vent = Constraint(model.T, rule=hum_vent)

    solver = SolverFactory('gurobi')
    solver.solve(model, tee=False)

    return {
        'Temp_r1':         [value(model.Temp[1, t]) for t in range(T)],
        'Temp_r2':         [value(model.Temp[2, t]) for t in range(T)],
        'Hum':             [value(model.H[t])       for t in range(T)],
        'h_r1':            [value(model.p[1, t])    for t in range(T)],
        'h_r2':            [value(model.p[2, t])    for t in range(T)],
        'v':               [value(model.v[t])       for t in range(T)],
        'low_override_r1': [int(round(value(model.u[1, t]))) for t in range(T)],
        'low_override_r2': [int(round(value(model.u[2, t]))) for t in range(T)],
        'price':           list(prices),
        'Occ_r1':          list(occ1),
        'Occ_r2':          list(occ2),
    }


# =============================================================================
# 3. Feature vector definition
# =============================================================================
def state(T_r1, T_r2, H, price, occ1, occ2):
    params = sc.get_fixed_data()
    T_LOW = float(params['temp_min_comfort_threshold'])
    return np.array([
        1.0, T_r1, T_r2, H, price, occ1, occ2,
    ])

N_FEATURES = 7


# =============================================================================
# 4. Training procedure (executed when script is run directly)
# =============================================================================
if __name__ == "__main__":
    params  = sc.get_fixed_data()
    T_TOTAL = int(params['num_timeslots'])

    # Load training data
    price_data = pd.read_csv(os.path.join(OUT_OF_SAMPLE_DIR, 'OutOfSamplePriceData.csv'))
    occ_room1  = pd.read_csv(os.path.join(OUT_OF_SAMPLE_DIR, 'OutOfSampleOccupancyRoom1.csv'))
    occ_room2  = pd.read_csv(os.path.join(OUT_OF_SAMPLE_DIR, 'OutOfSampleOccupancyRoom2.csv'))

    print(f"Data loaded — Prices: {price_data.shape}, Occ1: {occ_room1.shape}, Occ2: {occ_room2.shape}")

    # Collect hindsight‑optimal trajectories
    N_DAYS = len(price_data)
    trajectories = []
    for day in range(N_DAYS):
        prices = price_data.iloc[day, :].values
        occ1   = occ_room1.iloc[day, :].values
        occ2   = occ_room2.iloc[day, :].values
        traj = solve_milp_with_results(prices, occ1, occ2, params)
        trajectories.append(traj)
    print(f"\nTrajectories collected: {len(trajectories)} days x {T_TOTAL} timeslots")

    # Build V* targets and feature matrices
    N      = len(trajectories)
    V_star = np.zeros((N, T_TOTAL))
    State  = np.zeros((T_TOTAL, N, N_FEATURES))

    for n, traj in enumerate(trajectories):
        prices = np.array(traj['price'])
        h_r1   = np.array(traj['h_r1'])
        h_r2   = np.array(traj['h_r2'])
        v      = np.array(traj['v'])
        T_r1_s = np.array(traj['Temp_r1'])
        T_r2_s = np.array(traj['Temp_r2'])
        H_s    = np.array(traj['Hum'])
        occ1_s = np.array(traj['Occ_r1'])
        occ2_s = np.array(traj['Occ_r2'])

        P_vent = float(params['ventilation_power'])
        cost_per_hour = prices * (h_r1 + h_r2 + P_vent * v)

        for t in range(T_TOTAL):
            V_star[n, t] = -np.sum(cost_per_hour[t+1:])
            if t < T_TOTAL - 1:
                State[t, n] = state(T_r1_s[t+1], T_r2_s[t+1], H_s[t+1],
                                    prices[t], occ1_s[t], occ2_s[t])
            else:
                State[t, n] = state(T_r1_s[t], T_r2_s[t], H_s[t],
                                    prices[t], occ1_s[t], occ2_s[t])

    # OLS regression per timeslot (you can replace with Ridge if desired)
    #eta = np.zeros((T_TOTAL, N_FEATURES))
    #for t in range(T_TOTAL):
        #eta[t], _, _, _ = np.linalg.lstsq(State[t], V_star[:, t], rcond=None)
       # print(f"t={t}: eta = {eta[t].round(3)}")

    # Alternative Ridge (commented out)
    # from sklearn.linear_model import Ridge
    from sklearn.linear_model import Ridge

    # --- Ridge regression per timeslot (initial fit) ---
    eta = np.zeros((T_TOTAL, N_FEATURES))
    ridge = Ridge(alpha=1.0, fit_intercept=False)

    for t in range(T_TOTAL):
        ridge.fit(State[t], V_star[:, t])
        eta[t] = ridge.coef_
        print(f"t={t}: eta = {eta[t].round(3)}")

    # Save initial weights
    output_path = os.path.join(BASE_DIR, 'task4', 'eta.npy')
    np.save(output_path, eta)
    print(f"\nWeights saved to '{output_path}' — shape: {eta.shape}")

    # In‑sample R² diagnostic
    for t in range(T_TOTAL):
        pred   = State[t] @ eta[t]
        ss_res = np.sum((V_star[:, t] - pred) ** 2)
        ss_tot = np.sum((V_star[:, t] - V_star[:, t].mean()) ** 2)
        r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
        print(f"t={t}: R² = {r2:.4f}")

    # --- FVI iterations with Ridge refit ---
    N_ITER = 30

    for iteration in range(N_ITER):
        V_fvi = np.zeros((N, T_TOTAL))

        for t in range(T_TOTAL - 1, -1, -1):  # backwards like slide 44
            for n, traj in enumerate(trajectories):
                prices = np.array(traj['price'])
                occ1_s = np.array(traj['Occ_r1'])
                occ2_s = np.array(traj['Occ_r2'])
                T_r1_s = np.array(traj['Temp_r1'])
                T_r2_s = np.array(traj['Temp_r2'])
                H_s    = np.array(traj['Hum'])
                low_r1 = np.array(traj['low_override_r1'])
                low_r2 = np.array(traj['low_override_r2'])

                if t == T_TOTAL - 1:
                    # Terminal: no future value
                    V_fvi[n, t] = 0.0
                else:
                    P_vent = float(params['ventilation_power'])
                    milp_cost = prices[t] * (traj['h_r1'][t] + traj['h_r2'][t] + P_vent * traj['v'][t])
                    next_phi = State[t + 1, n]
                    V_fvi[n, t] = -milp_cost + eta[t + 1] @ next_phi

            # Ridge refit on new targets
            ridge.fit(State[t], V_fvi[:, t])
            eta[t] = ridge.coef_
            print(f"FVI iter {iteration+1}, t={t}: eta = {eta[t].round(3)}")

    np.save(output_path, eta)
    print(f"FVI weights saved — shape: {eta.shape}")
    ridge = Ridge(alpha=1.0, fit_intercept=False)
    for t in range(T_TOTAL):
        ridge.fit(State[t], V_star[:, t])
        eta[t] = ridge.coef_
        print(f"t={t}: eta = {eta[t].round(3)}")

    