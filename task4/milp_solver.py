import numpy as np
from pyomo.environ import (
    ConcreteModel, RangeSet, Set, Var, Objective, Constraint,
    NonNegativeReals, Reals, Binary, SolverFactory, value, minimize
)


def solve_milp_with_results(prices, occ1, occ2, params):
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
        return sum(
            prices[t] * (sum(model.p[r, t] for r in R) + P_vent * model.v[t])
            for t in model.T
        )
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
