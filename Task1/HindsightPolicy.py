# -*- coding: utf-8 -*-
"""
HindsightPolicy.py (Task 6)
============================
Optimal in Hindsight policy — wraps the Task 1 MILP as a policy class.

HOW IT WORKS:
  - Knows the full day's prices and occupancies in advance (reads CSV at t=0)
  - At t=0: solves the full MILP for the entire day → stores optimal plan
  - At each hour t: returns the pre-planned action for that hour
  - Serves as the BEST POSSIBLE benchmark — no real policy can beat this
"""

import numpy as np
import pandas as pd
import sys
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = "/Users/manostsili/Desktop/dtu/courses/decision making under uncertainty /assignment_DC"
GIVEN_DIR = os.path.join(BASE_DIR, "given")
DATA_DIR  = os.path.join(BASE_DIR, "data")
sys.path.insert(0, GIVEN_DIR)

from pyomo.environ import *
import v2_SystemCharacteristics as SC


class HindsightPolicy:
    """
    Optimal in Hindsight policy.

    At the start of each day (t=0), solves the full MILP with perfect
    knowledge of the entire day's prices and occupancies.
    Stores the optimal plan, then executes it hour by hour.
    """

    def __init__(self):
        """Load CSV data once at initialization."""
        self.price_df = pd.read_csv(os.path.join(DATA_DIR, "v2_PriceData.csv"))
        self.occ1_df  = pd.read_csv(os.path.join(DATA_DIR, "OccupancyRoom1.csv"))
        self.occ2_df  = pd.read_csv(os.path.join(DATA_DIR, "OccupancyRoom2.csv"))
        self.params   = SC.get_fixed_data()

        self.planned_p1  = None
        self.planned_p2  = None
        self.planned_v   = None
        self.current_day = -1


    def select_action(self, state):
        """
        Called by the environment at each hour.
        At t=0: solve MILP for full day → store plan.
        At t>0: return pre-planned action.
        """
        t = state["current_time"]

        # ── New day: solve MILP with full knowledge ───────────────────────────
        if t == 0:
            self.current_day += 1
            day_idx = self.current_day

            price_row = self.price_df.iloc[day_idx].values
            prices    = price_row[1:]                          # hours t=0..9
            occ1      = self.occ1_df.iloc[day_idx].values     # hours t=0..9
            occ2      = self.occ2_df.iloc[day_idx].values     # hours t=0..9

            self._solve_milp(prices, occ1, occ2)

        # ── Return pre-planned action for this hour ───────────────────────────
        return {
            "HeatPowerRoom1": self.planned_p1[t],
            "HeatPowerRoom2": self.planned_p2[t],
            "VentilationON":  self.planned_v[t]
        }


    def _solve_milp(self, prices, occ1, occ2):
        """Solves Task 1 MILP and stores the optimal plan."""
        params    = self.params
        T         = params['num_timeslots']
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
        T1_init   = params['T1']
        T2_init   = params['T2']
        H_init    = params['H']
        M_temp    = 100
        M_hum     = 200
        R         = [1, 2]

        occ = {(1, t): occ1[t] for t in range(T)}
        occ.update({(2, t): occ2[t] for t in range(T)})

        model        = ConcreteModel()
        model.T      = RangeSet(0, T - 1)
        model.R      = Set(initialize=[1, 2])
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
                prices[t] * (sum(model.p[r, t] for r in model.R) + P_vent * model.v[t])
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

        self.planned_p1 = [value(model.p[1, t]) for t in range(T)]
        self.planned_p2 = [value(model.p[2, t]) for t in range(T)]
        self.planned_v  = [round(value(model.v[t])) for t in range(T)]