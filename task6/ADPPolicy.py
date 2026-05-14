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
import pandas as pd
import numpy as np
from pyomo.environ import (
    Binary,
    ConcreteModel,
    NonNegativeReals,
    Objective,
    SolverFactory,
    Var,
    minimize,
    value,
)



BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
GIVEN_DIR = os.path.join(BASE_DIR, "given")
sys.path.append(GIVEN_DIR)

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
        if low_r1 == 1 and T_r1 >= T_ok:
            low_r1 = 0
        if low_r1 == 1 and T_r1 < T_ok:
            model.p1.fix(P_max)
        if T_r1 > T_high:
            model.p1.fix(0)

        if T_r2 < T_low:
            low_r2 = 1
        if low_r2 == 1 and T_r2 >= T_ok:
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
