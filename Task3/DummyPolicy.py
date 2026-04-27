# -*- coding: utf-8 -*-
"""
Dummy Policy (Task 6)
======================
Never heats, never ventilates.
Leaves everything up to the overrule controllers.
This is the simplest possible policy and serves as
the baseline/lower bound for performance comparison.
"""

class DummyPolicy:
    def select_action(self, state):
        return {
            "HeatPowerRoom1": 0.0,  # never heats room 1
            "HeatPowerRoom2": 0.0,  # never heats room 2
            "VentilationON":  0     # never turns ventilation on
        }