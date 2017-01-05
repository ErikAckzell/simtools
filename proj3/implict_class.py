# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:30:16 2017

@author: erik
"""


import assimulo


class MySolver(assimulo.problem.Implicit_Problem):
    def state_events(self, t, y, yd, sw):
        pass

    def handle_event(self, solver, event_info):
        pass
