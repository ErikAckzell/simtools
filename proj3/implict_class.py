# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:30:16 2017

@author: erik
"""


import assimulo
from assimulo.solvers import IDA
import scipy


class MyProblem(assimulo.problem.Implicit_Problem):
    def __init__(self):
        self.mS = 3.0e-4  # Mass of sleeve [kg]
        self.JS = 5.0e-9  # Moment of inertia of the sleeve [kgm]
        self.mB = 4.5e-3  # Mass of bird [kg]
        self.masstotal = self.mS + self.mB  # total mass
        self.JB = 7.0e-7  # Moment of inertia of bird [kgm]
        self.r0 = 2.5e-3  # Radius of the bar [m]
        self.rS = 3.1e-3  # Inner Radius of sleeve [m]
        self.hS = 5.8e-3  # 1/2 height of sleeve [m]
        self.lS = 1.0e-2  # verical distance sleeve origin to spring origin [m]
        self.lG = 1.5e-2  # vertical distance spring origin to bird origin [m]
        self.hB = 2.0e-2  # y coordinate beak (in bird coordinate system) [m]
        self.lB = 2.01e-2  # -x coordinate beak (in bird coordinate system) [m]
        self.cp = 5.6e-3  # rotational spring constant [N/rad]
        self.g = 9.81  #  [m/s^2]

        self.M = scipy.array([
                             [self.mS + self.mB, self.mB * self.lS, self.mB * self.lG],
                             [self.mB * self.lS, self.JS + self.mB * self.lS ** 2, self.mB * self.lS * self.lG],
                             [self.mB * self.lG, self.mB * self.lS * self.lG, self.JB + self.mB * self.lG ** 2]
                             ])

        t0 = 0
        y0 = scipy.zeros(10)
        yd0 = scipy.zeros(10)
        switches0 = [False, True, False]
        super().__init__(self.res, y0, yd0, t0, sw0=switches0)

    def res(self, t, y, yd, sw):
        """
        y = [z,         0
             phi_s,     1
             phi_b,     2
             lambda_1,  3
             lambda_2,  4
             phi_sp,    5
             phi_bp,    6
             zp,        7
             lambda_1p, 8
             lambda_2p] 9

        yd= [zp,        0
             phi_sp,    1
             phi_bp,    2
             lambda_1p, 3
             lambda_2p, 4
             zpp,       5
             phi_spp,   6
             phi_bpp,   7
             lambda_1pp,8
             lambda_2pp]9
        """

        if sw[0]:
            G = scipy.array([
                            [0, 1, 0],
                            [0, 0, 1]
                            ])
            gvec = y[3:5]
        elif sw[1]:
            G = scipy.array([
                            [0, self.hS, 0],
                            [1, self.rS, 0]
                            ])
            gvec = scipy.array([self.rS - self.r0 + self.hS * y[1],
                               y[5] + self.rS * y[6]])
        elif sw[2]:
            G = scipy.array([
                            [0, - self.hS, 0],
                            [1, self.rS, 0]
                            ])
            gvec = scipy.array([self.rS - self.r0 - self.hS * y[1],
                               y[5] + self.rS * y[6]])

        ff = scipy.array([- (self.mS + self.mB) * self.g,
                         self.cp * (y[2] - y[1]) - self.mB * self.lS * self.g,
                         self.cp * (y[1] - y[2]) - self.mB * self.lG * self.g])

        res_1 = yd[0:5] - y[5:10]
        res_2 = scipy.dot(self.M, yd[5:8]) - ff + scipy.dot(G.T, y[3:5])
        res_3 = gvec
        print('res_1: ', res_1)
        print('res_2: ', res_2)
        print('res_3: ', res_3)

        return scipy.hstack((res_1, res_2, res_3)).flatten()

    def state_events(self, t, y, yd, sw):
        e_0 = self.hS * y[0] + self.rS - self.r0  # switch to state 2
        e_1 = self.hS * y[0] - self.rS - self.r0  # switch to state 3
        e_2 = y[3]  # switch to state 1
        e_3 = y[3]  # switch to state 1
        e_4 = self.hB * y[1] - self.lS - self.lG + self.lB + self.r0  # beak hits the pole

        return scipy.array([e_0, e_1, e_2, e_3, e_4])

    def handle_event(self, solver, event_info):
        state_info = event_info[0]
        if state_info.any():
            if state_info[0] and solver.sw[0] and solver.y[6] < 0:
                solver.sw = [False, True, False]
            elif state_info[1] and solver.sw[0] and solver.y[6] > 0:
                solver.sw = [False, False, True]
            elif state_info[2] and solver.sw[1]:
                solver.sw = [True, False, False]
            elif state_info[3] and solver.sw[2] and solver.y[6] < 0:
                solver.sw = [True, False, False]
            elif state_info[4] and solver.sw[2] and solver.y[6] > 0:
                solver.y[1] = - solver.y[1]
                solver.yd[1] = - solver.yd[1]


if __name__ == '__main__':
    mod = MyProblem()
    sim = IDA(mod)

    sim.simulate(1)
