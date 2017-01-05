# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:30:16 2017

@author: erik
"""


import assimulo
from assimulo.solvers import IDA
import scipy
import equations_of_motion


class MyProblem(assimulo.problem.Implicit_Problem):
    """
    y = [phi_s,     0
         phi_b,     1
         z,         2
         lambda_1,  3
         lambda_2,  4
         phi_sp,    5
         phi_bp,    6
         zp,        7
         lambda_1p, 8
         lambda_2p] 9

    yd= [phi_sp,    0
         phi_bp,    1
         zp,        2
         lambda_1p, 3
         lambda_2p, 4
         phi_spp,   5
         phi_bpp,   6
         zpp,       7
         lambda_1pp,8
         lambda_2pp]9
    """
    def __init__(self):
        self.mS = 3.0e-4 # Mass of sleeve [kg]
        self.JS = 5.0e-9 # Moment of inertia of the sleeve [kgm]
        self.mB = 4.5e-3 # Mass of bird [kg]
        self.JB = 7.0e-7 # Moment of inertia of bird [kgm]
        self.r0 = 2.5e-3 # Radius of the bar [m]
        self.rS = 3.1e-3 # Inner Radius of sleeve [m]
        self.hS = 5.8e-3 # 1/2 height of sleeve [m]
        self.lS = 1.0e-2 # verical distance sleeve origin to spring origin [m]
        self.lG = 1.5e-2 # vertical distance spring origin to bird origin [m]
        self.hB = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]
        self.lB = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
        self.cp = 5.6e-3 # rotational spring constant [N/rad]
        self.g  = 9.81 #  [m/s^2]
        t0 = 0
        y0 = scipy.zeros(10)
        yd0 = scipy.zeros(10)
        switches0 = [False, True, False]
        super().__init__(self.res, y0, yd0, t0, sw0=switches0)

    def res(self, t, y, yd, sw):
        if sw[0]:
            return self.res1(t, y, yd, sw)
        elif sw[1]:
            return self.res2(t, y, yd, sw)
        elif sw[2]:
            return self.res3(t, y, yd, sw)

    def res1(self, t, y, yd, sw):
        """
        y = [phi_s,     0
             phi_b,     1
             z,         2
             lambda_1,  3
             lambda_2,  4
             phi_sp,    5
             phi_bp,    6
             zp,        7
             lambda_1p, 8
             lambda_2p] 9

        yd= [phi_sp,    0
             phi_bp,    1
             zp,        2
             lambda_1p, 3
             lambda_2p, 4
             phi_spp,   5
             phi_bpp,   6
             zpp,       7
             lambda_1pp,8
             lambda_2pp]9
        """
        mS = 3.0e-4 # Mass of sleeve [kg]
        JS = 5.0e-9 # Moment of inertia of the sleeve [kgm]
        mB = 4.5e-3 # Mass of bird [kg]
        masstotal=mS+mB # total mass
        JB = 7.0e-7 # Moment of inertia of bird [kgm]
        r0 = 2.5e-3 # Radius of the bar [m]
        rS = 3.1e-3 # Inner Radius of sleeve [m]
        hS = 5.8e-3 # 1/2 height of sleeve [m]
        lS = 1.0e-2 # verical distance sleeve origin to spring origin [m]
        lG = 1.5e-2 # vertical distance spring origin to bird origin [m]
        hB = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]
        lB = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
        cp = 5.6e-3 # rotational spring constant [N/rad]
        g  = 9.81 #  [m/s^2]

        res_1 = (mS + mB) * yd[7] + mB * lS * yd[5] + mB * lG * yd[6] +\
            (mS + mB) * g
        res_2 = mB * lS * yd[7] + (JS + mB * lS ** 2) * yd[5] +\
            mB * lS * lG * yd[6] - cp * (y[1] - y[0]) + mB * lS * g + y[3]
        res_3 = mB * lG * yd[7] + mB * lS * lG * yd[5] +\
            (JB + mB * lG ** 2) * yd[6] - cp * (y[0] - y[1]) + mB * lG * g + y[4]
        res_4 = y[3]
        res_5 = y[4]

        return scipy.hstack((res_1, res_2, res_3, res_4, res_5)).flatten()


    def res2(self, t, y, yd, sw):
        """
        y = [phi_s,     0
             phi_b,     1
             z,         2
             lambda_1,  3
             lambda_2,  4
             phi_sp,    5
             phi_bp,    6
             zp,        7
             lambda_1p, 8
             lambda_2p] 9

        yd= [phi_sp,    0
             phi_bp,    1
             zp,        2
             lambda_1p, 3
             lambda_2p, 4
             phi_spp,   5
             phi_bpp,   6
             zpp,       7
             lambda_1pp,8
             lambda_2pp]9
        """
        mS = 3.0e-4 # Mass of sleeve [kg]
        JS = 5.0e-9 # Moment of inertia of the sleeve [kgm]
        mB = 4.5e-3 # Mass of bird [kg]
        masstotal=mS+mB # total mass
        JB = 7.0e-7 # Moment of inertia of bird [kgm]
        r0 = 2.5e-3 # Radius of the bar [m]
        rS = 3.1e-3 # Inner Radius of sleeve [m]
        hS = 5.8e-3 # 1/2 height of sleeve [m]
        lS = 1.0e-2 # verical distance sleeve origin to spring origin [m]
        lG = 1.5e-2 # vertical distance spring origin to bird origin [m]
        hB = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]
        lB = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
        cp = 5.6e-3 # rotational spring constant [N/rad]
        g  = 9.81 #  [m/s^2]

        res_1 = (mS + mB) * yd[7] + mB * lS * yd[5] + mB * lG * yd[6] +\
            (mS + mB) * g + y[4]
        res_2 = mB * lS * yd[7] + (JS + mB * lS ** 2) * yd[5] +\
            mB * lS * lG * yd[6] - cp * (y[1] - y[0]) + mB * lS * g + hS * y[3] +\
            rS * y[4]
        res_3 = mB * lG * yd[7] + mB * lS * lG * yd[5] +\
            (JB + mB * lG ** 2) * yd[6] - cp * (y[0] - y[1]) + mB * lG * g
        res_4 = rS - r0 + hS * y[0]
        res_5 = y[7] + rS * y[5]

        return scipy.hstack((res_1, res_2, res_3, res_4, res_5)).flatten()


    def res3(self, t, y, yd, sw):
        """
        y = [phi_s,     0
             phi_b,     1
             z,         2
             lambda_1,  3
             lambda_2,  4
             phi_sp,    5
             phi_bp,    6
             zp,        7
             lambda_1p, 8
             lambda_2p] 9

        yd= [phi_sp,    0
             phi_bp,    1
             zp,        2
             lambda_1p, 3
             lambda_2p, 4
             phi_spp,   5
             phi_bpp,   6
             zpp,       7
             lambda_1pp,8
             lambda_2pp]9
        """
        mS = 3.0e-4 # Mass of sleeve [kg]
        JS = 5.0e-9 # Moment of inertia of the sleeve [kgm]
        mB = 4.5e-3 # Mass of bird [kg]
        masstotal=mS+mB # total mass
        JB = 7.0e-7 # Moment of inertia of bird [kgm]
        r0 = 2.5e-3 # Radius of the bar [m]
        rS = 3.1e-3 # Inner Radius of sleeve [m]
        hS = 5.8e-3 # 1/2 height of sleeve [m]
        lS = 1.0e-2 # verical distance sleeve origin to spring origin [m]
        lG = 1.5e-2 # vertical distance spring origin to bird origin [m]
        hB = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]
        lB = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
        cp = 5.6e-3 # rotational spring constant [N/rad]
        g  = 9.81 #  [m/s^2]

        res_1 = (mS + mB) * yd[7] + mB * lS * yd[5] + mB * lG * yd[6] +\
            (mS + mB) * g + y[4]
        res_2 = mB * lS * yd[7] + (JS + mB * lS ** 2) * yd[5] +\
            mB * lS * lG * yd[6] - cp * (y[1] - y[0]) + mB * lS * g - hS * y[3] +\
            rS * y[4]
        res_3 = mB * lG * yd[7] + mB * lS * lG * yd[5] +\
            (JB + mB * lG ** 2) * yd[6] - cp * (y[0] - y[1]) + mB * lG * g
        res_4 = rS - r0 - hS * y[0]
        res_5 = y[7] + rS * y[5]

        return scipy.hstack((res_1, res_2, res_3, res_4, res_5)).flatten()


    def state_events(self, t, y, yd, sw):
        e_0 = self.hS * y[0] + self.rS - self.r0 # switch to state 2
        e_1 = self.hS * y[0] - self.rS - self.r0 # switch to state 3
        e_2 = y[3] # switch to state 1
        e_3 = y[3] # switch to state 1
        e_4 = self.hB * y[1] - self.lS - self.lG + self.lB + self.r0 # beak hits the pole

#        if sw[0]:
#            if y[6] < 0:
#                e[0] = self.hS * y[0] + self.rS - self.r0 # switch to state 2
#            elif y[6] > 0:
#                e[1] = self.hS * y[0] - self.rS - self.r0 # switch to state 3
#        elif sw[1]:
#            e[2] = y[3] # switch to state 1
#        elif sw[2]:
#            if y[6] < 0:
#                e[3] = y[3] # switch to state 1
#            elif y[6] > 0:
#                e[4] = self.hB * y[1] - self.lS - self.lG + self.lB + self.r0 # beak hits the pole
#        else:
#            pass

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
