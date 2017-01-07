# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:30:16 2017

@author: erik
"""


import assimulo
from assimulo.solvers import IDA
import scipy
import scipy.optimize
import matplotlib.pyplot


class MyProblem(assimulo.problem.Implicit_Problem):
    def __init__(self, y0):
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
        self.y0 = y0
        yd0 = scipy.concatenate((y0[5:10], scipy.zeros(5))).flatten()
        switches0 = [False, True, False]
        super().__init__(self.res, self.y0, yd0, t0, sw0=switches0)

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
#            print('state 1')
            G = scipy.array([
                            [0, 1, 0],
                            [0, 0, 1]
                            ])
            gvec = y[3:5]
        elif sw[1]:
#            print('state 2')
            G = scipy.array([
                            [0, self.hS, 0],
                            [1, self.rS, 0]
                            ])
            gvec = scipy.array([self.rS - self.r0 + self.hS * y[1],
                               y[5] + self.rS * y[6]])
        elif sw[2]:
#            print('state 3, phi_bp: ', y[7], 'res: ', self.hB * y[2] - self.lS - self.lG + self.lB + self.r0)
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

        return scipy.hstack((res_1, res_2, res_3)).flatten()

    def state_events(self, t, y, yd, sw):
        e_0 = self.hS * y[1] + self.rS - self.r0  # switch to state 2
        e_1 = self.hS * y[1] - self.rS - self.r0  # switch to state 3
        e_2 = y[3]  # switch to state 1
        e_3 = y[3]  # switch to state 1
        e_4 = self.hB * y[2] - self.lS - self.lG + self.lB + self.r0  # beak hits the pole

#        print(e_0, e_1, e_2, e_3, e_4)

        return scipy.array([e_0, e_1, e_2, e_3, e_4])

    def handle_event(self, solver, event_info):
        state_info = scipy.array(event_info[0])
#        print(state_info)
        if state_info.any():
            if state_info[0] and solver.sw[0] and solver.y[7] < 0:
                solver.sw = [False, True, False]
                I_m = self.mB * self.lG * solver.yd[0] + self.mB * self.lS * self.lG * solver.yd[1] + (self.JB + self.mB * self.lG ** 2) * solver.yd[2]
                solver.yd[0] = 0
                solver.yd[1] = 0
                solver.y[5] = 0
                solver.y[6] = 0
                solver.yd[2] = I_m / (self.JB + self.mB * self.lG ** 2)
                solver.y[7] = I_m / (self.JB + self.mB * self.lG ** 2)
                print('state 2')
            elif state_info[1] and solver.sw[0] and solver.y[7] > 0:
                solver.sw = [False, False, True]
                I_m = self.mB * self.lG * solver.yd[0] + self.mB * self.lS * self.lG * solver.yd[1] + (self.JB + self.mB * self.lG ** 2) * solver.yd[2]
                solver.yd[0] = 0
                solver.yd[1] = 0
                solver.y[5] = 0
                solver.y[6] = 0
                solver.yd[2] = I_m / (self.JB + self.mB * self.lG ** 2)
                solver.y[7] = I_m / (self.JB + self.mB * self.lG ** 2)
                print('state 3')
            elif state_info[2] and solver.sw[1]:
                solver.sw = [True, False, False]
                print('state 1')
            elif state_info[3] and solver.sw[2] and solver.y[7] < 0:
                solver.sw = [True, False, False]
            elif state_info[4] and solver.sw[2] and solver.y[7] > 0:
                solver.y[7] = - solver.y[7]
                solver.yd[2] = - solver.yd[2]
                print('state 1')


def res(y):
    """
    y = [phi_s, phi_b, lamb_1, lamb_2]
    """
    phi_s, phi_b, lamb_1, lamb_2 = y


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

    return scipy.array([(mS + mB) * g + lamb_2,
                        cp * (phi_b - phi_s) - mB * lS * g - hS * lamb_1 - rS * lamb_2,
                        cp * (phi_s - phi_b) - mB * lG * g,
                        rS - r0 + hS * phi_s])


def get_init_vals():
    return scipy.optimize.fsolve(res, scipy.zeros(4))

if __name__ == '__main__':
    phi_s0, phi_b0, lamb_1, lamb_2 = get_init_vals()
    y0 = scipy.array([0, phi_s0, phi_b0, lamb_1, lamb_2, 0, 0, 8, 0, 0])
    mod = MyProblem(y0)
    sim = IDA(mod)
    mod.algvar = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]

    sim.algvar = [1, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    sim.suppress_alg = True

#    sim.rtol = 1e-4
#    sim.atol = 1e-4

    t, y, ncp = sim.simulate(0.53)
#    sim.plot(mask=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    z = y[:, 0]
    phi_s = y[:, 1]
    phi_b = y[:, 2]
    phi_sp = y[:, 6]
    phi_bp = y[:, 7]
    matplotlib.pyplot.close('all')
    matplotlib.pyplot.plot(t, z, label='z')
    matplotlib.pyplot.plot(t, phi_s, label='phi_s')
    matplotlib.pyplot.plot(t, phi_b, label='phi_b')
    matplotlib.pyplot.plot(t, phi_sp, label='phi_sp')
    matplotlib.pyplot.plot(t, phi_bp, label='phi_bp')

    matplotlib.pyplot.legend()
    matplotlib.pyplot.show()


#    for y0 in [scipy.array([0, -0.1034, -0.2216] + [0] * 7), scipy.zeros(10),
#                           scipy.ones(10)]:
#        try:
#            sim.y0 = y0
#            sim.simulate(1)
#            sim.plot(mask=[1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
#        except:
#            print('failed')
#
