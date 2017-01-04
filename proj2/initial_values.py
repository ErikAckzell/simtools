# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 12:11:28 2017

@author: erik
"""

import scipy
import scipy.linalg


def g(theta, y):
    """
    Index-3 constraints
    """

    # Geometry
    xa, ya = -.06934, -.00227
    xb, yb = -0.03635, .03273
    d = 28.e-3
    e = 2.e-2

    rr = 7.e-3
    ss = 35.e-3
    u = 4.e-2
    zf, zt = 2.e-2, 4.e-2
    # Initial computations and assignments

    beta, gamma, phi, delta, omega, epsilon = y[0:6]
    sibe, siga, siph, side, siom, siep = scipy.sin(y[0:6])
    cobe, coga, coph, code, coom, coep = scipy.cos(y[0:6])

    sibeth = scipy.sin(beta+theta)
    cobeth = scipy.cos(beta+theta)
    siphde = scipy.sin(phi+delta)
    cophde = scipy.cos(phi+delta)
    siomep = scipy.sin(omega+epsilon)
    coomep = scipy.cos(omega+epsilon)

    #     Index-3 constraint
    g = scipy.zeros((6,))
    g[0] = rr*cobe - d*cobeth - ss*siga - xb
    g[1] = rr*sibe - d*sibeth + ss*coga - yb
    g[2] = rr*cobe - d*cobeth - e*siphde - zt*code - xa
    g[3] = rr*sibe - d*sibeth + e*cophde - zt*side - ya
    g[4] = rr*cobe - d*cobeth - zf*coomep - u*siep - xa
    g[5] = rr*sibe - d*sibeth - zf*siomep + u*coep - ya

    return g


def get_init_vals():
    theta = 0
    y = scipy.zeros(7)
    y[1] = theta
    y[0], y[2], y[3], y[4], y[5], y[6] = scipy.optimize.fsolve(lambda x: g(theta, x), scipy.zeros(6))
    return y


if __name__ == '__main__':
    print(get_init_vals())
