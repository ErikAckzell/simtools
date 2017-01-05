# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:34:03 2017

@author: erik
"""


import scipy


def res1(t, y, yp, sw):
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

    yp= [phi_sp,    0
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

    res_1 = (mS + mB) * yp[7] + mB * lS * yp[5] + mB * lG * yp[6] +\
        (mS + mB) * g
    res_2 = mB * lS * yp[7] + (JS + mB * lS ** 2) * yp[5] +\
        mB * lS * lG * yp[6] - cp * (y[1] - y[0]) + mB * lS * g + y[3]
    res_3 = mB * lG * yp[7] + mB * lS * lG * yp[5] +\
        (JB + mB * lG ** 2) * yp[6] - cp * (y[0] - y[1]) + mB * lG * g + y[4]
    res_4 = y[3]
    res_5 = y[4]

    return scipy.hstack((res_1, res_2, res_3, res_4, res_5)).flatten()


def res2(t, y, yp, sw):
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

    yp= [phi_sp,    0
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

    res_1 = (mS + mB) * yp[7] + mB * lS * yp[5] + mB * lG * yp[6] +\
        (mS + mB) * g + y[4]
    res_2 = mB * lS * yp[7] + (JS + mB * lS ** 2) * yp[5] +\
        mB * lS * lG * yp[6] - cp * (y[1] - y[0]) + mB * lS * g + hS * y[3] +\
        rS * y[4]
    res_3 = mB * lG * yp[7] + mB * lS * lG * yp[5] +\
        (JB + mB * lG ** 2) * yp[6] - cp * (y[0] - y[1]) + mB * lG * g
    res_4 = rS - r0 + hS * y[0]
    res_5 = y[7] + rS * y[5]

    return scipy.hstack((res_1, res_2, res_3, res_4, res_5)).flatten()


def res3(t, y, yp, sw):
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

    yp= [phi_sp,    0
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

    res_1 = (mS + mB) * yp[7] + mB * lS * yp[5] + mB * lG * yp[6] +\
        (mS + mB) * g + y[4]
    res_2 = mB * lS * yp[7] + (JS + mB * lS ** 2) * yp[5] +\
        mB * lS * lG * yp[6] - cp * (y[1] - y[0]) + mB * lS * g - hS * y[3] +\
        rS * y[4]
    res_3 = mB * lG * yp[7] + mB * lS * lG * yp[5] +\
        (JB + mB * lG ** 2) * yp[6] - cp * (y[0] - y[1]) + mB * lG * g
    res_4 = rS - r0 - hS * y[0]
    res_5 = y[7] + rS * y[5]

    return scipy.hstack((res_1, res_2, res_3, res_4, res_5)).flatten()
