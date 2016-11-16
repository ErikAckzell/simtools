""" Module for all the different available step functions.
"""

def BDF4(self, tres, yres, h):
    """
    BDF-4 with Zero order predictor, using scipy.optimize.fsolve

    alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1 + alpha_3*y_nm2 + alpha_4*y_nm3 = h*f(t_np1,y_np1)
    alpha = [25/12, -4, 3, -4/3, 1/4]
    """
    alpha = [25/12, -4, 3, -4/3, 1/4]
    f=self.problem.rhs

    t_np1 = tres[-1] + h
    result = fsolve(lambda y: alpha[0] * y +
                              alpha[1] * yres[-1] +
                              alpha[2] * yres[-2] +
                              alpha[3] * yres[-3] +
                              alpha[4] * yres[-4] -
                              h * f(t_np1, y),
                              yres[-1],
                              xtol=self.tol,
                              full_output=1)
    if result[2] == 1:
        y_np1 = result[0]
        self.statistics["nfcns"] += result[1]['nfev']
        return t_np1, y_np1
    else:
        raise Explicit_ODE_Exception('fsolve did not find a solution')


def BDF3(self, tres, yres, h):
    """
    BDF-3 with Zero order predictor, using scipy.optimize.fsolve

    alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1 + alpha_3*y_nm2 = h*f(t_np1,y_np1)
    alpha = [11/6, -3, 3/2, -1/3]
    """
    alpha = [11./6., -3., 3./2., -1./3.]
    f=self.problem.rhs

    t_np1 = tres[-1] + h
    result = fsolve(lambda y: alpha[0] * y +
                              alpha[1] * yres[-1] +
                              alpha[2] * yres[-2] +
                              alpha[3] * yres[-3] -
                              h * f(t_np1, y),
                              yres[-1],
                              xtol=self.tol,
                              full_output=1)
    if result[2] == 1:
        y_np1 = result[0]
        self.statistics["nfcns"] += result[1]['nfev']
        return t_np1, y_np1
    else:
        raise Explicit_ODE_Exception('fsolve did not find a solution')


def EE(self, t, y, h, floatflag=False):
    """
    This calculates the next step in the integration with explicit Euler.
    """
    self.statistics["nfcns"] += 1

    f = self.problem.rhs
    if floatflag:
        return t + h, y + h * f(t, y)
    else:
        return t[-1] + h, y[-1] + h * f(t[-1], y[-1])


def BDF2_FPI(self, tres, yres, h):
    """
    BDF-2 with Zero order predictor, using FPI

    alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
    alpha=[3/2,-2,1/2]
    """
    alpha = [3./2., -2., 1./2]
    f = self.problem.rhs

    y_n, y_nm1 = yres[-2:]

    t_np1 = tres[-1] + h

    y_np1_i=y_n

    y = yres[-1]

    for i in range(self.maxit):
        self.statistics["nfcns"] += 1
        y_np1=(-(alpha[1]*yres[-1]+alpha[2]*yres[-2])+h*f(t_np1,y))/alpha[0]
        if SL.norm(y - y_np1) < self.tol:
            return t_np1, y_np1
        y = y_np1
    else:
        raise Explicit_ODE_Exception('Corrector could not converge within % iterations'%i)


def BDF2(self, tres, yres, h):
    """
    BDF-2 with Zero order predictor, using scipy.optimize.fsolve

    alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
    alpha=[3/2,-2,1/2]
    """
    alpha = [3./2., -2., 1./2]
    f = self.problem.rhs

    t_np1 = tres[-1] + h
    result = fsolve(lambda y: alpha[0] * y +
                              alpha[1] * yres[-1] +
                              alpha[2] * yres[-2] -
                              h * f(t_np1, y),
                              yres[-1],
                              xtol=self.tol,
                              full_output=1)
    if result[2] == 1:
        y_np1 = result[0]
        self.statistics["nfcns"] += result[1]['nfev']
        return t_np1, y_np1
    else:
        raise Explicit_ODE_Exception('fsolve did not find a solution')
