from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
from scipy.optimize import fsolve
import scipy
import unittest
from assimulo.solvers import CVode


class BDF(Explicit_ODE):
    """
    BDF methods
    """
    tol=1.e-8
    maxit=1000
    maxsteps=5000

    def __init__(self, problem, order=3, jacobian=None):
        Explicit_ODE.__init__(self, problem) #Calls the base class

        #Solver options
        self.options["h"] = 0.01

        #Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

        self.jac = jacobian
        self.order = order

        if order == 2:
            self.step_BDF = self.step_BDF2
        elif order == 3:
            self.step_BDF = self.step_BDF3
        elif order == 4:
            self.step_BDF = self.step_BDF4
        else:
            raise ValueError('Only BDF methods of order 2-4 are implemented')

    def _set_h(self,h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h=property(_get_h,_set_h)

    def integrate(self, t, y, tf, opts):
        """
        _integrates (t,y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))

        #Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            if i<self.order:  # initial steps
                t_np1, y_np1 = self.step_EE(t, y, h)
                y = y_np1
            else:
                t_np1, y_np1 = self.step_BDF(tres[-self.order:],
                                              yres[-self.order:],
                                              h)
            tres.append(t_np1)
            yres.append(y_np1.copy())
            t = t_np1
            h=min(self.h,np.abs(tf-t))
        else:
            raise Explicit_ODE_Exception('Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres

    def step_EE(self, t, y, h):
        """
        This calculates the next step in the integration with explicit Euler.
        """
        self.statistics["nfcns"] += 1

        f = self.problem.rhs
        return t + h, y + h*f(t, y)

    def step_BDF2(self, tres, yres, h):
        """
        BDF-2 with Zero order predictor, using scipy.optimize.fsolve

        alpha_0*y_np1+alpha_1*y_n+alpha_2*y_nm1=h f(t_np1,y_np1)
        alpha=[3/2,-2,1/2]
        """
        alpha=[3./2.,-2.,1./2]
        f=self.problem.rhs

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


    def step_BDF3(self, tres, yres, h):
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

    def step_BDF4(self, tres, yres, h):
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

    def print_statistics(self, verbose=NORMAL):
        self.log_message('Final Run Statistics            : {name} \n'.format(name=self.problem.name),        verbose)
        self.log_message(' Step-length                    : {stepsize} '.format(stepsize=self.options["h"]), verbose)
        self.log_message(' Number of Steps                : '+str(self.statistics["nsteps"]),          verbose)
        self.log_message(' Number of Function Evaluations : '+str(self.statistics["nfcns"]),         verbose)

        self.log_message('\nSolver options:\n',                                    verbose)
        self.log_message(' Solver            : BDF{}'.format(self.order),                     verbose)
        self.log_message(' Solver type       : Fixed step\n',                      verbose)

#TODO: remove first pend function.
def pend(t,y):
    # came with the example
    #g=9.81    l=0.7134354980239037
    gl=13.7503671
    return np.array([y[1],-gl*np.sin(y[0])])

#Define another Assimulo problem
def pend(t, y, k=1):
    """
    This is the right hand side function of the differential equation
    describing the elastic pendulum
    y: 1x4 array
    k: float
    """
    yprime = scipy.array([y[2],
                          y[3],
                          -y[0] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                          (scipy.sqrt(y[0] ** 2 + y[1] ** 2)),
                          -y[1] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                          (scipy.sqrt(y[0] ** 2 + y[1] ** 2)) - 1])
    return yprime

class BDFtests(unittest.TestCase):
    def setUp(self):
        self.y0 = scipy.array([0.9, 0.1, 0, 0])
        self.klist = [0] + [10 ** i for i in range(0, 4)]
        self.orderlist = [2, 3, 4]

    def get_pendulum_rhs(self, k):
        def pend(t, y, k=k):
            """
            This is the right hand side function of the differential equation
            describing the elastic pendulum
            y: 1x4 array
            k: float
            """
            yprime = scipy.array([y[2],
                                  y[3],
                                  -y[0] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                                  (scipy.sqrt(y[0] ** 2 + y[1] ** 2)),
                                  -y[1] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                                  (scipy.sqrt(y[0] ** 2 + y[1] ** 2)) - 1])
            return yprime
        return pend

    def test_order_4_varying_k(self):
        order = 4
        for k in self.klist:
            pend = self.get_pendulum_rhs(k)
            pend_mod = Explicit_Problem(pend, y0=self.y0)
            pend_mod.name = 'Nonlinear Pendulum, k = {}'.format(k)
            exp_sim = BDF(pend_mod, order=order) #Create a BDF solver
            t, y = exp_sim.simulate(5)
            exp_sim.plot(mask=[1, 1, 0, 0])
            mpl.show()

    def test_varying_order_k_1000(self):
        k = 1000
        pend = self.get_pendulum_rhs(k)
        for order in self.orderlist:
            pend_mod = Explicit_Problem(pend, y0=self.y0)
            pend_mod.name = 'Nonlinear Pendulum, k = {}'.format(k)
            exp_sim = BDF(pend_mod, order=order) #Create a BDF solver
            t, y = exp_sim.simulate(10)
            exp_sim.plot(mask=[1, 1, 0, 0])
            mpl.show()

    def test_excited_pendulum_order_4_k_100(self):
        k = 100
        phi = 2 * scipy.pi - 0.3
        x = scipy.cos(phi)
        y = scipy.sin(phi)
        order = 4
        pend = self.get_pendulum_rhs(k)
        initial_values = scipy.array([[x, y, 0, 0],
                                      [x+1.0, y, 0, 0],
                                      [x+2.0, y, 0, 0],
                                      [x+5.0, y, 0, 0],
                                      [x+10.0, y, 0, 0],
                                      [x+100.0, y, 0, 0]])
        for y0 in initial_values:
            pend_mod = Explicit_Problem(pend, y0=y0)
            pend_mod.name = \
              'Nonlinear Pendulum, k = {k}, init = {init}'.format(k=k, init=y0)
            exp_sim = BDF(pend_mod, order=order) #Create a BDF solver
            t, y = exp_sim.simulate(10)
            exp_sim.plot(mask=[1, 1, 0, 0])
            mpl.show()



if __name__ == '__main__':
    ##---- TASK 1 ----##
    def pend(t, y, k=100):
            """
            This is the right hand side function of the differential equation
            describing the elastic pendulum
            y: 1x4 array
            k: float
            """
            yprime = scipy.array([y[2],
                                  y[3],
                                  -y[0] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                                  (scipy.sqrt(y[0] ** 2 + y[1] ** 2)),
                                  -y[1] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                                  (scipy.sqrt(y[0] ** 2 + y[1] ** 2)) - 1])
            return yprime

    phi = 2 * scipy.pi - 0.3
    x = scipy.cos(phi)
    y = scipy.sin(phi)
    y0 = scipy.array([x, y, 0, 0])
    t0 = 0

    mod = Explicit_Problem(pend, y0, t0)

    sim = CVode(mod)
    t, y = sim.simulate(10)
    sim.plot(mask=[1, 1, 0, 0])
    mpl.show()


    unittest.main()
    # appropriate initial values: 0.9, 0.1, 0, 0
#    pend_mod=Explicit_Problem(pend, y0=np.array([0.9, 0.1, 0, 0]))
    #pend_mod=Explicit_Problem(pend, y0=np.array([2.*np.pi,1.]))
#    pend_mod.name='Nonlinear Pendulum'

    #Define an explicit solver
#    exp_sim = BDF(pend_mod, order=3) #Create a BDF solver
#    t, y = exp_sim.simulate(10)
#    exp_sim.plot(mask=[1, 1, 0, 0])
#    mpl.show()
