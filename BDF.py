from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
import scipy
import unittest
from assimulo.solvers import CVode

import step


class BDF(Explicit_ODE):
    """
    BDF methods
    """
    tol = 1.e-8
    maxit = 1000
    maxsteps = 5000

    def __init__(self, problem, order, corrector='Newton'):
        Explicit_ODE.__init__(self, problem) #Calls the base class

        # Solver options
        self.options["h"] = 0.01

        # Statistics
        self.statistics["nsteps"] = 0
        self.statistics["nfcns"] = 0

        self.order = order

        if order == 1:
            self.step_BDF = step.EE
        elif order == 2 and corrector == 'FPI':
            self.step_BDF = step.BDF2_FPI
        elif order == 2:
            self.step_BDF = step.BDF2
        elif order == 3:
            self.step_BDF = step.BDF3
        elif order == 4:
            self.step_BDF = step.BDF4
        else:
            raise ValueError('Only BDF methods of order 2-4 are implemented')

    def _set_h(self, h):
            self.options["h"] = float(h)

    def _get_h(self):
        return self.options["h"]

    h = property(_get_h, _set_h)

    def integrate(self, t, y, tf, opts):
        """
        _integrates (t, y) values until t > tf
        """
        h = self.options["h"]
        h = min(h, abs(tf-t))

        # Lists for storing the result
        tres = []
        yres = []

        for i in range(self.maxsteps):
            if t >= tf:
                break
            self.statistics["nsteps"] += 1

            if i < self.order:  # initial steps
                t_np1, y_np1 = step.EE(self, t, y, h, floatflag=True)
                y = y_np1
            else:
                t_np1, y_np1 = self.step_BDF(self, tres[-self.order:],
                                             yres[-self.order:],
                                             h)
            tres.append(t_np1)
            yres.append(y_np1.copy())
            t = t_np1
            h = min(self.h, np.abs(tf - t))
        else:
            raise Explicit_ODE_Exception(
                    'Final time not reached within maximum number of steps')

        return ID_PY_OK, tres, yres


    def get_statistics(self):
        """ Gather statistics and return them as a string. """

        # Gather necessary information.
        name = self.problem.name
        stepsize = self.options["h"]
        num_steps = str(self.statistics['nsteps'])
        num_func_eval = str(self.statistics['nfcns'])
        order = self.order

        def heading(string):
            """ Return if the string is a heading. """
            return not string.startswith(' ')

        def get_padded_format(string):
            """ Return dynamically padded format string. """
            if heading(string): # No padding on headings.
                return string
            # Only count length to the first ':'.
            current_len = len(string.split(':')[0])
            len_diff = max_len - current_len
            # Only add padding in front of the first ':'.
            string = string.replace(':', "{}:".format(len_diff*' '), 1)
            return string

        message_primitives = [
                    ("Final Run Statistics: {}", name),
                    ("  Step-length: {}", stepsize),
                    ("  Number of Steps: {}", num_steps),
                    ("  Number of Function Evaluations: {}", num_steps),
                    ("", ""), # newline.
                    ("Solver options:", ""),
                    ("  Solver: BDF: {}", order),
                    ("  Solver type: Fixed step.", "" ),
                ]

        # Get maximum length of the formatting strings.
        max_len = max([len(format_string) for (format_string, _) in
            message_primitives])
        # Add two.
        max_len += 2

        # Iterate over all message primitives and append to buffer.
        string_buffer = []
        for format_string, data in message_primitives:
            final_format = get_padded_format(format_string)
            string_buffer.append(final_format.format(data))

        # Return buffer as a string.
        return '\n'.join(string_buffer)


    def print_statistics(self, verbose=NORMAL):
        """ Use get_statistics() method and send to logger. """
        self.log_message(self.get_statistics(), verbose)


def pend_rhs_function(k):
    """
    This is the right hand side function of the differential equation
    describing the elastic pendulum. High k => stiffer pendulum
    y: 1x4 array
    k: float
    """

    def pendulum_eq_rhs(t, y, k=k):
        yprime = scipy.array(
                     [y[2],
                      y[3],
                      -y[0] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                      (scipy.sqrt(y[0] ** 2 + y[1] ** 2)),
                      -y[1] * k * (scipy.sqrt(y[0] ** 2 + y[1] ** 2) - 1) /
                      (scipy.sqrt(y[0] ** 2 + y[1] ** 2)) - 1])
        return yprime
    return pendulum_eq_rhs


class DefaultData(object):
    """ Class that holds the default data for the tests. """

    def __init__(self):
        self.phi = 2 * scipy.pi - 0.3
        self.x = scipy.cos(self.phi)
        self.y = scipy.sin(self.phi)
        self.y0 = scipy.array([self.x, self.y, 0, 0])
        self.k = 100
        self.klist = [0] + [10 ** i for i in range(0, 4)]
        self.orderlist = [2, 3, 4]


def run_permutations():

    # Create default data.
    dd = DefaultData()
    # Set order of BDF.
    bdf_order = 4
    # Set maximum time for simulation.
    sim_tmax = 5

    def run_single_plot(bdf_order, sim_tmax):
        # Get method defining right hand side of pendulum equation.
        pend_func = pend_rhs_function(dd.k)
        # Generate the model based on the function fetched above.
        pend_mod = Explicit_Problem(pend_func, y0=dd.y0)
        pend_mod.name = 'Nonlinear Pendulum, k = {}'.format(dd.k)
        # Create BDF solver.
        exp_sim = BDF(pend_mod, order=bdf_order)
        # Run the simulation.
        t, y = exp_sim.simulate(sim_tmax)
        # Plot the result.
        exp_sim.plot(mask=[1, 1, 0, 0])
        # Show the plot.
        mpl.show()

    run_single_plot(bdf_order, sim_tmax)


class BDFtests(unittest.TestCase):
    def setUp(self):
        phi = 2 * scipy.pi - 0.3
        self.y0 = scipy.array([x, y, 0, 0])
        self.klist = [0] + [10 ** i for i in range(0, 4)]

#    def test_order_4_varying_k(self):
#        order = 4
#        for k in self.klist:
#            pend_func = pend_rhs_function(k)
#            pend_mod = Explicit_Problem(pend_func, y0=self.y0)
#            pend_mod.name = 'Nonlinear Pendulum, k = {}'.format(k)
#            exp_sim = BDF(pend_mod, order=order) #Create a BDF solver
#            t, y = exp_sim.simulate(5)
#            exp_sim.plot(mask=[1, 1, 0, 0])
#            mpl.show()
#
#    def test_varying_order_k_1000(self):
#        k = 1000
#        pend_func = pend_rhs_function(k)
#        for order in self.orderlist:
#            pend_mod = Explicit_Problem(pend_func, y0=self.y0)
#            pend_mod.name = 'Nonlinear Pendulum, k = {}'.format(k)
#            exp_sim = BDF(pend_mod, order=order) #Create a BDF solver
#            t, y = exp_sim.simulate(10)
#            exp_sim.plot(mask=[1, 1, 0, 0])
#            mpl.show()

    def test_excited_pendulum_order_4_k_100(self):
        k = 100
        phi = 2 * scipy.pi - 0.3
        x = scipy.cos(phi)
        y = scipy.sin(phi)
        order = 4
        pend_func = pend_rhs_function(k)
        initial_values = scipy.array([[x, y, 0, 0],
                                      [x+0.01, y, 0, 0],
                                      [x+0.1, y, 0, 0],
                                      [x+0.5, y, 0, 0]])
#                                      [x+0.99, y, 0, 0],
#                                      [x+1.0, y, 0, 0]])
        for y0 in initial_values:
            for k in self.klist:
                pend_func = pend_rhs_function(k)
                pend_mod = Explicit_Problem(pend_func, y0=y0)
                pend_mod.name = \
                  'Nonlinear Pendulum, k = {k}, init = {init}'.format(k=k, init=y0)
                exp_sim = BDF(pend_mod, order=order) #Create a BDF solver
                t, y = exp_sim.simulate(10)
                exp_sim.plot(mask=[1, 1, 0, 0])
                mpl.show()

#    def test_order_2_FPI(self):
#        k = 100
#        order = 2
#        corrector = 'FPI'
#        pend_func = pend_rhs_function(k)
#        pend_mod = Explicit_Problem(pend_func, y0=self.y0)
#        pend_mod.name = \
#         'Nonlinear Pendulum, FPI, k = {k}, init = {init}'.format(k=k, init=y0)
#        exp_sim = BDF(pend_mod, order=order, corrector=corrector)
#        t, y = exp_sim.simulate(10)
#        exp_sim.plot(mask=[1, 1, 0, 0])
#        mpl.show()
#
#    def test_EE_k_influence(self):
#        order = 1
#        for k in self.klist:
#            pend_func = pend_rhs_function(k)
#            pend_mod = Explicit_Problem(pend_func, y0=self.y0)
#            pend_mod.name = \
#             'Nonlinear Pendulum, EE, k = {k}, init = {init}'.format(k=k,
#                                                                     init=self.y0)
#            exp_sim = BDF(pend_mod, order=order)
#            t, y = exp_sim.simulate(10)
#            exp_sim.plot(mask=[1, 1, 0, 0])
#            mpl.show()
#
# ========================================
#  CVODE
# ========================================
#
#    def test_CVode_method_params_influence(self):
#        k = 100
#        phi = 2 * scipy.pi - 0.3
#        x = scipy.cos(phi)
#        y = scipy.sin(phi)
#        t0 = 0
#        y0 = scipy.array([x+0.2, y, 0, 0])
#        pend_func = pend_rhs_function(k)
#        maxordlist = list(range(0, 7, 2))
#        atollist = [scipy.array([2, 1, 2, 2]) * 10 ** (-i)
#                    for i in range(0, 7, 2)]
#        rtollist = [10 ** (-i) for i in range(0, 7, 2)]
#        for maxord in maxordlist:
#            for atol in atollist:
#                for rtol in rtollist:
#                    mod = Explicit_Problem(pend_func, y0, t0)
#                    mod.name = \
#                     'Nonlinear Pendulum, CVode, k={k}, stretched, \n\
#                      maxord={maxord}, atol={atol}, rtol={rtol}'.format(k=k,
#                                        maxord=maxord, atol=atol, rtol=rtol)
#                    sim = CVode(mod)
#                    sim.maxord = maxord
#                    sim.atol = atol
#                    sim.rtol = rtol
#                    t, y = sim.simulate(10)
#                    sim.plot(mask=[1, 1, 0, 0])
#                    mpl.show()
#
#    def test_CVode_k_influence(self):
#        phi = 2 * scipy.pi - 0.3
#        x = scipy.cos(phi)
#        y = scipy.sin(phi)
#        t0 = 0
#        y0 = scipy.array([x+0.2, y, 0, 0])
#        for k in self.klist:
#            pend_func = pend_rhs_function(k)
#            mod = Explicit_Problem(pend_func, y0, t0)
#            mod.name = \
#             'Nonlinear Pendulum, CVode, k={k}, stretched'.format(k=k)
#            sim = CVode(mod)
#            t, y = sim.simulate(10)
#            sim.plot(mask=[1, 1, 0, 0])
#            mpl.show()

def task_1(k, x_start_offset):
    """ Method that performs and plots the first task from project 1. """

    phi = 2 * scipy.pi - 0.3
    x = scipy.cos(phi)
    y = scipy.sin(phi)
    y0 = scipy.array([x + x_start_offset, y, 0, 0])
    t0 = 0

    mod = Explicit_Problem(pend_rhs_function(k), y0, t0, k)

    sim = CVode(mod)
    t, y = sim.simulate(10)
    sim.plot(mask=[1, 1, 0, 0])
    mpl.show()


if __name__ == '__main__':
    ##---- TASK 1 ----##
    task_1_k = 100
    task_1_start_offset = 0.1
#    task_1(task_1_k, task_1_start_offset)
    ##----
#    unittest.main()
    run_permutations()
