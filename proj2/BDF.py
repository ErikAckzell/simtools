from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
import numpy as np
import matplotlib.pyplot as mpl
import scipy.linalg as SL
import scipy
import unittest
from assimulo.solvers import CVode
import squeezer_HsnppkU
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
#                t_np1, y_np1 = step.EE(self, t, y, h, floatflag=True)
#                y = y_np1
                t_np1 = t + h
                y = y

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
        self.init_list = [scipy.array([self.x, self.y, 0, 0])]
        self.k = 100
        self.k_list = [0] + [10 ** i for i in range(0, 4)]
        self.order_list = [2, 3, 4]
        self.name = "Nonlinear Pendulum"


def run_simulations(plot=True):
    """ Method that runs all the defined simulations. """

    # Create default data.
    default = DefaultData()

    def INFO(string):
        """ Function that prints information to console."""
        print("[i]: {}".format(string))


    def SEPARATE_OUTPUT(string=""):
        """ Separate output sections from each other. """
        print("{}\n\n".format(string))


    def run_permutations(test_case_group, plot):
        """ Possible variations:
                - k-values.
                - initial values.
                - bdf order.
        """

        # Iterate over all combinations of the changing variables listed above.
        for k_value in test_case_group.k_list:
            for initial_values in test_case_group.init_values_list:
                for bdf_order in test_case_group.order_list:
                    # Gather missing data.
                    name = test_case_group.name
                    sim_tmax = test_case_group.sim_tmax

                    # Construct information string.
                    fmt_string = ("Running simulation for k = {} "+
                                 "\n     initial_values: {}"+
                                 "\n     bdf_order: {}\n")

                    INFO(fmt_string.format(k_value,
                                           initial_values,
                                           bdf_order))

                    # Create new single test case.
                    case = SingleTestCase(name,
                                          bdf_order,
                                          sim_tmax,
                                          k_value,
                                          initial_values)
                    # Toggle plotting.
                    case.plot = plot
                    # Run the single test case.
                    run_single_case(case)
                    SEPARATE_OUTPUT()

        SEPARATE_OUTPUT("\n\n[!] #### Group done, continuing with next group of tests.")


    def run_single_case(test_case):
        """ Run a single test case. """
        # Get method defining right hand side of pendulum equation.
        pend_func = pend_rhs_function(test_case.k)
        # Generate the model based on the function fetched above.
        pend_mod = Explicit_Problem(pend_func, y0=test_case.init)
        pend_mod.name = test_case.name
        # Create BDF solver.
        exp_sim = BDF(pend_mod, order=test_case.order)
        # Run the simulation.
        t, y = exp_sim.simulate(test_case.sim_tmax)
        # Should we plot?
        if test_case.plot:
            # Plot the result.
            exp_sim.plot(mask=[1, 1, 0, 0])
            # Show the plot.
            mpl.show()


    class GroupTestCase():
        """ Class representing a group of BDF test cases. """
        def __init__(self, name, order_list, sim_tmax, k_list, init_values_list):
            self.name = name
            self.order_list = order_list
            self.sim_tmax= sim_tmax
            self.k_list = k_list
            self.init_values_list = init_values_list
            self.plot = True


    class SingleTestCase():
        """ Class representing a single BDF test case. """
        def __init__(self, name, order, sim_tmax, k, init):
            self.name = name
            self.order = order
            self.sim_tmax = sim_tmax
            self.k = k
            self.init = init
            self.plot = True


    def dict_2_group_case(dictionary):
        """ Turn data in dictionary to Test Case object. """
        return GroupTestCase(dictionary['name'],
                             dictionary['order_list'],
                             dictionary['sim_tmax'],
                             dictionary['k_list'],
                             dictionary['init_value_list'])

    # Test order 4 BDF with varying k's.
    ord_4_var_k = {
        'name': default.name,
        'order_list': [4],
        'sim_tmax': 5,
        'k_list': default.k_list,
        'init_value_list': default.init_list,
        }

    # Test varying order with k = 1000
    var_ord_k_1000 = {
        'name': default.name,
        'order_list': default.order_list,
        'sim_tmax': 10,
        'k_list': [1000],
        'init_value_list': default.init_list,
        }

    # Test excited pendulum for different initial values with k = 100.
    excited_pend_var_init = {
        'name': default.name,
        'order_list': [4],
        'sim_tmax': 10,
        'k_list': [100],
        'init_value_list': [
            [default.x, default.y, 0, 0],
            [default.x+0.01, default.y, 0, 0],
            [default.x+0.1, default.y, 0, 0],
            [default.x+0.5, default.y, 0, 0],
            ]
    }

    test_cases = [
            ord_4_var_k,
            var_ord_k_1000,
            excited_pend_var_init,
            ]

    for case_dict in test_cases:
        run_permutations(dict_2_group_case(case_dict), plot)


#class BDFtests(unittest.TestCase):

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

    y0 = squeezer_HsnppkU.init_squeezer()

    t0 = 0

    mod = Explicit_Problem(squeezer_HsnppkU.squeezer, y0, y0)

    sim = BDF(mod, 2)

    t, y = sim.simulate(10)

    sim.plot()

    ##---- TASK 1 ----##
#    task_1_k = 100
#    task_1_start_offset = 0.1
##    task_1(task_1_k, task_1_start_offset)
#    ##----
#    run_simulations(plot=True)
