from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
from assimulo.solvers import CVode
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.linalg as SL
import scipy
import step
import sys
import unittest



# Variables for saving plots.
plot_template_name = "{}.pdf"
plot_folder = os.path.join("TeX", "includes", "figures")

global_figure_counter = 1

# Global dimensions.
A4_inch_width = 8.3
row_inch_height = 2.0

# Class for default values.
class DefaultData(object):
    """ Class that holds the default data for the tests. """

    def __init__(self):
        self.phi = 2 * scipy.pi - 0.3
        self.x = scipy.cos(self.phi)
        self.y = scipy.sin(self.phi)
        self.init_list = [scipy.array([self.x, self.y, 0, 0])]
        self.k = 100
        self.k_list = [0, 1, 5, 10, 100, 1000]
        self.order_list = [2, 3, 4]
        self.name = "Nonlinear Pendulum: "


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


#    def get_statistics(self):
#        """ Gather statistics and return them as a string. """
#
#        # Gather necessary information.
#        name = self.problem.name
#        stepsize = self.options["h"]
#        num_steps = str(self.statistics['nsteps'])
#        num_func_eval = str(self.statistics['nfcns'])
#        order = self.order
#
#        def heading(string):
#            """ Return if the string is a heading. """
#            return not string.startswith(' ')
#
#        def get_padded_format(string):
#            """ Return dynamically padded format string. """
#            if heading(string): # No padding on headings.
#                return string
#            # Only count length to the first ':'.
#            current_len = len(string.split(':')[0])
#            len_diff = max_len - current_len
#            # Only add padding in front of the first ':'.
#            string = string.replace(':', "{}:".format(len_diff*' '), 1)
#            return string
#
#        message_primitives = [
#                    ("Final Run Statistics: {}", name),
#                    ("  Step-length: {}", stepsize),
#                    ("  Number of Steps: {}", num_steps),
#                    ("  Number of Function Evaluations: {}", num_steps),
#                    ("", ""), # newline.
#                    ("Solver options:", ""),
#                    ("  Solver: BDF: {}", order),
#                    ("  Solver type: Fixed step.", "" ),
#                ]
#
#        # Get maximum length of the formatting strings.
#        max_len = max([len(format_string) for (format_string, _) in
#            message_primitives])
#        # Add two.
#        max_len += 2
#
#        # Iterate over all message primitives and append to buffer.
#        string_buffer = []
#        for format_string, data in message_primitives:
#            final_format = get_padded_format(format_string)
#            string_buffer.append(final_format.format(data))
#
#        # Return buffer as a string.
#        return '\n'.join(string_buffer)
#
#
#    def print_statistics(self, verbose=NORMAL):
#        """ Use get_statistics() method and send to logger. """
#        self.log_message(self.get_statistics(), verbose)


#!! Include: task1
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
#!! Include end


def latex_get_figure_frame(width_mod, name, caption=""):
    """ Generate latex code for importing figure. """
    frame = "\\begin{{figure}}[H]\n"+\
            "   \\center"+\
            "   \\includegraphics[width={}\\textwidth]{{{}}}\n"+\
            "   {}\n"+\
            "\\end{{figure}}\n"
    if (caption):
        caption = "   \\caption{{{}}}".format(caption)
    return frame.format(width_mod, name, caption)

def filter_out_y(y_vector_list):
    """ Keep the first and second value of the vectors in the list. """
    first_line = []
    second_line = []
    for y1, y2, _, _ in list(y_vector_list):
        first_line.append(y1)
        second_line.append(y2)
    return first_line, second_line



def run_simulations(show_plot=True):
    """ Method that runs all the defined simulations. """

    # Create default data.
    default = DefaultData()

    def INFO(string):
        """ Function that prints information to console."""
        print("[i]: {}".format(string))


    def SEPARATE_OUTPUT(string=""):
        """ Separate output sections from each other. """
        print("{}\n\n".format(string))

    def get_subdim(permutations):
        """ Return grid coordinates for sub-figures based on permutations. """
        collumns = 3
        rows = int(permutations / collumns)
        if (permutations % collumns):
            rows += 1
        return rows, collumns

    def run_permutations(test_case, statistics, show_plot):
        """ Possible variations:
                - k-values.
                - initial values.
                - bdf order.
        """

        # Get title format.
        inner_title_format = test_case.get('title')

        # Get all varying variables.
        k_list = test_case.get('k_list')
        initial_values_list = test_case.get('init_value_list')
        order_list = test_case.get('order_list')
        type = test_case.get('type')

        # Number of permutations.
        permutations = len(k_list)*len(initial_values_list)*len(order_list)

        # Initialize subplot variables.
        sp_dim_row, sp_dim_col = get_subdim(permutations)
        current_subplot = 1 # Starts at one.

        # Get width_mod if any.
        width_mod = test_case.get('width_mod', '')

        # Get caption, if any.
        caption = test_case.get('caption', '')

        # Make basis for one big plot.
        global global_figure_counter

        fig = plt.figure(global_figure_counter, figsize=(A4_inch_width,
                         row_inch_height*sp_dim_row))
        global_figure_counter += 1

        # Create list for appending runtime statistics.
        stat_list = statistics[test_case.get('name')] = []

        # Iterate over all combinations of the changing variables listed above.
        for k_value in k_list:
            for initial_values in initial_values_list:
                for bdf_order in order_list:
                    # Gather missing data.
                    name = test_case.get('name')
                    sim_tmax = test_case.get('sim_tmax')


                    # Construct information string.
                    fmt_string = ("Running simulation for k = {} "+
                                 "\n     initial_values: {}"+
                                 "\n     bdf_order: {}\n")

                    INFO(fmt_string.format(k_value,
                                           initial_values,
                                           bdf_order))

                    # Assemble all values for title formatting.
                    initial_fmt = "x: {:.2f}, y: {:.2f}"
                    formated_initial = initial_fmt.format(initial_values[0],
                                                          initial_values[1])
                    possible_title_values = {
                            'k': k_value,
                            'initial_values': formated_initial,
                            'order': bdf_order,
                            }

                    # Create title for plot by expanding dict to keywords.
                    title = inner_title_format.format(**possible_title_values)

                    # Create new single test case.
                    case = SingleTestCase(name,
                                          title,
                                          bdf_order,
                                          type,
                                          sim_tmax,
                                          k_value,
                                          initial_values,
                                          sp_dim_row,
                                          sp_dim_col,
                                          current_subplot,
                                          test_case)
                    # Toggle plotting.
                    case.show_plot = show_plot
                    # Run the single test case.
                    run_single_case(case, fig, stat_list)
                    SEPARATE_OUTPUT()
                    current_subplot += 1

        SEPARATE_OUTPUT("\n\n[!] #### Group done, continuing with next group of tests.")
        figure_filename = plot_template_name.format(name)
        plot_path = os.path.join(plot_folder, figure_filename)
        # Use tight layout, otherwise the titles collide with the axis.
        plt.tight_layout()
        plt.savefig(plot_path)
        # Generate latex code to import the figure.
        latex_import = latex_get_figure_frame(width_mod, plot_path.replace("TeX/", ''),
                                             caption)
        latex_output = figure_filename.replace('.pdf','.txt')
        with open(os.path.join(plot_folder, latex_output), 'w') as output:
            output.write(latex_import)

    def run_single_case(test_case, figure, stat_list):
        """ Run a single test case. """
        # Get method defining right hand side of pendulum equation.
        pend_func = pend_rhs_function(test_case.k)
        # Generate the model based on the function fetched above.
        pend_mod = Explicit_Problem(pend_func, y0=test_case.init)
        pend_mod.name = test_case.name
        # Create BDF solver.
        exp_sim = ""
        if test_case.type == "BDF":
            exp_sim = BDF(pend_mod, order=test_case.order)
        elif test_case.type == "CVODE":
            test_case_dict = test_case.test_case
            exp_sim = CVode(pend_mod)
            atol = test_case_dict.get('cvode_atol')
            rtol = test_case_dict.get('cvode_rtol')
            discretization = test_case_dict.get('cvode_discretization')
            maxorder = test_case_dict.get('cvode_maxorder')
            if atol: # Set CVode atol value.
                exp_sim.atol = atol
            if rtol: # Set CVode rtol value.
                exp_sim.rtol = rtol
            if maxorder: # Set CVode maxorder.
                exp_sim.maxord = maxorder
            if discretization: # Set CVode discretization.
                exp_sim.discr = discretization
        # Redirect stdout.
        old_sys_out = sys.stdout
        sys.stdout = StringIO()
        # Run the simulation.
        t, y = exp_sim.simulate(test_case.sim_tmax)
        # Append stats to stat_list.
        simulation_stats = exp_sim.get_statistics()
        stat_list.append(simulation_stats)
        print(simulation_stats.keys())
        # Pick out the first and second value of y.
        first_line, second_line = filter_out_y(y)
        # Grab the subplot data.
        sp_dim_row = test_case.sp_dim_row
        sp_dim_col = test_case.sp_dim_col
        sp_current = test_case.current_subplot
        # Create new subplot.
        subplot = figure.add_subplot(sp_dim_row, sp_dim_col, sp_current,
                                     adjustable='box')
        # Plot the different linet and add legend.
        plt.plot(t, first_line, label='x')
        plt.plot(t, second_line, label='y')
        # Put legend in upper left corner.
        plt.legend(loc='upper right', frameon=False)
        # Add title.
        subplot.title.set_text(test_case.title)
        # Should we show the plot?
        if test_case.show_plot:
            plt.show()
        # Restore stdout.
        sys.stdout = old_sys_out


    class SingleTestCase():
        """ Class representing a single BDF test case. """
        def __init__(self, name, title, order, type, sim_tmax, k, init,
                sp_dim_row, sp_dim_col, current_subplot, test_case):
            self.name = name
            self.title = title
            self.order = order
            self.type = type
            self.sim_tmax = sim_tmax
            self.k = k
            self.init = init
            self.sp_dim_row = sp_dim_row
            self.sp_dim_col = sp_dim_col
            self.current_subplot = current_subplot
            self.plot = True
            self.test_case = test_case

    # Task 3, testing slightly stretched spring with BDF-2,3,4 method for
    # various values of k.

    delta_x = 0.2

    def get_name(order):
        """ Substitute BDF-1 with EE, since it uses explicit Euler."""
        if (order == 1):
            return "EE"
        else:
            return "BDF-{}".format(order)

    def generate_plots_different_k(opt_dict):
        """ Return dictionary for generating figures for task 3/4 using BDF or
        CVode. """
        return {
            'name': opt_dict.get('name'),
            'title': opt_dict.get('title'),
            'order_list': opt_dict.get('order', [0]),
            'type': opt_dict.get('type', "BDF"),
            'sim_tmax': 8,
            'caption': opt_dict.get('caption'),
            'k_list': default.k_list,
            'init_value_list': [[default.x+delta_x, default.y, 0, 0]],
            'title_values': "",
            'cvode_atol': opt_dict.get('atol'),
            'cvode_rtol': opt_dict.get('rtol'),
            'cvode_maxorder': opt_dict.get('maxorder'),
            'cvode_discretization': opt_dict.get('discretization'),
        }


    # Generate test cases for task 3.
    fmt_caption = "Simulation done by {} for different values of k, with "+\
                  "$x_{{init}}$={:.2f}."
    start_x = default.x+delta_x
    task3_simulations = []
    for order in range(1,5):
        name = get_name(order)
        task3_opts = {
                'name': "task3_ord{}".format(order),
                'order': [order],
                'caption': fmt_caption.format(name, start_x),
                'title': "{}, k: {{k}}.".format(name),
                'type': "BDF",
                }
        task3_simulations.append(generate_plots_different_k(task3_opts))

    # Generate test cases for task 4 with default values of atol, rtol and
    # maxorder.
    fmt = "Simulation for varying k with CVODE, with $x_{{init}}$={:.2f}."
    task4_opts = {
                'name': "CVODE_var_k",
                'caption': fmt.format(start_x),
                'title': "CVODE, k: {k}.",
                'type': "CVODE",
            }
    task4_simulations = []
    task4_simulations.append(generate_plots_different_k(task4_opts))

    def linfill(a, b, num):
        """ linfill num values between a and b. """

        def float_range(a, b, step):
            """ Range for floats, not including b. """
            while a <= b:
                yield a
                a += step

        diff = b-a
        step = diff/(num-1)
        return float_range(a, b, step)

    # Task4: Add output for different CVODE atol values.
    for index, atol in enumerate(linfill(1e-2, 1, 4)):
        fmt = "Simulation for varying k with CVODE, with"+\
              " $x_{{init}}$={:.2f}, atol={:.2f}."
        task4_opts = {
                    'name': "CVODE_atol_{}".format(index),
                    'caption': fmt.format(start_x, atol),
                    'title': "k: {{k}}, atol: {:.2f}.".format(atol),
                    'type': "CVODE",
                    'atol': atol,
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Task4: Add output for different CVODE rtol values.
    for index, rtol in enumerate(linfill(0, 1, 4)):
        fmt = "Simulation for varying k with CVODE, with"+\
              " $x_{{init}}$={:.2f}, rtol={:.2f}."
        task4_opts = {
                    'name': "CVODE_rtol_{}".format(index),
                    'caption': fmt.format(start_x, rtol),
                    'title': "k: {{k}}, rtol: {:.2f}.".format(rtol),
                    'type': "CVODE",
                    'rtol': rtol,
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Task4: Add output for different CVODE maxord values.
    for index, maxorder in enumerate([1,2,5]):
        maxorder = int(maxorder)
        fmt = "Simulation for varying k with CVODE, with"+\
              " $x_{{init}}$={:.2f}, maxorder={}."
        task4_opts = {
                    'name': "CVODE_maxorder_{}".format(index),
                    'caption': fmt.format(start_x, maxorder),
                    'title': "k: {{k}}, maxorder: {}.".format(maxorder),
                    'type': "CVODE",
                    'maxorder': maxorder,
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Task4: Run default values for BDF and Adams.
    for index, discretization in enumerate(['BDF','Adams']):
        fmt = "Simulation for varying k with CVODE using {}, with"+\
              " $x_{{init}}$={:.2f}."
        task4_opts = {
                    'name': "CVODE_discretization_{}".format(index),
                    'caption': fmt.format(discretization, start_x),
                    'title': "{} - k: {{k}}.".format(discretization),
                    'type': "CVODE",
                    'discretization': discretization,
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Test order 4 BDF with varying k's.
    ord_4_var_k = {
        'name': "ord_4_var_k",
        'title': "BDF: {order}, k={k}",
        'order_list': [4],
        'sim_tmax': 5,
        'caption': "Simulation for different values of k with BDF order 4.",
        'k_list': default.k_list,
        'init_value_list': default.init_list,
        'title_values': 'k_list'
        }

    # Test varying order with k = 1000
    var_ord_k_1000 = {
        'name': "var_ord_k_1000",
        'title': "k: {k}, BDF-order: {order}",
        'order_list': default.order_list,
        'sim_tmax': 10,
        'k_list': [1000],
        'caption': "Simulation for different BDF orders, k = 1000.",
        'init_value_list': default.init_list,
        }

    # Test excited pendulum for different initial values with k = 100.
    excited_pend_var_init = {
        'name': "excited_pend_var_init",
        'title': "{initial_values}",
        'order_list': [4],
        'sim_tmax': 10,
        'k_list': [100],
        'width_mod': 1.0,
        'caption': "Excited pendulum for different initial values with k = 100.",
        'init_value_list': [
            [default.x, default.y, 0, 0],
            [default.x+0.01, default.y, 0, 0],
            [default.x+0.1, default.y, 0, 0],
            [default.x+0.5, default.y, 0, 0],
            ]
    }

    test_cases = [
#            ord_4_var_k,
#            var_ord_k_1000,
#            excited_pend_var_init,
#            *task3_simulations,
            *task4_simulations,
            ]

    statistics = {}
    for case_dict in test_cases:
        run_permutations(case_dict, statistics, False)


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
#        plt.show()
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
#            plt.show()
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
#                    plt.show()
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
#            plt.show()

def task_1(k, x_start_offset):
    """ Method that performs and plots the first task from project 1. """

    phi = 2 * scipy.pi - 0.3
    x = scipy.cos(phi)
    y_start = scipy.sin(phi)
    x_start = x+x_start_offset
    y0 = scipy.array([x_start, y_start, 0, 0])
    t0 = 0

    mod = Explicit_Problem(pend_rhs_function(k), y0, t0, k)

    div = 2
    fig = plt.figure(-1, figsize=(A4_inch_width/div, A4_inch_width/(2*div)))
    sim = CVode(mod)
    t, y = sim.simulate(8)
    first_line, second_line = filter_out_y(y)
    plt.plot(t, first_line, label='x')
    plt.plot(t, second_line, label='y')
    plt.legend(loc='upper right', frameon=False)

    figure_name = "task_1"
    figure_filename = "{}.pdf".format(figure_name)

#    fig.suptitle("Task 1: Pendulum with RHS and CVODE.")
    plot_path = os.path.join(plot_folder, figure_filename)
    plt.savefig(plot_path)

    width_mod = 0.6
    caption = "Pendulum simulation with RHS and CVODE, "+\
              "$x_{{init}}$={:.2f}, $y_{{init}}$={:.2f}.".format(x_start,
                                                                 y_start)

    latex_import = latex_get_figure_frame(width_mod, plot_path.replace("TeX/", ''),
                                          caption)
    latex_output = figure_filename.replace('.pdf','.txt')
    with open(os.path.join(plot_folder, latex_output), 'w') as output:
        output.write(latex_import)

if __name__ == '__main__':
    # Make sure there is a dir where we can save plots.
    if not os.path.isdir(plot_folder):
        os.makedirs(plot_folder)
    ##---- TASK 1 ----##
    task_1_k = 100
    task_1_start_offset = 0.1
    task_1(task_1_k, task_1_start_offset)
    ##----
    run_simulations(show_plot=False)
