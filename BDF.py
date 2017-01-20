import matplotlib

from assimulo.explicit_ode import Explicit_ODE
from assimulo.ode import *
from assimulo.solvers import CVode
from io import StringIO
from collections import OrderedDict

import matplotlib.pyplot as plt
import math
import numpy as np
import os
import re
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
        self.k_list = [0.1, 1, 5, 10, 100, 1000]
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
    if not width_mod: # Avoid printing None.
        width_mod = ''
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

    def run_permutations(case_dict, statistics, show_plot):
        """ Possible variations:
                - k-values.
                - initial values.
                - bdf order.
        """

        # Get title format.
        inner_title_format = case_dict.get('title')

        # Get all varying variables.
        k_list = case_dict.get('k_list')
        initial_values_list = case_dict.get('init_value_list')
        order_list = case_dict.get('order_list')
        type = case_dict.get('type')

        # Number of permutations.
        permutations = len(k_list)*len(initial_values_list)*len(order_list)

        # Initialize subplot variables.
        sp_dim_row, sp_dim_col = get_subdim(permutations)
        current_subplot = 1 # Starts at one.

        # Get width_mod if any.
        width_mod = case_dict.get('width_mod', '')

        # Get caption, if any.
        caption = case_dict.get('caption', '')

        # Make basis for one big plot.
        global global_figure_counter

        fig = plt.figure(global_figure_counter, figsize=(A4_inch_width,
                         row_inch_height*sp_dim_row))
        global_figure_counter += 1

        # Create list for appending runtime statistics.
        stat_list = statistics[case_dict.get('name')] = []

        # Iterate over all combinations of the changing variables listed above.
        for k_value in k_list:
            for initial_values in initial_values_list:
                for bdf_order in order_list:
                    # Gather missing data.
                    name = case_dict.get('name')
                    sim_tmax = case_dict.get('sim_tmax')


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
                                          case_dict)
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
        latex_output = figure_filename.replace('.pdf','.tex')
        with open(os.path.join(plot_folder, latex_output), 'w') as output:
            output.write(latex_import)


    def run_single_case(case_dict, figure, stat_list):
        """ Run a single test case. """
        # Get method defining right hand side of pendulum equation.
        pend_func = pend_rhs_function(case_dict.k)
        # Generate the model based on the function fetched above.
        pend_mod = Explicit_Problem(pend_func, y0=case_dict.init)
        pend_mod.name = case_dict.name
        # Create BDF solver.
        exp_sim = ""
        if case_dict.type == "BDF":
            exp_sim = BDF(pend_mod, order=case_dict.order)
        elif case_dict.type == "CVODE":
            case_dict_dict = case_dict.case_dict
            exp_sim = CVode(pend_mod)
            atol = case_dict_dict.get('cvode_atol')
            rtol = case_dict_dict.get('cvode_rtol')
            discretization = case_dict_dict.get('cvode_discretization')
            maxorder = case_dict_dict.get('cvode_maxorder')
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
        t, y = exp_sim.simulate(case_dict.sim_tmax)
        # Append stats to stat_list.
        simulation_stats = exp_sim.get_statistics()
        stat_list.append((simulation_stats, case_dict))
        # Pick out the first and second value of y.
        first_line, second_line = filter_out_y(y)
        # Grab the subplot data.
        sp_dim_row = case_dict.sp_dim_row
        sp_dim_col = case_dict.sp_dim_col
        sp_current = case_dict.current_subplot
        # Create new subplot.
        subplot = figure.add_subplot(sp_dim_row, sp_dim_col, sp_current,
                                     adjustable='box')
        # Plot the different linet and add legend.
        plt.plot(t, first_line, label='x')
        plt.plot(t, second_line, label='y')
        # Put legend in upper left corner.
        plt.legend(loc='upper right', frameon=False)
        # Add title.
        subplot.title.set_text(case_dict.title)
        # Should we show the plot?
        if case_dict.show_plot:
            plt.show()
        # Restore stdout.
        sys.stdout = old_sys_out


    class SingleTestCase():
        """ Class representing a single BDF test case. """
        def __init__(self, name, title, order, type, sim_tmax, k, init,
                sp_dim_row, sp_dim_col, current_subplot, case_dict):
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
            self.case_dict = case_dict

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
            'k_list': opt_dict.get('k_list',  default.k_list),
            'init_value_list': [[default.x+delta_x, default.y, 0, 0]],
            'title_values': "",
            'cvode_atol': opt_dict.get('atol'),
            'cvode_rtol': opt_dict.get('rtol'),
            'cvode_maxorder': opt_dict.get('maxorder'),
            'cvode_discretization': opt_dict.get('discretization'),
            'stat_caption': opt_dict.get('stat_caption'),
            'width_mod': opt_dict.get('width_mod'),
        }


        # Task3: Generate test cases.
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
#                'width_mod': 0.8,
                }
        task3_simulations.append(generate_plots_different_k(task3_opts))

    # Task4: Generate test cases for task 4 with default values of atol, rtol and
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
    for index, atol in enumerate(linfill(1e-2, 1, 3)):
        fmt = "Simulation for varying k using CVODE, "+\
              " $x_{{init}}$={:.2f}, atol={:.2f}."
        stat_caption = "Iterations and steps for varying atol values."
        task4_opts = {
                    'name': "CVODE_atol_{}".format(index),
                    'caption': fmt.format(start_x, atol),
                    'title': "k: {{k}}, atol: {:.2f}.".format(atol),
                    'type': "CVODE",
                    'stat_caption': stat_caption,
                    'atol': atol,
                    'k_list': [1, 10, 100],
                    'width_mod': 0.8,
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Task4: Add output for different CVODE rtol values.
    for index, rtol in enumerate(linfill(0.1, 1, 3)):
        fmt = "Simulation for varying k using CVODE, "+\
              " $x_{{init}}$={:.2f}, rtol={:.2f}."
        stat_caption = "Iterations and steps for varying rtol values."
        task4_opts = {
                    'name': "CVODE_rtol_{}".format(index),
                    'caption': fmt.format(start_x, rtol),
                    'title': "k: {{k}}, rtol: {:.2f}.".format(rtol),
                    'type': "CVODE",
                    'stat_caption': stat_caption,
                    'rtol': rtol,
                    'k_list': [1, 10, 100],
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Task4: Add output for different CVODE maxord values.
    for index, maxorder in enumerate([1,2,5]):
        maxorder = int(maxorder)
        stat_caption = "Iterations and steps for varying maxorder values."
        fmt = "Simulation for varying k using CVODE, "+\
              " $x_{{init}}$={:.2f}, maxorder={}."
        task4_opts = {
                    'name': "CVODE_maxorder_{}".format(index),
                    'caption': fmt.format(start_x, maxorder),
                    'title': "k: {{k}}, maxorder: {}.".format(maxorder),
                    'type': "CVODE",
                    'stat_caption': stat_caption,
                    'maxorder': maxorder,
                    #'k_list': [1, 10, 100],
                }
        task4_simulations.append(generate_plots_different_k(task4_opts))

    # Task4: Run default values for BDF and Adams.
    only_discretization = []
    for index, discretization in enumerate(['BDF','Adams']):
        fmt = "Simulation for varying k using CVODE using {}, "+\
              " $x_{{init}}$={:.2f}."
        stat_caption = "Iterations and steps for varying discretizations."
        task4_opts = {
                    'name': "CVODE_discretization_{}".format(index),
                    'caption': fmt.format(discretization, start_x),
                    'title': "{} - k: {{k}}.".format(discretization),
                    'type': "CVODE",
                    'stat_caption': stat_caption,
                    'discretization': discretization,
                }
        plots = generate_plots_different_k(task4_opts)
        task4_simulations.append(plots)
        only_discretization.append(plots)

    case_dicts = [
            *task3_simulations,
            *task4_simulations,
#             *only_discretization,
            ]

    statistics = OrderedDict()
    for case_dict in case_dicts:
        run_permutations(case_dict, statistics, False)

    classes = {}
    for name, data in statistics.items():
        if re.match(r".*_\d+$", name):
            category_name = re.sub(r"_\d+$", "", name)
            category_list = classes.get(category_name)
            if not category_list:
                category_list = classes[category_name] = []
            category_list.append(data)

    for class_name, data_list in classes.items():

        # Create class stat figure.
        stat_figure_filename = "{}_stats.pdf".format(class_name)
        stat_path = os.path.join(plot_folder, stat_figure_filename)

        fig, ax = plt.subplots(2, sharex=True)

        # Iterate over all data from this test-class.
        for tuple_list in data_list:
            k_values = []
            statistics = { # Pick data that is interesting.
                    "nsteps": [],
                    "nniters": [],
                    }

            def grab_cvode_variable(case_dict):
                """ Iterate though all the case_dict variable and extract the
                value that is currently being used and use that as a label. """
                for key, data in case_dict.case_dict.items():
                    if not key.startswith("cvode_"):
                        continue
                    if not data:
                        continue
                    if type(data) == float:
                        data = "{:.2f}".format(data)
                    return "{} = {}".format(key.replace("cvode_", ""), data)

            for data, case_dict in tuple_list:
                 k_values.append(case_dict.k)
                 variable = grab_cvode_variable(case_dict)
                 for key in statistics:
                     statistics[key].append(data[key])

            k_values = sorted(k_values)
            log_k_values = [math.log(val, 10) for val in k_values]

            # Round to 1 decimal place.
            tick_lablels = ["{:.1f}".format(val) for val in log_k_values]

            # Plot line for current test case.
            for index, (key, data) in enumerate(statistics.items()):
                log_data = [(val if val <= 0 else math.log(val, 10)) for val in data]
                label = grab_cvode_variable(case_dict)
                line = ax[index].plot(log_k_values, log_data, label=label)
                ax[index].set_xticks(log_k_values)
                ax[index].set_xticklabels(tick_lablels)
                ax[index].legend()

            ax[1].set_xlabel("log_10(k)")
            ax[0].set_ylabel("Steps (log_10)")
            ax[1].set_ylabel("Iterations (log_10)")

            ax[0].set_title("Runtime statistics.")

        first_case_dict = data_list[0][0][1].case_dict
        statistics_caption = first_case_dict['stat_caption']
        latex_import = latex_get_figure_frame(0.7, stat_path.replace("TeX/", ''),
                                              statistics_caption)

        plt.savefig(stat_path)
        latex_output = stat_figure_filename.replace('.pdf','.tex')
        with open(os.path.join(plot_folder, latex_output), 'w') as output:
            output.write(latex_import)
        plt.close(fig) # Close plot to avoid re-usage.


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
    latex_output = figure_filename.replace('.pdf','.tex')
    plt.close(fig)
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
