import numpy as np

from assimulo.problem import Implicit_Problem
from assimulo.solvers import IDA

def residual(t, y, yd):

    gravity = 9.82

    res_0 = yd[0]-y[2]
    res_1 = yd[1]-y[3]
    res_2 = yd[2]+y[4]*y[0]
    res_3 = yd[3]+y[4]*y[1]+gravity
    res_4 = y[2]**2+y[3]**2-y[4]*(y[0]**2+y[1]**2)-y[1]*gravity

    return np.array([res_0, res_1, res_2, res_3, res_4])

def main():

    # Initial conditions.
    t0 = 0.0 # Initial time.
    y0 = [1.0, 0.0, 0.0, 0.0, 0.0]
    yd0 = [0.0, 0.0, 0.0, -9.82, 0.0]

    # Set up model.
    model = Implicit_Problem(residual, y0, yd0, t0)
    model.name = 'Pendulum'

    # Create solver.
    sim = IDA(model)

    # Simulate.
    tfinal = 10.0
    ncp = 500 # Number of communication / return points.
    t, y, yd = sim.simulate(tfinal, ncp)

    # Plot.
    sim.plot()


if __name__ == "__main__":
    main()
