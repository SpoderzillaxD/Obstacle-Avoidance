#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 15:29:29 2025

@author: hamza61b
"""




"""
distance traveled by the vehicle
"""
global current_progress
current_progress = 0.0
import do_mpc
import numpy as np
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches
import math
import pdb
import sys
from map import Map, Obstacle
from reference_path import ReferencePath
import time
from scipy.interpolate import splprep, splev




class simple_bycicle_model:
    def __init__(self, reference_path, vehicle_length, vehicle_width, dt):

        self.length = vehicle_length
        self.width = vehicle_width
        self.safety_margin = self.width / np.sqrt(2) 
        self.reference_path = reference_path
        self.waypoint= 0
        self.current_waypoint = self.reference_path.waypoints[self.waypoint]
        self.dt = dt

    def model_setup(self):

        model_type = "continuous"
        self.model = do_mpc.model.Model(model_type)

        # States:
        x = self.model.set_variable("_x","x")
        y = self.model.set_variable("_x","y")
        psi = self.model.set_variable("_x","psi")
        v = self.model.set_variable("_x","v")
        e_y = self.model.set_variable("_x","e_y")

        # Input:
        a = self.model.set_variable("_u","a")
        heading = self.model.set_variable("_u","heading")

        # Reference
        x_ref = self.model.set_variable("_tvp","x_ref")
        y_ref = self.model.set_variable("_tvp","y_ref")
        psi_ref = self.model.set_variable("_tvp","psi_ref")
        v_ref = self.model.set_variable("_tvp","v_ref")
        ey_lb = self.model.set_variable("_tvp","ey_lb")
        ey_ub = self.model.set_variable("_tvp","ey_ub")
        
        #define error
        psi_error = atan2(sin(psi - psi_ref), cos(psi - psi_ref))
            
        self.model.set_expression('psi_error', psi_error)
        
        stage_cost = (
            100000 * (e_y - (ey_lb + ey_ub) / 2) ** 2
            + psi_error ** 2
            + 0.1 * (x - x_ref) ** 2
            + 0.1 * (y - y_ref) ** 2
            + 0.01 * heading**2
            + 0.01 * a**2
        )
        self.model.set_expression('cost', stage_cost)
        
        self.model.set_rhs("x", v * cos(psi))
        self.model.set_rhs("y", v * sin(psi))
        self.model.set_rhs("psi", v * heading / self.length)
        self.model.set_rhs("v", a)
        self.model.set_rhs("e_y", v * sin(psi_error))

        self.model.setup()

    def get_current_waypoint(self):
        
        distance = np.cumsum(reference_path.segment_lengths)
        next_wp = np.argmax(distance > current_progress)
        prev_wp = max(0, next_wp - 1)
        
        if abs(distance[next_wp] - current_progress) <= abs(distance[prev_wp] - current_progress):
            self.waypoint = next_wp
        else:
            self.waypoint = prev_wp
        
        self.current_waypoint = reference_path.waypoints[self.waypoint]


# Colors
CAR = '#F1C40F'
CAR_OUTLINE = '#B7950B'


class Simulator:

    def __init__(self, bicycle):
        self.bicycle = bicycle
        self.simulator = do_mpc.simulator.Simulator(bicycle.model)

        self.tvp_template = self.simulator.get_tvp_template()
        self.simulator.set_tvp_fun(self.tvp_fun)

        self.simulator.set_param(t_step=0.05)

        self.simulator.setup()

    def tvp_fun(self,t_now):
        
        current_waypoint = self.bicycle.reference_path.get_waypoint(self.bicycle.waypoint)
        self.tvp_template["x_ref"] = current_waypoint.x
        self.tvp_template["y_ref"] = current_waypoint.y
        self.tvp_template["psi_ref"] = current_waypoint.psi
        self.tvp_template["v_ref"] = current_waypoint.v_ref or 0
        
        
        return self.tvp_template

    #################################################################
    #                                                               #
    #   This plotting function is cited from:                       #
    #   the file "spatial_bicycle_models.py" of matssteinweg, ZTH   #
    #   to use the simulator created by the author                  #
    #                                                               #
    #################################################################
    def show(self, states):
        '''
        Display car on current axis.
        '''
        x, y, psi = states[0], states[1], states[2]

        # Get car's center of gravity
        cog = (x, y)
        # Get current angle with respect to x-axis
        yaw = np.rad2deg(psi)
        # Draw rectangle
        car = plt_patches.Rectangle(
            cog,
            width=self.bicycle.length,
            height=self.bicycle.width,
            angle=yaw,
            facecolor=CAR,
            edgecolor=CAR_OUTLINE,
            zorder=20,
        )

        # Shift center rectangle to match center of the car
        car.set_x(
            car.get_x()
            - (
                self.bicycle.length / 2 * np.cos(psi)
                - self.bicycle.width / 2 * np.sin(psi)
            )
        )
        car.set_y(
            car.get_y()
            - (
                self.bicycle.width / 2 * np.cos(psi)
                + self.bicycle.length / 2 * np.sin(psi)
            )
        )

        # Add rectangle to current axis
        ax = plt.gca()
        ax.set_facecolor('#404040')
        ax.add_patch(car)



class MPC:
    
    def __init__(self, bicycle):

        self.bicycle = bicycle
        self.model = bicycle.model
        
        
        self.horizon = 15
        self.dt = 0.05
        self.length = 0.12
        self.width = 0.06

        self.mpc = do_mpc.controller.MPC(self.model)
        setup_mpc = {
            'n_robust': 0,
            'n_horizon': self.horizon,
            't_step': self.dt,
            'state_discretization': 'collocation',
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)

        # define the objective function and constriants
        
        lterm = (100000 * (self.model.x['e_y'] - (self.model.tvp['ey_lb'] + self.model.tvp['ey_ub']) / 2) ** 2
            + (self.model.aux['psi_error']) ** 2 + 0.1*(self.model.x['x'] - self.model.tvp['x_ref']) ** 2
            + 0.1*(self.model.x['y'] - self.model.tvp['y_ref']) ** 2 + 0.01 * self.model.u['heading']**2   # penalize steering input
    + 0.01 * self.model.u['a']**2
        )
        
        mterm = (100000 * (self.model.x['e_y'] - (self.model.tvp['ey_lb'] + self.model.tvp['ey_ub']) / 2) ** 2
            + 0.1 * (self.model.x['v'] - self.model.tvp['v_ref']) ** 2) + 0.1*(self.model.x['x'] - self.model.tvp['x_ref']) ** 2 + 0.1*(self.model.x['y'] - self.model.tvp['y_ref']) ** 2

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.constraints()

        # provide time-varing parameters: setpoints/references
        self.tvp_template = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.tvp_fun)

        self.mpc.setup()

    def tvp_fun(self,t_now):
        
        ey_ub, ey_lb, _  = self.bicycle.reference_path.update_path_constraints(
            self.bicycle.waypoint + 1,
            self.horizon,
            2 * self.width / np.sqrt(2),
            self.width / np.sqrt(2),
        )
        for h in range(self.horizon):
            current_waypoint = self.bicycle.reference_path.get_waypoint(
                self.bicycle.waypoint + h
            )
            self.tvp_template['_tvp', h, 'x_ref'] = current_waypoint.x
            self.tvp_template['_tvp', h, 'y_ref'] = current_waypoint.y
            self.tvp_template['_tvp', h, 'psi_ref'] = current_waypoint.psi
            self.tvp_template['_tvp', h, 'ey_lb'] = ey_lb[h]
            self.tvp_template['_tvp', h, 'ey_ub'] = ey_ub[h]
            self.tvp_template['_tvp', h, 'v_ref'] = current_waypoint.v_ref or 0
        return self.tvp_template

    def constraints(self):

        # states constraints
        self.mpc.bounds['lower', '_x', 'x'] = -np.inf
        self.mpc.bounds['upper', '_x', 'x'] = np.inf
        self.mpc.bounds['lower', '_x', 'y'] = -np.inf
        self.mpc.bounds['upper', '_x', 'y'] = np.inf
        self.mpc.bounds['lower', '_x', 'psi'] = - 2 * np.pi
        self.mpc.bounds['upper', '_x', 'psi'] = 2 * np.pi
        self.mpc.bounds['lower', '_x', 'v'] = 0.0
        self.mpc.bounds['upper', '_x', 'v'] = 1
        self.mpc.bounds['lower', '_x', 'e_y'] = -2
        self.mpc.bounds['upper', '_x', 'e_y'] = 2

        # input constraints
        self.mpc.bounds['lower', '_u', 'a'] = -0.5
        self.mpc.bounds['upper', '_u', 'a'] = 0.5
        self.mpc.bounds['lower', '_u', 'heading'] = -1
        self.mpc.bounds['upper', '_u', 'heading'] = 1

    def control(self, x0):

        self.bicycle.get_current_waypoint()
        u_ = self.mpc.make_step(x0)
        u = np.array([u_[0], u_[1]])

        return u

    def distance_update(self, states):
        global current_progress
        v, psi = states[3], states[2]
        vel = v * np.cos(self.mpc.data['_aux', 'psi_error'][0])
        current_progress += vel * self.dt




def environment_setup():

    # Load map file
    map_file = 'maps/sim_map.png'
    map = Map(file_path=map_file, origin=[-1, -2], resolution=0.005)

    # Specify waypoints
    wp_x = [-0.75, -0.25, -0.25, 0.25, 0.25, 1.25, 1.25,
            0.75, 0.75, 1.25, 1.25, -0.75, -0.75, -0.25]
    wp_y = [-1.5, -1.5, -0.5, -0.5, -1.5, -
            1.5, -1, -1, -0.5, -0.5, 0, 0, -1.5, -1.5]

    reference_path = ReferencePath(
        map,
        wp_x,
        wp_y,
        resolution = 0.05,
        smoothing_distance=5,
        max_width=0.23,
        circular=True,
    )

    # Add obstacles
    obs1 = Obstacle(cx=0.0, cy=0.05, radius=0.08)
    obs2 = Obstacle(cx=-0.85, cy=-0.5, radius=0.08)
    obs3 = Obstacle(cx=-0.75, cy=-1.5, radius=0.08)
    obs4 = Obstacle(cx=-0.35, cy=-1.0, radius=0.08)
    obs5 = Obstacle(cx=0.35, cy=-1.0, radius=0.08)
    obs6 = Obstacle(cx=0.78, cy=-1.47, radius=0.08)
    obs7 = Obstacle(cx=0.73, cy=-0.9, radius=0.08)
    obs8 = Obstacle(cx=1.2, cy=0.0, radius=0.08)
    obs9 = Obstacle(cx=0.67, cy=-0.05, radius=0.08)
    map.add_obstacles(
        [obs1, obs2, obs3, obs4, obs5, obs6, obs7, obs8, obs9])

    return reference_path


def MPC_Problem_setup(reference_path):
    '''
    Get configured do-mpc modules:
    '''
    # model setup
    bicycle = simple_bycicle_model(reference_path, 0.12, 0.06,0.05) 
    bicycle.model_setup()

    Controller = MPC(bicycle)

    Sim = Simulator(bicycle)

    return bicycle, Controller, Sim


if __name__ == '__main__':

    reference_path = environment_setup()
    bicycle, Controller, Sim = MPC_Problem_setup(reference_path)

    #Initial state from environment
    x0 = np.array([bicycle.reference_path.waypoints[0].x, bicycle.reference_path.waypoints[0].y,
                   bicycle.reference_path.waypoints[0].psi, 0.3, 0])
    Controller.mpc.x0 = x0
    Sim.simulator.x0 = x0

    # Use initial state to set the initial guess.
    Controller.mpc.set_initial_guess()
    
    import time

    # Start timing
    start_time = time.time()

    '''
    Run MPC main loop:
    '''
    t = 0
    while current_progress < reference_path.length:
        #Control
        u = Controller.control(x0)
        # Simulate car
        x0 = Sim.simulator.make_step(u)
        Controller.distance_update(x0)
        pred_x = Controller.mpc.data.prediction(('_x', 'x'), t_ind=t)[0]
        pred_y = Controller.mpc.data.prediction(('_x', 'y'), t_ind=t)[0]
        reference_path.show(pred_x, pred_y)
        Sim.show(x0)

        # update boundary
        Controller.constraints()

        t += 1
        plt.axis('off')
        plt.pause(0.001)


#################################################################################

# === End timing ===
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total simulation time: {elapsed_time:.2f} seconds")

