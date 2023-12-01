#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:25:18 2023

@author: lucasfragara
"""

from copy import deepcopy
import SimAnn as SA
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from generate_data import generate_data
import numpy as np
import sys
import time
sys.path.append(
    "/Users/lucasfragara/Desktop/Bocconi/CS 2/Individual assignment")


class Optimizer:

    def __init__(self, n, seed=None):

        if seed:
            np.random.seed(seed)

        self.data = generate_data(n, "3233248")
        self.n = n
        self.config = self.init_config()  # {x: , y:}

    def init_config(self):

        # TODO: Check this covers all possible combinations (i.e even borders)
        return {"x": np.random.randint(0, self.n-1), "y": np.random.randint(0, self.n-1)}

    def cost(self):

        x_t, y_t = self.config["x"], self.config["y"]

        return self.data[x_t, y_t]

    def propose_move(self):
        n = self.n

        x_prop = np.random.choice(
            [(self.config["x"]-1) % n, (self.config["x"]+1) % n])
        y_prop = np.random.choice(
            [(self.config["y"]-1) % n, (self.config["y"]+1) % n])

        return (x_prop, y_prop)

    def accept_move(self, move):

        x_prop, y_prop = move

        self.config["x"] = x_prop
        self.config["y"] = y_prop

    def compute_delta_cost(self, move):

        x_prop, y_prop = move
        x_t, y_t = self.config["x"], self.config["y"]

        return self.data[x_prop, y_prop] - self.data[x_t, y_t]

    def display(self, fig):

        plt.clf()
        ax = plt.axes(projection='3d')

        x = np.arange(self.n)
        y = np.arange(self.n)

        X, Y = np.meshgrid(x, y)

        z = self.data

        ax.scatter(self.config["x"], self.config["y"],
                   self.data[self.config["x"], self.config["y"]], color="red", s=100)

        ax.plot_surface(Y, X, z, alpha=0.5)
        ax.set_xlabel("x-axis")
        ax.set_ylabel("y-axis")
        ax.set_zlabel("z-axis")

    def copy(self):
        return deepcopy(self)


optimizer = Optimizer(1000, seed=123)
print(optimizer.config)

# best_probl = SA.simann(optimizer, mcmc_steps=100,
#                      beta_list=SA.linearbeta(0.001, 1, 1000), seed=10)

beta_list = SA.linearT(500, 100)

best_probl, hist = SA.simann(optimizer, mcmc_steps=10,
                             beta_list=beta_list, seed=10)


print("Actual min: ", np.min(best_probl.data))

#fig = plt.figure()
best_probl.display(fig)
