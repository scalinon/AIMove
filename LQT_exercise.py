'''
    Linear Quadratic tracker applied on a via point example

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Jeremy Maceiras <jeremy.maceiras@idiap.ch>,
    Sylvain Calinon <https://calinon.ch>

    This file is part of RCFS.

    RCFS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License version 3 as
    published by the Free Software Foundation.

    RCFS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with RCFS. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import matplotlib.pyplot as plt

# Parameters
# ===============================
param = {
    "nbData" : 100, # Number of data points
    "nbVarPos" : 2, # Dimension of position data
    "nbDeriv" : 2,  # Number of static and dynamic features (2 -> [x,dx])
    "dt" : 1e-2, # Time step duration
    "rfactor" : 1e-8 # Control cost
}

nb_var = param["nbVarPos"] * param["nbDeriv"] # Dimension of state vector
R = np.identity( (param["nbData"]-1) * param["nbVarPos"] ) * param["rfactor"]  # Control cost matrix

param["muQ"] = np.vstack((  # Sparse reference
    np.zeros(((param["nbData"]-1) * nb_var , 1)),
    np.vstack(( 1, 2, np.zeros((nb_var - param["nbVarPos"],1)) ))
))

Q = np.zeros(( nb_var * param["nbData"] , nb_var * param["nbData"] ))   # Task precision
Q[-nb_var:,-nb_var:] = np.identity(nb_var)


# EXERCISE
# ============================================
# 1) Extend the code to a viapoints task (instead of a single target to reach)
# 2) Observe the different behaviors when using velocity commands or acceleration commands


# Dynamical System settings (discrete)
# =====================================
A = np.identity(nb_var)
if param["nbDeriv"]==2:
    A[:param["nbVarPos"],-param["nbVarPos"]:] = np.identity(param["nbVarPos"]) * param["dt"]

B = np.zeros((nb_var , param["nbVarPos"]))
derivatives = [ param["dt"],param["dt"]**2 /2 ][:param["nbDeriv"]]
for i in range(param["nbDeriv"]):
    B[i*param["nbVarPos"]:(i+1)*param["nbVarPos"]] = np.identity(param["nbVarPos"]) * derivatives[::-1][i]

# Build Sx and Su transfer matrices
Su = np.zeros((nb_var*param["nbData"],param["nbVarPos"] * (param["nbData"]-1))) # It's maybe n-1 not sure
Sx = np.kron(np.ones((param["nbData"],1)),np.eye(nb_var,nb_var))

M = B
for i in range(1,param["nbData"]):
    Sx[i*nb_var:param["nbData"]*nb_var,:] = np.dot(Sx[i*nb_var:param["nbData"]*nb_var,:],A)
    Su[nb_var*i:nb_var*i+M.shape[0],0:M.shape[1]] = M
    M = np.hstack((np.dot(A,M),B)) # [0,nb_state_var-1]


# Batch LQR Reproduction
# =====================================
x0 = np.zeros((nb_var,1))
u_hat = np.linalg.inv(Su.T @ Q @ Su + R) @ Su.T @ Q @ (param["muQ"] - Sx @ x0)
x_hat = (Sx @ x0 + Su @ u_hat).reshape((-1,nb_var))


# Plotting
# =========
plt.figure()
plt.title("2D Trajectory")
plt.scatter(x_hat[0,0],x_hat[0,1],c='black',s=100)
plt.scatter(param["muQ"][-nb_var],param["muQ"][-nb_var+1],c='red',s=100)
plt.plot(x_hat[:,0], x_hat[:,1], c='black')
plt.axis("off")
plt.gca().set_aspect('equal', adjustable='box')

fig,axs = plt.subplots(2,1)
axs[0].scatter(param["nbData"], param["muQ"][-nb_var], color='red')
axs[0].plot(x_hat[:,0], c='black')
axs[0].set_ylabel("$x_1$")
axs[0].set_xticks([0,param["nbData"]])
axs[0].set_xticklabels(["0","T"])

axs[1].scatter(param["nbData"], param["muQ"][-nb_var+1], color='red')
axs[1].plot(x_hat[:,1], c='black')
axs[1].set_ylabel("$x_2$")
axs[1].set_xlabel("$t$")
axs[1].set_xticks([0,param["nbData"]])
axs[1].set_xticklabels(["0","T"])

plt.show()
