'''
    Movement primitives applied to a 2D trajectory

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
from math import factorial
import os
from pathlib import Path

# Building piecewise constant basis functions
def build_phi_piecewise(param):
    index_list = np.linspace(0, param["nbData"], param["nbFct"]+1)
    phi = np.zeros((param["nbData"],param["nbFct"]))
    for i in range(param["nbFct"]):
        phi[int(index_list[i]):int(index_list[i+1]), i] = 1
    return phi

# Building radial basis functions (RBFs)
def build_phi_rbf(param):
    Ts = np.arange(0,param["nbData"],1)
    bw = param["nbData"] / param["nbFct"] 
    avg = bw // 2
    sig = bw
    phi = []
    for i in range(param["nbFct"]):
        phi_k = 1 / (2*np.pi * sig) * np.exp(-1*(Ts-avg)**2 / (2*sig**2))
        phi += [phi_k]
        avg += bw
    return np.asarray(phi).T

# Building Bernstein basis functions
def build_phi_bernstein(param):
    t = np.linspace(0,1,param["nbData"])
    phi = np.zeros((param["nbData"],param["nbFct"]))
    for i in range(param["nbFct"]):
        phi[:,i] = factorial(param["nbFct"]-1) / (factorial(i) * factorial(param["nbFct"]-1-i)) * (1-t)**(param["nbFct"]-1-i) * t**i
    return phi

# Building Fourier basis functions
def build_phi_fourier(param):
    d = np.ceil((param["nbFct"]-1)/2)
    k = np.arange(-d,d+1,1).reshape((-1,1))
    param["nbFct"] = len(k)
    t = np.linspace(0,1,param["nbData"]).reshape((-1,1))
    phi = np.exp( t.T * k * 2 * np.pi * 1j ) / (param["nbData"])
    return phi.T


# General param parameters
# ===============================
param = {
    "nbFct" : 10, # Number of basis functions
    "nbVar" : 2, # Dimension of position data 
    "nbData" : 200, # Number of datapoints in a trajectory
    "nbDemos" : 5, # Number of demonstrations
}

# Load handwriting data
FILE_PATH = os.path.dirname(os.path.realpath(__file__))
DATASET_LETTER_FILE = "data/2Dletters/S.npy"
# x = np.load(str(Path(FILE_PATH,DATASET_LETTER_FILE)))[0,:,:2].flatten() #Load single demonstration
X = np.load(str(Path(FILE_PATH,DATASET_LETTER_FILE)))[:param["nbDemos"],:,:2].reshape((param["nbDemos"],param["nbData"]*2)).T #Load multiple demonstrations

t = np.linspace(0,1,param["nbData"]) # Time range

# Fourier basis functions require additional symmetrization (mirroring) if the signal is a discrete motion 
# x = np.hstack(( x , x[::-1] ))
# param["nbData"] *= 2


# Generate MP with various basis functions
# ============================================
# phi = build_phi_piecewise(param) #Piecewise constant
# phi = build_phi_rbf(param) #Radial basis functions
phi = build_phi_bernstein(param) #Bernstein polynomials 
# phi = build_phi_fourier(param) #Fourier basis functions

psi = np.kron(phi, np.identity(param["nbVar"])) # Transform to multidimensional basis functions

# w = np.linalg.solve(psi.T @ psi + np.identity(param["nbVar"] * param["nbFct"]) * 1e-8, psi.T @ x) # Estimation of superposition weights from data
# w = np.linalg.pinv(psi) @ x # Estimation of superposition weights from data
# w = np.linalg.inv(psi.T @ psi) @ psi.T @ x # Estimation of superposition weights from data
W = np.linalg.pinv(psi) @ X # Estimation of superposition weights from data (for multiple demonstrations)
w = np.mean(W, 1) # Average w

x_hat = psi @ w # Reconstruction data

# Fourier basis functions require de-symmetrization of the signal after processing (for visualization)
# param["nbData"] /= 2
# x = x[:int(param["nbData"]*param["nbVar"])]
# x_hat = x_hat[:int(param["nbData"]*param["nbVar"])] 


# Plotting
# =========
fig,axs = plt.subplots(2,1)
# axs[0].plot(x[::2], x[1::2], c='black', label='Demonstration')
axs[0].plot(X[::2], X[1::2], c='black', label='Demonstration')
axs[0].plot(x_hat[::2], x_hat[1::2], c='r', label='Reproduction')
axs[0].axis("off")
axs[0].axis("equal")
axs[0].legend()

axs[1].set_xticks([0,param["nbData"]])
axs[1].set_xticklabels(["0","T"])
for j in range(param["nbFct"]):
    axs[1].plot(phi[:,j])
axs[1].set_ylabel("$\phi_k$")

plt.show()
