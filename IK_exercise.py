'''
    Forward and inverse kinematics example

    Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
    Written by Sylvain Calinon <https://calinon.ch>

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
import matplotlib.pyplot as fig

T = 50 #Number of datapoints
D = 3 #State space dimension (x1,x2,x3)
l = np.array([2, 2, 1]); #Robot links lengths
fh = np.array([-2, 1]) #Desired target for the end-effector
x = np.ones(D) * np.pi / D #Initial robot pose

# EXERCISE
# =====================================
# 1) Replace the line below by a forward kinematics function f=f(x) that computes the location of
# the different articulations of the robot, including the end-effector.
# Try to write this function so that it can be used with any dimension D, and try to write it
# in matrix form to have a code that is as compact as possible.
f = np.random.rand(2,D);
f = np.concatenate((np.zeros([2,1]), f), axis=1) #Add robot base (for plotting)

# 2) Write an inverse kinematics algorithm that will compute x so that the end-effector f[:,-1]
# reaches a desired target fh.
# Try to write this function so that it can be used with any dimension D, and try to write it
# in matrix form to have a code that is as compact as possible.

fig.scatter(fh[0], fh[1], color='r', marker='.', s=10**2) #Plot target
fig.plot(f[0,:], f[1,:], color='k', linewidth=1) #Plot robot

fig.axis('off')
fig.axis('equal')
fig.show()
