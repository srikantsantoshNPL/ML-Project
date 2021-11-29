# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 16:01:48 2021

@author: ss38
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import hilbert,chirp


f = np.linspace(0,200,200)
t = np.linspace(0,1,200)
f_l = 100
omega_l = 2*np.pi*f_l
def X(f):
    return 1/(1+(f**2))
def Y(f):
    return f/(1+f**2)
# plt.plot(f,X(f-100))
# plt.plot(f,Y(f-100))
def M_x(t,f):
    return X(f-100)*np.cos(omega_l*t)+Y(f-100)*np.sin(omega_l*t)
#plt.plot(t,M_x(t,f)*np.sin(50*t))

analytic_signal = hilbert(M_x(t,f))
amplitude_envelope = np.abs(analytic_signal)
inst_phase = np.unwrap(np.angle(analytic_signal))
#plt.plot(t,amplitude_envelope)
X_new = amplitude_envelope*np.cos(inst_phase)
Y_new = amplitude_envelope*np.sin(inst_phase)
plt.plot(t,X_new)
plt.plot(t,Y_new)
plt.plot(t,amplitude_envelope)

