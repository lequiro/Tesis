# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:57:23 2024

@author: Luis Quispe
"""

import numpy as np
from scipy.fft import fft, ifft
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython import get_ipython
get_ipython().run_line_magi"c('matplotlib', 'qt5')

"""
L = 100  # Domain length
T = 50  # Total simulation time
nx = 256  # Number of grid points
nt = 100  # Number of time steps
nu = 0.1  # Diffusion coefficient
A = 1.0  # Nonlinearity coefficient


x,dx = np.linspace(0, L, nx,endpoint=False, retstep=True)
t,dt = np.linspace(0,T, nt, endpoint=False, retstep=True)
k = 2*np.pi*np.fft.fftfreq(nx, d=dx) 

u = np.ones((nx, nt))
u_hat = np.ones((nx, nt), dtype=complex)



u0 = np.cos((2 * np.pi * x) / L) + 0.1 * np.cos((4 * np.pi * x) / L) #condicion inicial
u0_hat = (1 / nx) * np.fft.fftshift(np.fft.fft(u0))

u[:,0] = u0
u_hat[:,0] = u0_hat



for i, valor in enumerate(t[:-1]):
    k1 = dt * (k**2 * u_hat[:,i] - k**4 * u_hat[:,i] + (k**2)/2 * np.fft.fft(np.real(np.fft.ifft(u_hat[:,i]))**2))
    
    k2 = dt * (k**2 * (u_hat[:,i] + k1/2) - k**4 * (u_hat[:,i] + k1/2) + (k**2)/2 * np.fft.fft(np.real(np.fft.ifft((u_hat[:,i] + k1/2)))**2))
    u_hat[:,i+1]= u_hat[:,i] + k2
    
    u[:,i+1] =  np.real(nx * np.fft.ifft(np.fft.ifftshift(u_hat[:,i+1])))
"""


#%%

L = 50  # Longitud del dominio
T = 0.36  # Tiempo total de simulación
nx = 512  # Número de puntos de la cuadrícula
nt = 10000  # Número de pasos de tiempo


x, dx = np.linspace(0, L, nx, endpoint=False, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=False, retstep=True)
k = np.arange(-nx/2, nx/2, 1)

u = np.ones((nx, nt))
u_hat = np.ones((nx, nt), dtype=complex)

u0 = np.cos((2 * np.pi * x) / L) + 0.1 * np.cos((4 * np.pi * x) / L)  # Condición inicial
u0_hat = (1 / nx) * np.fft.fftshift(np.fft.fft(u0))

u[:, 0] = u0
u_hat[:, 0] = u0_hat

for i, valor in enumerate(t[:-1]):
    
    #esta parte calcula el término no lineal
    fx = 1j*k*u_hat[:,i]
    ux   =  np.fft.irfft(fx)
    fux = np.fft.rfft(ux**2)
    
    k1 = dt * (k**2 * u_hat[:, i] - k**4 * u_hat[:, i] - fux/2)
    
    #calcula  el término no lineal evaluado en u_hat[:, i] + k1 / 2
    fx_2 = 1j*k*(u_hat[:, i] + k1 / 2)
    ux_2   =  np.fft.irfft(fx_2)
    fux_2 = np.fft.rfft(ux_2**2)
        
    k2 = dt * (k**2 * (u_hat[:, i] + k1 / 2) - k**4 * (u_hat[:, i] + k1 / 2) - fux_2/2)
    u_hat[:, i + 1] = u_hat[:, i] + k2

    # Aplicar la transformada inversa de Fourier solo a la columna actual
    u[:, i + 1] = np.real(nx * np.fft.ifft(np.fft.ifftshift(u_hat[:, i + 1])))


plt.figure(figsize=(10, 6))
plt.imshow(u, extent=[0, T, 0, L], aspect='auto')
plt.colorbar(label='u')
plt.xlabel('Tiempo')
plt.ylabel('Espacio')
plt.show()

#%%
#%%
plt.plot(k,u0_hat)
