import numpy as np
from scipy.fft import fft, ifft
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

#%%
L = 22  # Longitud del dominio
T = 100  # Tiempo total de simulación
nx = 64  # Número de puntos de la cuadrícula
nt = 10000  # Número de pasos de tiempo


x, dx = np.linspace(0, L, nx, endpoint=False, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=False, retstep=True)
k = np.fft.fftfreq(nx, dx) * 2 * np.pi 

u = np.ones((nx, nt))
u_hat = np.ones((nx, nt), dtype=complex)

u0 = np.sin(x) + np.sin(2*x) + np.sin(3*x)  # Condición inicial
u0_hat = np.fft.fftshift(np.fft.fft(u0))

u[:, 0] = u0
u_hat[:, 0] = u0_hat

for i, valor in enumerate(t[:-1]):
    
    #esta parte calcula el término no lineal
    fx = u_hat[:,i]
    ux   =  np.real(np.fft.ifft(fx))
    fux = np.fft.fft(ux**2)
    
    k1 = dt * ( (k**2 * u_hat[:, i]) - (k**4 * u_hat[:, i]) - 0.5j*k*fux )
    
    #calcula  el término no lineal evaluado en u_hat[:, i] + k1 / 2
    fx_2 = (u_hat[:, i] + k1 / 2)
    ux_2   =  np.real(np.fft.ifft(fx_2))
    fux_2 = np.fft.fft(ux_2**2)
        
    k2 = dt * ( (k**2 * (u_hat[:, i] + k1 / 2)) - (k**4 * (u_hat[:, i] + k1 / 2)) - 0.5j*k*fux_2 )
    u_hat[:, i + 1] = u_hat[:, i] + k2

    # k1 = dt * (Laplacian*u_hat - 0.5j*k*fft(np.real(ifft(u_hat))**2))
    # k2 = dt * (Laplacian*(u_hat + 0.5*k1) - 0.5j*k*fft(np.real(ifft(u_hat + 0.5*k1))**2))
    # k3 = dt * (Laplacian*(u_hat + 0.5*k2) - 0.5j*k*fft(np.real(ifft(u_hat + 0.5*k2))**2))
    # k4 = dt * (Laplacian*(u_hat + k3) - 0.5j*k*fft(np.real(ifft(u_hat + k3))**2))
    # u_hat = u_hat + alpha * (k1 + 2*k2 + 2*k3 + k4) / 6

u = np.real(np.fft.ifft(np.fft.ifftshift(u_hat), axis=0))


plt.figure(figsize=(10, 6))
plt.imshow(u, extent=[0, T, 0, L], aspect='auto')
plt.colorbar(label='u')
plt.xlabel('Tiempo')
plt.ylabel('Espacio')
plt.show()

#%%
plt.plot(k,u0_hat)
