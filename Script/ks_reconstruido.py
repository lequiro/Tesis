import numpy as np
# from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'qt5')

#%%
L = 22  # Longitud del dominio
T = 100  # Tiempo total de simulación
nx = 64  # Número de puntos de la cuadrícula
dt = 1e-3
nt = int(T/dt)  # Número de pasos de tiempo

x, dx = np.linspace(0, L, nx, endpoint=False, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=False, retstep=True)
k = 2*np.pi*np.fft.rfftfreq(nx, dx)

u = np.ones((nx, nt))
u_hat = np.ones((nx//2 + 1, nt), dtype=complex)

# Condicion inicial
# Deben ser L-periodicas
p  = 2*np.pi/L
# u0 = np.sin(p*x) + 0.5*np.sin(2*p*x) + 0.1*np.sin(3*p*x - L/3)
u0 = np.cos((2 * np.pi * x) / L) + 0.1 * np.cos((4 * np.pi * x) / L)
u0_hat = np.fft.rfft(u0)

u[:, 0] = u0
u_hat[:, 0] = u0_hat

# Parseval para DFT
# Esto compara la energía total, no media.
# El 2.0 en la parte de Fourier está para
# compensar por lo numeros de onda con k<0
# que no se usan en la rfft.
real_space = np.sum(u0**2)

uh = np.real(u0_hat*u0_hat.conjugate())
uh[1:] = 2.0*uh[1:] 
fourier_space = (1/nx)*np.sum(uh)
print("Check Parseval", real_space, fourier_space)

# Check derivatives
ux = np.gradient(u0, dx)
fx = np.fft.irfft(1.0j*k*u0_hat)
plt.figure()
plt.plot(x, ux, label='fin dif')
plt.plot(x, fx, label='fourier')
plt.legend()

def first_deriv(uu, kk):
    return 1.0j*kk*uu

def second_deriv(uu, kk):
    return -kk**2*uu

def fourth_deriv(uu, kk):
    return kk**4*uu

ord = 2 # order of RK
for i in range(1, nt):
    
    u_prev = u_hat[:, i-1]
    u_hat[:, i] = u_prev
    for oo in range(ord, 0, -1):
        # Non-linear term
        fx = u_hat[:,i]
        ux = np.fft.irfft(fx)
        fux = np.fft.rfft(ux**2)
        
        u_hat[:,i] = u_prev + (dt/oo) * (
            - 0.5*first_deriv(fux, k)
            - second_deriv(u_hat[:,i], k) 
            - fourth_deriv(u_hat[:,i], k)
            )

        # de-aliasing
        u_hat[0, i] = 0.0
        u_hat[nx//3:, i] = 0.0

u = np.fft.irfft(u_hat, axis=0)

plt.figure()
plt.plot(u[:, -1])

plt.figure(figsize=(10, 6))
plt.imshow(u, extent=[0, T, 0, L], aspect='auto')
plt.colorbar(label='u')
plt.xlabel('Tiempo')
plt.ylabel('Espacio')

plt.figure()
plt.loglog(k, np.abs(u_hat[:,-1])**2)

plt.figure()
plt.plot(t, np.mean(u**2, axis=0))

plt.show()
