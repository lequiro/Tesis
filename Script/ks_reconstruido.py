import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

#%%
def solve_KS(initial_condition, spatial_resolution, time_steps, order=4):
    '''
    Nota: k y dt tienen que ser definidos previamente 
        
    Parameters
    ----------
    initial_condition : u0
    spatial_resolution : nx ---> número de puntos espaciales
    time_steps : nt ---> número de puntos temporales
    order : El orden del Runge-Kutta

    Returns
    -------
    u : Real space u
    u_hat : Fourier space u
    
    '''
    u = np.ones((spatial_resolution, time_steps))
    u_hat = np.ones((spatial_resolution//2 + 1, time_steps), dtype=complex)
    
    initial_condition_hat = np.fft.rfft(initial_condition)
    u[:, 0] = initial_condition
    u_hat[:, 0] = initial_condition_hat

    for i in tqdm(range(1, time_steps)):
        u_prev = u_hat[:, i-1]
        u_hat[:, i] = u_prev
        for oo in range(order, 0, -1):
            # Non-linear term
            fx = u_hat[:,i]
            ux = np.fft.irfft(fx)
            fux = np.fft.rfft(ux**2)
            
            u_hat[:,i] = u_prev + (dt/oo) * (
                - (0.5*1.0j*k*fux)
                + ((k**2)*u_hat[:,i]) 
                - ((k**4)*u_hat[:,i])
                )

            # de-aliasing
            u_hat[0, i] = 0.0 #mandar el modo cero a cero
            u_hat[spatial_resolution//3:, i] = 0.0 #matar los modos espúreos

    u = np.fft.irfft(u_hat, axis=0)
    
    return u, u_hat

#%%
L = 22 # Longitud del dominio
T = 60  # Tiempo total de simulación
nx = 64 # Número de puntos de la cuadrícula
dt = 1e-3
nt = int(T/dt)  # Número de pasos de tiempo

x, dx = np.linspace(0, L, nx, endpoint=False, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=False, retstep=True)
k = 2*np.pi*np.fft.rfftfreq(nx, dx) #te da las frecuencias en radianes dado un número de puntos y un espaciado



#Condiciones Iniciales: deben ser L-periódicas
p  = 2*np.pi/L
# u0 = np.sin(p*x) + 0.5*np.sin(2*p*x) + 0.1*np.sin(3*p*x - L/3)
u0 = np.cos(p* x) + 0.1 * np.cos(2*p* x)


u, u_hat = solve_KS(u0, nx, nt)

#corto el transitorio
u = u[:,20_000:]
#%%

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot first graph
axs[0, 0].imshow(u, extent=[20, T, 0, L], aspect='auto')
axs[0, 0].set_title('u')
axs[0, 0].set_xlabel('Tiempo')
axs[0, 0].set_ylabel('Espacio')
fig.colorbar(axs[0, 0].imshow(u, extent=[0, T, 0, L], aspect='auto'), ax=axs[0, 0])

# Plot second graph
axs[0, 1].loglog(k, np.abs(u_hat[:,-1]))
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('Densidad de energía $|u(k)|$')
axs[0, 1].set_title('Densidad de energía')

# Plot third graph
seleccion_x = np.sort(np.random.choice(nx, size=20, replace=False))
for i in seleccion_x:
    axs[1, 0].plot(t[20_000:], u[i, :])
axs[1, 0].set_xlabel('Tiempo')
axs[1, 0].set_ylabel('Amplitud')
axs[1, 0].set_title('20 selecciones aleatorias de u')

# Plot fourth graph
real_space = np.sum(np.square(u), axis=0)
axs[1, 1].plot(t[20_000:], real_space)
axs[1, 1].set_xlabel('Tiempo')
axs[1, 1].set_ylabel('Energía total $\sum_{columnas} u^2$')
axs[1, 1].set_title('Energía total')

plt.tight_layout()
plt.show()
#%%
plt.close('all')
plt.loglog( k, np.abs(u_hat[:,2]), label= 't=2e-3')
plt.loglog( k, np.abs(u_hat[:,1_000]), label= 't=1')
plt.loglog( k, np.abs(u_hat[:,2_000]), label= 't=2')
plt.loglog( k, np.abs(u_hat[:,5_000]), label= 't=5')
plt.loglog( k, np.abs(u_hat[:,10_000]), label= 't=10')
# plt.loglog( k, np.abs(u_hat[:,20_000]), label= 't=20')
# plt.loglog( k, np.abs(u_hat[:,40_000]), label= 't=40')
plt.legend()