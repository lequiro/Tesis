import numpy as np
from tqdm import tqdm
import os
#directorio donde guardo los datasets
corrida=2
dh=32
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Datasets\dh_{dh}\{corrida}\variables'
os.chdir(path)

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


def generar_funcion_periodica_aleatoria(num_terminos):
    funcion = np.zeros_like(x)
    for i in range(num_terminos):
        frecuencia = np.random.randint(1, 4)  # Frecuencia aleatoria
        amplitud = np.random.uniform(0.1, 0.95)  # Amplitud aleatoria
        # fase = np.random.uniform(0, 2*np.pi)  # Fase aleatoria
        funcion = funcion +  amplitud * np.cos(frecuencia * p * x)
    return funcion

#%%
L = 22 # Longitud del dominio
T = 400 # Tiempo total de simulación
nx = 128 # Número de puntos de la cuadrícula
dt = 1e-4
nt = int(T/dt)  # Número de pasos de tiempo
corte_transitorio = 40_000

x, dx = np.linspace(0, L, nx, endpoint=True, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=True, retstep=True)
k = 2*np.pi*np.fft.rfftfreq(nx, dx) #te da las frecuencias en radianes dado un número de puntos y un espaciado
p  = 2*np.pi/L

#Creo el data set de entrenamiento
u0 = 0.3*np.cos(3*p*x) + 0.4*np.cos(2*p*x) + .5*np.cos(p*x)
u_train, u_hat_train = solve_KS(u0, nx, nt)
u_train = u_train[:,corte_transitorio:]


#Creo el data set de test

u0 = 0.3*np.cos(p*x) + 0.4*np.cos(2*p*x) + .5*np.cos(5*p*x) + 0.2*np.cos(p*x + 0.5*np.pi)
u_test, u_hat_test = solve_KS(u0, nx, nt)
u_test = u_test[:,corte_transitorio:]

#%%
#guardo los datasets
np.save(r'variables\u_train.npy', u_train)
np.save(r'variables\u_hat_train.npy', u_hat_train[:,corte_transitorio:])
np.save(r'variables\u_test.npy', u_test)
np.save(r'variables\u_hat_test.npy', u_hat_test[:,corte_transitorio:])
np.savez(r'variables\data.npz', L=L, T=T, nx=nx, t=t[corte_transitorio:], k=k)