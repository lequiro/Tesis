import numpy as np
from tqdm import tqdm
import os
#directorio donde guardo los datasets
dataset = 3
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Convolutional_dataset{dataset}'
os.chdir(path)
#%%
def solve_KS(initial_condition, spatial_resolution, time_steps, order=4, corte_transitorio=40_000, save_interval=10):
    initial_condition_hat = np.fft.rfft(initial_condition)
    u_hat = np.copy(initial_condition_hat)

    output = []
    output_hat = []

    for i in tqdm(range(1, time_steps)):
        u_prev = np.copy(u_hat)

        for oo in range(order, 0, -1):
            # Non-linear term
            ux = np.fft.irfft(u_prev)
            fux = np.fft.rfft(ux**2)

            u_hat = u_prev + (dt/oo) * (
                - (0.5*1.0j*k*fux)
                + ((k**2)*u_hat) 
                - ((k**4)*u_hat)
                )

            # De-aliasing
            u_hat[0] = 0.0  # Mandar el modo cero a cero
            u_hat[spatial_resolution//3:] = 0.0  # Matar los modos espúreos

        if i >= corte_transitorio and i % save_interval == 0:
            u = np.fft.irfft(u_hat, axis=0)
            output.append(u)
            output_hat.append(u_hat)

    return np.array(output).T, np.array(output_hat).T


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


x, dx = np.linspace(0, L, nx, endpoint=True, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=True, retstep=True)
k = 2*np.pi*np.fft.rfftfreq(nx, dx) #te da las frecuencias en radianes dado un número de puntos y un espaciado
p  = 2*np.pi/L

#Creo el data set de entrenamiento
u0 = 0.3*np.cos(3*p*x) + 0.4*np.cos(5*p*x) + .5*np.cos(4*p*x)
u_train, u_hat_train = solve_KS(u0, nx, nt)
u_train = u_train / np.max(np.abs(u_train),axis=0) #normalización


#Creo el data set de test

u0 = 0.3*np.cos(p*x) + 0.4*np.cos(2*p*x) + .5*np.cos(5*p*x +  7*np.pi ) + 0.2*np.cos(4*p*x + 0.5*np.pi)
u_test, u_hat_test = solve_KS(u0, nx, nt)
u_test = u_test / np.max(np.abs(u_test),axis=0) #normalización

#%%
#guardo los datasets
np.save(r'u_train.npy', u_train)
np.save(r'u_hat_train.npy', u_hat_train)
np.save(r'u_test.npy', u_test)
np.save(r'u_hat_test.npy', u_hat_test)
np.savez(r'data.npz', L=L, T=T, nx=nx, t=t[40_000:], k=k)
