import numpy as np
import pickle

dataset = 1
npz_path = f'../../data_KS/Convolutional_dataset{dataset}/data.npz'
data = np.load(npz_path)

def solve_KS(initial_condition, spatial_resolution, time_steps, order=4):
    initial_condition_hat = np.fft.rfft(initial_condition)
    u_hat = np.copy(initial_condition_hat)

    u = np.fft.irfft(u_hat, axis=0)

    for i in range(1, time_steps):
        u_prev = np.copy(u_hat)

        for oo in range(order, 0, -1):
            # Término no lineal
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
            
    u = np.fft.irfft(u_hat, axis=0)

    return u

def generar_funcion_periodica_aleatoria(num_terminos):
    funcion = np.zeros_like(x)
    for i in range(num_terminos):
        frecuencia = np.random.randint(1, 5)  # Frecuencia aleatoria
        amplitud = np.random.uniform(0.4, 0.95)  # Amplitud aleatoria
        fase = np.random.uniform(0, 2*np.pi)  # Fase aleatoria
        funcion = funcion +  amplitud * np.cos(frecuencia * p * x + fase)
    return funcion

def generar_dataset(num_iterations, u0, max_value):
    dataset = []

    for i in range(num_iterations):
        gen_timesteps = np.random.randint(nt)

        u = solve_KS(u0, nx, gen_timesteps)
        normalized_u = u / max_value
        
        normalized_u0 = u0 / max_value

        data_tuple = (normalized_u0, normalized_u, gen_timesteps)
        dataset.append(data_tuple)
        u0 = np.copy(u)
        if i % 100 == 0:
            print('Va por el:', i)
    return dataset

# Parámetros
L = 22  # Longitud del dominio
T = 3 # Tiempo total de simulación
nx = 128  # Número de puntos de la cuadrícula
dt = 1e-4
nt = int(T/dt)  # Número de pasos de tiempo

x, dx = np.linspace(0, L, nx, endpoint=True, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=True, retstep=True)
k = 2*np.pi*np.fft.rfftfreq(nx, dx)  # Frecuencias en radianes
p = 2*np.pi / L

u0 = generar_funcion_periodica_aleatoria(5)
u0 = solve_KS(u0, nx, 50_000)
max_value = data['max_train']  # Use max_value from the .npz file
final_dataset = generar_dataset(10_000, u0, max_value)


#Guardar el dataset
with open("final_dataset.pkl", "wb") as f:
    pickle.dump(final_dataset, f)
