import json
import os
import pickle
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from rk4_functions import RK4Model
from cnn_functions import build_autoencoder, build_decoder, build_encoder, neural_configuration

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

version = 3
######################################################-.EVOLUTION WITH PSEUDOESPECTRAL.-############################################3

with open(f"../version_{version}/train_dataset.pkl", "rb") as f:
    dataset_loaded = pickle.load(f)

initial_conditions = np.array(list(zip(*dataset_loaded))[0])
u_final = np.array(list(zip(*dataset_loaded))[1])
times = np.array(list(zip(*dataset_loaded))[-1])

L = 22 # Longitud del dominio
T = 3 # Tiempo total de simulación
nx = 128 # Número de puntos de la cuadrícula
dt = 0.0001
nt = int(T/dt)  # Número de pasos de tiempo

x, dx = np.linspace(0, L, nx, endpoint=True, retstep=True)
t, dt = np.linspace(0, T, nt, endpoint=True, retstep=True)
k = 2*np.pi*np.fft.rfftfreq(nx, dx) #te da las frecuencias en radianes dado un número de puntos y un espaciado
p  = 2*np.pi/L

def solve_KS(initial_condition, spatial_resolution, time_steps, order=4):
    
    initial_condition_hat = np.fft.rfft(initial_condition)
    u_hat = np.copy(initial_condition_hat)

    output = []
    for i in range(time_steps):
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

        u = np.fft.irfft(u_hat, axis=0)
        output.append(u)
            
        if i % 10_000 == 0:
            print('va por el: ', i/10_000)

    return np.array(output).T

idx = 6
og_evolution = solve_KS(initial_conditions[idx], 128 , 30_000)

####################################################################-.NODE.-###################################################################3

##########################################-.LOAD PARAMS AND DATASETS.-########################################

json_files = sorted([f for f in os.listdir('../../autoencoder/files') if f.endswith('.json')])

with open(f'../../autoencoder/files/{json_files[0]}') as f:
    params = json.load(f)
    
with open(f"../version_{version}/encoded_dataset.pkl", "rb") as f:
    encoded_dataset = pickle.load(f)
    
##########################################-.MODEL PARAMETERS.-########################################

dh = params['dh']
lr = params['lr']
lista_neuronas_encoder = params['lista_neuronas_encoder']
kernel_size = params['kernel_size']
strides = params['strides']
config_encoder, config_decoder = neural_configuration(lista_neuronas_encoder, kernel_size, strides)

##########################################-.BUILD AUTOENCODER.-########################################

autoencoder = build_autoencoder(params['N'], config_encoder, config_decoder, dh, lr)

h5_files = [file for file in os.listdir('../../autoencoder/files') if file.endswith('.h5')]
autoencoder.load_weights(f'../../autoencoder/files/{h5_files[-1]}')

##########################################-.BUILD MODELS.-########################################

input_shape = (params['N'], 1)
encoder, encoded_shape = build_encoder(input_shape, config_encoder, params['dh'])
decoder = build_decoder(encoded_shape, config_decoder, dh)

encoder.set_weights(autoencoder.get_layer('encoder').get_weights())
decoder.set_weights(autoencoder.get_layer('decoder').get_weights())

encoder.summary()
decoder.summary()

##########################################-.BUILD NODE.-########################################

base_model = Sequential([
    Input(shape=(8,), name='input_layer'),
    Dense(16, activation='tanh', name='dense'),
    Dense(16, activation='tanh', name='dense_1'),
    Dense(8, name='dense_2')
])

rk4_model = RK4Model(base_model)
rk4_model.base_model.summary()


weights_directory = sorted([f for f in os.listdir(f"../version_{version}") if f.endswith('.h5')])
rk4_model.load_weights(f'../version_{version}/{weights_directory[-1]}')

##########################################-.NODE EVOLUTION.-########################################


batch = encoded_dataset[idx:idx+2]
initial_state_vector_batch = np.array(list(zip(*batch))[0])
num_steps_value_batch = np.array([30_000,30_000])


_, final_states = rk4_model.forward_rk4_nn_all( tf.constant(np.array(initial_state_vector_batch), dtype=tf.float32),
                                        tf.constant(np.array(num_steps_value_batch),dtype=tf.int32) , 
                                        tf.constant(dt,dtype=tf.float32))

first_signal_decoded = decoder( final_states[:30000, 0, :].numpy()).numpy().squeeze(axis=-1).T

##########################################-.PLOTS.-########################################

fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)


axes[0].set_title("OG Evolution (True Solution)")
img1 = axes[0].imshow(og_evolution, aspect='auto', cmap='viridis', 
                      extent=[0, og_evolution.shape[1]*dt, 0, L])
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Spatial coordinate")
cbar1 = fig.colorbar(img1, ax=axes[0], label='u')


axes[1].set_title("First Signal Decoded (NODE + Decoder)")
img2 = axes[1].imshow(first_signal_decoded, aspect='auto', cmap='viridis', 
                      extent=[0, first_signal_decoded.shape[1]*dt, 0, L])
axes[1].set_xlabel("Time")
axes[1].set_ylabel("Spatial coordinate")
cbar2 = fig.colorbar(img2, ax=axes[1], label='u')

plt.show()


point_index = 64
time = np.arange(0, og_evolution.shape[1]*dt, dt)

plt.figure(figsize=(8,4))
plt.title(f"Time Evolution at Spatial Point Index {point_index}")
plt.plot(time, og_evolution[point_index, :], label='Original (OG Evolution)', alpha=0.7)
plt.plot(time, first_signal_decoded[point_index, :], label='NODE + Decoder', alpha=0.7)
plt.xlabel("Time")
plt.ylabel("u")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 8))

plt.plot(og_evolution[:, -1], label="Solve_KS evolution", linestyle='-', color='blue', linewidth=2)
plt.plot(u_final[idx, :], label="NODE_evolution", linestyle='--', color='orange', linewidth=2)

plt.title("Comparación de Evolución temporal", fontsize=24)
plt.xlabel("x", fontsize=14)
plt.ylabel(r"$u(x,\tau)$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)


plt.tight_layout()
plt.show()


