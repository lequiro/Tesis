import json
import os
import pickle
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from rk4_functions import RK4Model
from cnn_functions import build_autoencoder, build_decoder, build_encoder, neural_configuration

version = 3
##########################################-.LOAD PARAMS AND DATASETS.-########################################

json_files = sorted([f for f in os.listdir('../../autoencoder/files') if f.endswith('.json')])

with open(f'../../autoencoder/files/{json_files[0]}') as f:
    params = json.load(f)

with open(f"../version_{version}/train_dataset.pkl", "rb") as f:
    dataset_loaded = pickle.load(f)
    
with open(f"../version_{version}/encoded_dataset.pkl", "rb") as f:
    encoded_dataset = pickle.load(f)

initial_conditions = np.array(list(zip(*dataset_loaded))[0])
u_final = np.array(list(zip(*dataset_loaded))[1])
times = np.array(list(zip(*dataset_loaded))[-1])

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

dt = 1e-4

##########################################-.PROCESS ENTIRE ENCODED DATASET.-########################################

encoded_dataset = np.array(encoded_dataset, dtype=object)  
num_samples = len(encoded_dataset)

decoded_final_states_all = []
final_states_all = []

batch_size = 2048
num_batches = (num_samples + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, num_samples)
    batch = encoded_dataset[start_idx:end_idx]
    
    initial_state_vector_batch = np.array(list(zip(*batch))[0])
    num_steps_value_batch = np.array(list(zip(*batch))[-1])
    
    final_states = rk4_model.forward_rk4_nn(tf.constant(np.array(initial_state_vector_batch), dtype=tf.float32),
                                            tf.constant(np.array(num_steps_value_batch), dtype=tf.int32), dt) # shape: (batch, 8)

    decoded_final_states_all.append(decoder(final_states).numpy().squeeze(axis=-1))
    final_states_all.append(final_states.numpy())


decoded_final_states_all = np.concatenate(decoded_final_states_all, axis=0)
final_states_all = np.concatenate(final_states_all, axis=0)

np.save(f'../version_{version}/encoded_final_states_rk4.npy', final_states_all)
np.save(f'../version_{version}/decoded_final_states_rk4.npy', decoded_final_states_all)