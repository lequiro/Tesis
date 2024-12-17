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
##########################################-.LOAD PARAMS AND DATASETS.-########################################

json_files = sorted([f for f in os.listdir('../../autoencoder/files') if f.endswith('.json')])

with open(f'../../autoencoder/files/{json_files[0]}') as f:
    params = json.load(f)

with open(f"../version_{version}/train_dataset.pkl", "rb") as f:
    dataset_loaded = pickle.load(f)
    
with open(f"../version_{version}/encoded_dataset.pkl", "rb") as f:
    encoded_dataset = pickle.load(f)
encoded_dataset = np.array(encoded_dataset, dtype=object) 

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
#%%
##########################################-.PLOTS.-########################################

fig, axes = plt.subplots(2, 1, figsize=(20, 14))

for i in range(5):
    # Decode the initial condition
    initial_decoded = decoder(tf.constant([encoded_dataset[i][0]], dtype=tf.float32)).numpy().squeeze(axis=-1)[0]
    # Decode the final condition
    final_decoded = decoder(tf.constant([encoded_dataset[i][1]], dtype=tf.float32)).numpy().squeeze(axis=-1)[0]

    # Plot initial conditions and their decoded counterparts
    axes[0].plot(initial_conditions[i], 'b.', label='Initial' if i == 0 else "")
    axes[0].plot(initial_decoded, 'r-', alpha=0.6, label='Decoded' if i == 0 else "")
    
    # Plot final conditions and their decoded counterparts
    axes[1].plot(u_final[i], 'b.', label='Final' if i == 0 else "")
    axes[1].plot(final_decoded, 'r-', alpha=0.6, label='Decoded' if i == 0 else "")

axes[0].set_title('Initial Conditions vs. Decoded')
axes[1].set_title('Final Conditions vs. Decoded')

# Add legends only once
axes[0].legend()
axes[1].legend()

plt.tight_layout()
plt.show()