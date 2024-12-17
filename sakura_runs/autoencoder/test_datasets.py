import json
import os
import numpy as np
import pandas as pd
from cnn_functions import neural_configuration, build_autoencoder

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

##########################################-.LOAD PARAMS AND DATASETS.-########################################
directory = 'files'
with open('files/architecture.json') as f:
    params = json.load(f)

dataset = params['dataset']
path = f'../data_KS/Convolutional_dataset{dataset}/'
u_train = np.load(path+'u_train.npy', mmap_mode='r')
u_test = np.load(path+'u_test.npy', mmap_mode='r')

dh = params['dh']
lr = params['lr']
lista_neuronas_encoder = params['lista_neuronas_encoder']
kernel_size = params['kernel_size']
strides = params['strides']
config_encoder, config_decoder = neural_configuration(lista_neuronas_encoder, kernel_size, strides)

###############################################  LOAD MODEL  ###############################################

autoencoder = build_autoencoder(params['N'], config_encoder, config_decoder, dh, lr)
autoencoder.summary()

h5_files = [file for file in os.listdir(directory) if file.endswith('.h5')]
autoencoder.load_weights(os.path.join(directory, f'{h5_files[-1]}'))

##########################################-.VALIDATION MSE BY EPOCH.-########################################

output_data = pd.read_csv('files/output.csv')

fig1 = plt.figure(1, figsize=(14, 8))
plt.plot(output_data['val_MeanSquaredError'], label= 'Validation MSE')

plt.title('Validation and Training Mean Squared Error (Log Scale)', fontsize=30)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Mean Squared Error', fontsize=20)

plt.yscale('log')
plt.legend(fontsize=14, loc='best')

print('El mínimo MSE es:', output_data['val_MeanSquaredError'].min())

##########################################-.PROCESS DATA.-########################################

def process_batches(data, model, batch_size):
    processed_data = []
    num_batches = int(np.ceil(data.shape[0] / batch_size))
    
    for i in range(num_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        processed_batch = model.predict(batch)
        processed_data.append(processed_batch)
        
    return np.concatenate(processed_data, axis=0)

batch_size = 2048

# Process u_train and u_test
encoded_u_train = process_batches(u_train.T, autoencoder, batch_size=batch_size)[:,:,0]
encoded_u_test = process_batches(u_test.T, autoencoder, batch_size=batch_size)[:,:,0]

##########################################-.SAVE PROCESSED DATA.-########################################

np.save('files/encoded_u_train.npy', encoded_u_train)
np.save('files/encoded_u_test.npy', encoded_u_test)