import json
import os
import pickle
import numpy as np
from cnn_functions import neural_configuration, build_encoder, build_autoencoder

##########################################-.LOAD PARAMS AND DATASETS.-########################################

json_files = sorted([f for f in os.listdir('../../autoencoder/files') if f.endswith('.json')])

with open(f'../../autoencoder/files/{json_files[0]}') as f:
    params = json.load(f)

path = f'dataset_{params['dataset']}'

with open("train_dataset.pkl", "rb") as f:
    dataset_loaded = pickle.load(f)

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

#############################################-.BUILD MODELS.-###########################################

input_shape = (params['N'], 1)
encoder, encoded_shape = build_encoder(input_shape, config_encoder, params['dh'])

encoder.set_weights(autoencoder.get_layer('encoder').get_weights())
encoder.summary()

#############################################-.PROCESS DATASET.-###########################################

def process_batches(data, model, batch_size):
    processed_data = []
    num_batches = int(np.ceil(data.shape[0] / batch_size))
    
    for i in range(num_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        processed_batch = model.predict(batch)
        processed_data.append(processed_batch)
        
    return np.concatenate(processed_data, axis=0)

batch_size = 2048
encoded_initial_conditions = process_batches(initial_conditions,encoder,batch_size)
encoded_u_final = process_batches(u_final,encoder,batch_size)

dataset_encoded = list(zip(encoded_initial_conditions, encoded_u_final, np.array(times)))

with open('encoded_dataset.pkl', 'wb') as f:
    pickle.dump(dataset_encoded, f)