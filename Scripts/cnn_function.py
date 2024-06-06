import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Input, Dense, Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from IPython import get_ipython
dataset = 3
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Convolutional_dataset{dataset}'
os.chdir(path)
get_ipython().run_line_magic('matplotlib', 'qt5')
del path
del dataset
#%%
u_train = np.load('u_train.npy')
u_test = np.load('u_test.npy')
#%%
# Custom loss function
@tf.function
def grad_loss(y_true, y_pred):
    dx = 1
    dy_true = tf.experimental.numpy.diff(y_true, axis=1)*dx
    dy_pred = tf.experimental.numpy.diff(y_pred, axis=1)*dx
    loss_deriv = tf.keras.losses.mse(dy_true, dy_pred)
    return loss_deriv


def neural_configuration(neuronas_encoder, kernel):
    encoder = []
    decoder = []
    neuronas_decoder = sorted(neuronas_encoder.copy())
    
    for i in neuronas_encoder:
        encoder.append((i, kernel))
    for m in neuronas_decoder:
        decoder.append((m, kernel))
    
    return encoder, decoder

def build_model(u,encoder_layers,decoder_layers,dh,lr):
    N = (u.T).shape[1]
    input_shape = (N, 1)
    # Encoder
    inputs = Input(shape=input_shape)
    x = inputs
    for filters, kernel_size in encoder_layers:
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
    x = Reshape((filters * N,))(x)
    encoded = Dense(dh, activation='relu')(x)

    # Decoder
    x = Dense(filters * N, activation='relu')(encoded)
    x = Reshape((N, filters))(x)
    for filters, kernel_size in decoder_layers:
        x = Conv1D(filters, kernel_size, activation='relu', padding='same')(x)
    decoded = Conv1D(1, 3, padding='same')(x)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)
    optimizer = Adam(learning_rate=lr)
    autoencoder.compile(optimizer=optimizer,
                        loss='mse',
                        metrics=['MeanSquaredError', grad_loss])
    return autoencoder

def train_model(u, model, num_epochs, num_batch_size, num_valid_split, prueba = 1):
    csv_logger = CSVLogger(f'prueba{prueba}\{num_epochs}_{num_batch_size}_{num_valid_split}_layers_{len(lista_neuronas_encoder)}.csv', append=False)
    start_time = time.time()
    history = model.fit(u.T, u.T,
              epochs=num_epochs,
              batch_size=num_batch_size,
              validation_split=num_valid_split,
              callbacks=[csv_logger],verbose=0)
    end_time = time.time()
    training_time = (end_time - start_time)/60
    print(f"Training time: {training_time} minutes")
    return history
#%%
dh= 64
lr=1e-4
lista_neuronas_encoder = [128,64,32,16,8]
kernel_size = 5
config_encoder, config_decoder = neural_configuration(lista_neuronas_encoder, kernel_size)

modelo1= build_model(u_train, config_encoder, config_decoder, dh, lr)
modelo_entrenado1 = train_model(u_train, modelo1, num_epochs=500, num_batch_size=1024, num_valid_split=0.25)
#%%
'''
# Save the weights
weights_file_path = 'autoencoder_weights.h5'
autoencoder.save_weights(weights_file_path)

# Later, to load the weights into a new model instance
new_autoencoder = build_model(N, dh, encoder_layers, decoder_layers, input_shape)
new_autoencoder.load_weights(weights_file_path)

# Verify by printing summaries
autoencoder.summary()
new_autoencoder.summary()
'''



