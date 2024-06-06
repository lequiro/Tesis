import numpy as np
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')
#%%
u_train = np.load('u_train.npy')
u_hat_train = np.load('u_hat_train.npy')
u_test = np.load('u_test.npy')
u_hat_test = np.load('u_hat_test.npy')
#%%
# Custom loss function
@tf.function
def grad_loss(y_true, y_pred):
    dx = 1
    dy_true = tf.experimental.numpy.diff(y_true, axis=1)*dx
    dy_pred = tf.experimental.numpy.diff(y_pred, axis=1)*dx
    loss_deriv = tf.keras.losses.mse(dy_true, dy_pred)
    return loss_deriv


def build_and_train_autoencoder(u_train, u_test, dh, num_batch_size, num_epochs, lr, encoder_layers, decoder_layers):
    N = (u_train.T).shape[1]


    x_data = u_train.T
    x_valid = u_test.T

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

    # Train
    start_time = time.time()

    history = autoencoder.fit(x_data, x_data,
                              epochs=num_epochs,
                              batch_size=num_batch_size,
                              validation_split=0.25,
                              verbose=0)

    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"Training time: {training_time} minutes")

    # Minimum validation MSE and the epoch
    val_mse = history.history['val_mean_squared_error']
    min_val_mse = min(val_mse)
    min_val_mse_epoch = val_mse.index(min_val_mse) + 1  # Epochs are 1-based

    return autoencoder, history, min_val_mse, min_val_mse_epoch
#%%
encoder_layers = [(32, 3), (16, 3), (8, 3)]  # Define encoder layers as (filters, kernel_size) tuples
decoder_layers = [(8, 3), (16, 3), (32, 3)]  # Define decoder layers similarly
autoencoder, history, min_val_mse, min_val_mse_epoch = build_and_train_autoencoder(u_train, u_test, dh=64, num_batch_size=1024, num_epochs=500, lr=1e-4, encoder_layers=encoder_layers, decoder_layers=decoder_layers)

