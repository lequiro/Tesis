import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

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

def build_model(u, encoder_layers, decoder_layers, dh, lr):
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

    return autoencoder
