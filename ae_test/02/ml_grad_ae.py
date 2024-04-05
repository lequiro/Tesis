''' AE test 02

Plain multilayer AE with MSE loss plus gradient loss
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger

def gaussian(x, x0):
    return np.exp(-1000*np.abs(x-x0)**2)

# Custom loss function
@tf.function
def grad_loss(y_true, y_pred):
    dx = 1
    dy_true = tf.experimental.numpy.diff(y_true, axis=1)*dx
    dy_pred = tf.experimental.numpy.diff(y_pred, axis=1)*dx
    loss_deriv = tf.keras.losses.mse(dy_true, dy_pred)
    return loss_deriv

# Custom loss function
@tf.function
def custom_loss(y_true, y_pred):
    # Original loss
    loss = tf.keras.losses.mse(y_true, y_pred)
    
    # Add the derivative term
    loss_deriv = grad_loss(y_true, y_pred)
    
    # Adjust the weights of the terms according to your needs
    alpha = 0.5
    loss = alpha * loss + (1 - alpha) * loss_deriv
    
    return loss

N = 200
x_dom = np.linspace(0, 1, num=N)

x_data = np.array([gaussian(x_dom, rr) for rr in np.random.rand(500)])
x_valid = np.array([gaussian(x_dom, rr) for rr in np.linspace(0,1,num=10)])

# Define the input shape
input_shape = (N,)

# Encoder
inputs = Input(shape=input_shape)
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
encoded = Dense(1)(x)

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(64, activation='relu')(x)
decoded = Dense(N)(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)
optimizer = Adam(learning_rate=1e-3)
autoencoder.compile(optimizer=optimizer,
                    loss=custom_loss,
                    metrics=['MeanSquaredError', grad_loss])

# Callback
csv_logger = CSVLogger('output.csv', append=True)

# Train
for ii in range(30):
    history = autoencoder.fit(x_data, x_data,
                              epochs=1000,
                              batch_size=32,
                              validation_split=0.2,
                              callbacks=[csv_logger],
                              verbose=0)


    # Save output
    preds = autoencoder(x_valid).numpy()
    np.save(f'preds{ii:02}', preds[1, :])

    # Change learning rate
    if ii==10:
        optimizer.learning_rate.assign(1e-4)
