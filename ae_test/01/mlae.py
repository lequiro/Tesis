''' AE test 01

Plain multilayer AE with MSE loss
'''
#%%

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import os

mes = '04'
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\{mes}_24'
os.chdir(path)
#%%
# Custom loss function

@tf.function
def grad_loss(y_true, y_pred):
    dx = 1
    dy_true = tf.experimental.numpy.diff(y_true, axis=1)*dx
    dy_pred = tf.experimental.numpy.diff(y_pred, axis=1)*dx
    loss_deriv = tf.keras.losses.mse(dy_true, dy_pred)
    return loss_deriv
#%%
u_total = np.load("u_total.npy")
u_hat_total = np.load("u_hat_total.npy")

concatenated_slices = [u_total[:,:,i] for i in range(u_total.shape[2])]
u_total_new = (np.concatenate(concatenated_slices, axis=1)).T

#%%

path = r'C:\Users\Luis Quispe\Documents\GitHub\Tesis\ae_test\01'
os.chdir(path)
#%%
N= u_total_new.shape[1]
x_data = u_total_new[::10,:]
x_valid = u_total_new[1::10,:]

# Define the input shape
input_shape = (N,)

# Encoder
inputs = Input(shape=input_shape)
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
encoded = Dense(10)(x) # este es el dh

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(N)(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)
optimizer = Adam(learning_rate=1e-3)
autoencoder.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['MeanSquaredError', grad_loss])

# Callback
csv_logger = CSVLogger('output.csv', append=True)

# Train
for ii in tqdm(range(30)):
    history = autoencoder.fit(x_data, x_data,
                              epochs=1000,
                              batch_size=8_192,
                              validation_split=0.2,
                              callbacks=[csv_logger],
                              verbose=0)


    # Save output
    preds = autoencoder(x_valid).numpy()
    np.save(f'preds{ii:02}', preds[1, :])

    # Change learning rate
    if ii==10:
        optimizer.learning_rate.assign(1e-4)


#%%