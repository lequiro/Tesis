''' AE test 01

Plain multilayer AE with MSE loss
'''
#%%
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
import os
from IPython import get_ipython

corrida=1
dh=32
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Datasets\dh_{dh}\{corrida}\variables'
os.chdir(path)
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
#%%
N= (u_train.T).shape[1]
num_batch_size = 128
num_epochs = 500
num_valid_split = 0.3
x_data = u_train.T[::10,:]
x_valid = u_test.T[::10,:]

# Define the input shape
input_shape = (N,)

# Encoder
inputs = Input(shape=input_shape)
x = Dense(128, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
encoded = Dense(dh)(x) # este es el dh

# Decoder
x = Dense(64, activation='relu')(encoded)
x = Dense(128, activation='relu')(x)
decoded = Dense(N)(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)
optimizer = Adam(learning_rate=1e-4)
autoencoder.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['MeanSquaredError', grad_loss])

# Callback
csv_logger = CSVLogger(rf'.\output_{num_batch_size}_{num_epochs}.csv', append=True)

# Train
history = autoencoder.fit(x_data, x_data,
                          epochs= num_epochs,
                          batch_size= num_batch_size,
                          validation_split=num_valid_split,
                          callbacks=[csv_logger],
                          verbose=0)

# Save output
preds = autoencoder(x_valid).numpy()
#%%
df = pd.read_csv(f'..\output_{num_batch_size}_{num_epochs}.csv')
#%%%
num = 20_000

plt.figure(figsize=(10, 6))
plt.plot(preds.T[:, num], label='predicción', linewidth=2)
plt.plot(x_valid.T[:, num], label='data KS', linewidth=2)
plt.title('Multi-layer autoencoder', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('u(x)', fontsize=14)
plt.legend(fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(f'..\mlae_{num}.png', dpi=300, bbox_inches='tight')
#%%
#LOSS FUNCTION
plt.figure(figsize=(10, 6))
plt.plot(df['MeanSquaredError'], label='MSE_TRAIN', linewidth=2)
plt.plot(df['val_MeanSquaredError'], label='MSE_VALID', linewidth=2)
plt.title('Training and Validation Mean Squared Error MLAE', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('..\mse_plot_mlae.png', dpi=300, bbox_inches='tight')
plt.show()
