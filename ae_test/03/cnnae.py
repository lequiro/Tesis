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
corrida=3
dh=128
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Convolutional'
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
lr= 4
N= (u_train.T).shape[1]
num_batch_size = 1024
num_epochs = 500
num_valid_split = 0.25
x_data = u_train.T[::10,:]
x_valid = u_test.T[::10,:]

# Define the input shape
input_shape = (N,1)

# Encoder
inputs = Input(shape=input_shape)
x = Conv1D(32, 3, activation='relu', padding='same')(inputs)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
x = Reshape((8*N,))(x)
encoded = Dense(dh)(x)

# Decoder
x = Dense(8*N, activation='relu')(encoded)
x = Reshape((N, 8))(x)
x = Conv1D(8, 3, activation='relu', padding='same')(x)
x = Conv1D(16, 3, activation='relu', padding='same')(x)
x = Conv1D(32, 3, activation='relu', padding='same')(x)
decoded = Conv1D(1, 3, padding='same')(x)

# Autoencoder model
autoencoder = Model(inputs, decoded)
optimizer = Adam(learning_rate=1e-4)
autoencoder.compile(optimizer=optimizer,
                    loss='mse',
                    metrics=['MeanSquaredError', grad_loss])

# Callback
csv_logger = CSVLogger(f'prueba{corrida}\output_cnnae_{dh}_{num_batch_size}_{num_epochs}_{lr}.csv', append=True)

# Train
start_time = time.time()

history = autoencoder.fit(x_data, x_data,
                          epochs=num_epochs,
                          batch_size=num_batch_size,
                          validation_split=num_valid_split,
                          callbacks=[csv_logger],
                          verbose=0)

end_time = time.time()
training_time = (end_time - start_time)/60
print(f"Training time: {training_time} minutes")
# Save output
preds = autoencoder(x_valid).numpy()[:,:,0]
#%%
df_cnnae = pd.read_csv(f'prueba{corrida}\output_cnnae_{dh}_{num_batch_size}_{num_epochs}_{lr}.csv')
#%%%
num = 25_000

plt.figure(figsize=(10, 6))
plt.plot(preds.T[:, num], label='predicción', linewidth=2)
plt.plot(x_valid.T[:, num], label='data KS', linewidth=2)
plt.title('Convolutional Neural Network', fontsize=16)
plt.xlabel('x', fontsize=14)
plt.ylabel('u(x)', fontsize=14)
plt.legend(fontsize=12)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.savefig(f'prueba{corrida}\cnnae_{num}.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
#LOSS FUNCTION
plt.figure(figsize=(10, 6))
plt.plot(df_cnnae['MeanSquaredError'], label='MSE_TRAIN', linewidth=2)
plt.plot(df_cnnae['val_MeanSquaredError'], label='MSE_VALID', linewidth=2)
plt.title('Training and Validation Mean Squared Error CNNAE', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('..\mse_plot_cnnae.png', dpi=300, bbox_inches='tight')
plt.show()
#%%
# Difference between valid and train
plt.figure(figsize=(10, 6))
plt.semilogy(df_cnnae['epoch'], df_cnnae['val_MeanSquaredError'] - df_cnnae['MeanSquaredError'])

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Validation MSE - Training MSE', fontsize=14)
plt.title('Difference between Validation and Training MSE over Epochs', fontsize=18)

plt.xticks(fontsize=12)

# Manually set the y-ticks to include only the desired ticks
desired_yticks = [10**-2, 10**-1, 10**-0.5]
plt.yticks(desired_yticks, fontsize=12)

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

