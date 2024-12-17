import json
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from cnn_functions import custom_loss, neural_configuration, build_autoencoder

# Load parameters
with open('files/architecture.json') as f:
    params = json.load(f)

# Load data
dataset = params['dataset']
path = f'../data_KS/Convolutional_dataset{dataset}/'
u_train = np.load(path + 'u_train.npy')

# Create encoder and decoder configuration
dh = params['dh']
lr = params['lr']
lista_neuronas_encoder = params['lista_neuronas_encoder']
kernel_size = params['kernel_size']
strides = params['strides']
config_encoder, config_decoder = neural_configuration(lista_neuronas_encoder, kernel_size, strides)

# Build and compile autoencoder
autoencoder = build_autoencoder(params['N'], config_encoder, config_decoder, dh, lr)
autoencoder.summary()

autoencoder.compile(optimizer=Adam(learning_rate=lr),
                    loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, params['alpha']),
                    metrics=['MeanSquaredError', lambda y_true, y_pred: custom_loss(y_true, y_pred, params['alpha'])])

csv_logger = CSVLogger('files/output.csv', append=True)
model_ckpt = ModelCheckpoint('files/{{epoch:04d}}.weights.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             save_weights_only=True,
                             save_freq='epoch')
reduce_lr = ReduceLROnPlateau(monitor='val_MeanSquaredError', factor=0.75, patience=params['patience'], min_lr=1e-6)
callbacks = [csv_logger, model_ckpt, reduce_lr]

# Train model
autoencoder.fit(u_train.T,
                u_train.T,
                epochs=50,
                batch_size=params['batch_size'],
                validation_split=0.25,
                callbacks=callbacks,
                verbose=1)
