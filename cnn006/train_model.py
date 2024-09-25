import json
import numpy as np
import os
import shutil
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from cnn_functions import custom_loss, neural_configuration, build_model

# Directory containing JSON parameter files
params_dir = 'files'

# Get the list of JSON files and sort them
json_files = sorted([f for f in os.listdir(params_dir) if f.endswith('.json')])

# Iterate over each sorted JSON file in the directory
for param_file in json_files:
    # Load parameters
    with open(os.path.join(params_dir, param_file)) as f:
        params = json.load(f)
        
    # Create directory
    directory_name = param_file.split('.')[0]
    directory_path = os.path.join(f'{params_dir}', f'{directory_name}')
    os.makedirs(directory_path, exist_ok=True)
    shutil.move(os.path.join(params_dir, param_file), os.path.join(params_dir, directory_name, param_file))
    
    # Load data
    dataset = params['dataset']
    path = f'../data_KS/Convolutional_dataset{dataset}/'
    u_train = np.load(path + 'u_train.npy')

    # Create and compile model
    dh = params['dh']
    lr = params['lr']
    lista_neuronas_encoder = params['lista_neuronas_encoder']
    kernel_size = params['kernel_size']
    strides = params['strides']
    config_encoder, config_decoder = neural_configuration(lista_neuronas_encoder, kernel_size, strides)

    autoencoder = build_model(params['N'], config_encoder, config_decoder, dh, lr)
    autoencoder.summary()

    autoencoder.compile(optimizer=Adam(learning_rate=lr),
                        loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, params['alpha']),
                        metrics=['MeanSquaredError', lambda y_true, y_pred: custom_loss(y_true, y_pred, params['alpha'])])

    # autoencoder.compile(optimizer=Adam(learning_rate=lr),
    #                     loss= custom_loss(params['alpha']),
    #                     metrics=['MeanSquaredError', custom_loss(params['alpha'])])

    # Prepare run directory and callbacks
    run_dir = os.path.join(f'{directory_path}')
    os.makedirs(run_dir, exist_ok=True)

    csv_logger = CSVLogger(f'{run_dir}/output.csv', append=True)
    model_ckpt = ModelCheckpoint(f'{run_dir}/{{epoch:04d}}.weights.h5',
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 save_freq = 'epoch')
    reduce_lr = ReduceLROnPlateau(monitor='val_MeanSquaredError', factor=0.75 , patience=params['patience'], min_lr=1e-6)
    callbacks = [csv_logger,model_ckpt,reduce_lr]
    # Train model
    autoencoder.fit(u_train.T,
                    u_train.T,
                    epochs= 300,
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    callbacks=callbacks,
                    verbose=1)

