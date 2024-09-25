import json
import os
import numpy as np
import pandas as pd
from cnn_functions import neural_configuration, build_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

param_dirs = sorted([d for d in os.listdir('files') if os.path.isdir(os.path.join('files', d))])

id_graficos_min_total = []
id_graficos_max_total = []
mse_values_total = []
for dir_i in param_dirs:
    param_dir = dir_i
    param_file = os.path.join('files', param_dir, f'{param_dir}.json')
    
    with open(param_file) as f:
        params = json.load(f)
    
    dataset = params['dataset']
    path = f'../data_KS/Convolutional_dataset{dataset}/'
    u_train = np.load(path+'u_train.npy', mmap_mode='r')
    u_test = np.load(path+'u_test.npy', mmap_mode='r')
    
    dh = params['dh']
    lr = params['lr']
    lista_neuronas_encoder = params['lista_neuronas_encoder']
    kernel_size = params['kernel_size']
    strides = params['strides']
    config_encoder, config_decoder = neural_configuration(lista_neuronas_encoder, kernel_size, strides)
    
    #######################################  Load model ###############################################
    autoencoder = build_model(params['N'], config_encoder, config_decoder, dh, lr)
    h5_directory = os.path.join('files', param_dir)
    h5_files = [file for file in os.listdir(h5_directory) if file.endswith('.h5')]
    weights_file = os.path.join('files', param_dir, f'{h5_files[-1]}')
    autoencoder.load_weights(weights_file)
    
    mse_values = []
    for graph_idx in range(1, u_test.shape[1] + 1, 500):
        u_pred = autoencoder(u_test.T[graph_idx-1:graph_idx, :]).numpy()[0, :, 0]
        mse = mean_squared_error(u_test.T[graph_idx, :], u_pred)
        mse_values.append(mse)
        
    mse_values = np.array(mse_values)
    mse_values_total.append(mse_values)
    
    graph_idx_min = np.argmax(mse_values)*500
    id_graficos_min_total.append(graph_idx_min)
    graph_idx_max = np.argmin(mse_values)*500
    id_graficos_max_total.append(graph_idx_max)
    
    average_mse = np.mean(mse_values)
    std_mse = np.std(mse_values)

    print(f"Average MSE: {average_mse}")
    print(f"Standard Deviation of MSE: {std_mse}")
    print('/n')
    
    output_csv_path = os.path.join('files', param_dir, 'output.csv')
    output_data = pd.read_csv(output_csv_path)
    val_mse = output_data['val_MeanSquaredError']
    mse = output_data['MeanSquaredError']
    
    fig3 = plt.figure(3)
    plt.plot(val_mse, label=f'val_MeanSquaredError {dir_i}')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (log scale)')
    plt.legend()
    plt.title('Validation and Training Mean Squared Error (Log Scale)')
      
    fig2 = plt.figure(2)
    plt.title('Plots with min MSE')
    u_pred = autoencoder(u_test.T[graph_idx_min-1:graph_idx_min, :]).numpy()[0, :, 0]
    plt.plot(u_pred, label=f'Predicted {dir_i}')
    plt.plot(u_test.T[graph_idx_min, :], '--', label=f'Actual {dir_i}', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('$u(x)$')
    plt.legend()
         
    fig4 = plt.figure(4)
    plt.title('Plots with Max MSE')
    u_pred = autoencoder(u_test.T[graph_idx_max-1:graph_idx_max, :]).numpy()[0, :, 0]
    plt.plot(u_pred, label=f'Predicted {dir_i}')
    plt.plot(u_test.T[graph_idx_max, :], '--', label=f'Actual {dir_i}', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('$u(x)$')
    plt.legend()
