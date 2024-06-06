import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
import pandas as pd

corrida = 1
dh = 32
path = r'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Convolutional\prueba3'
os.chdir(path)
get_ipython().run_line_magic('matplotlib', 'qt5')
#%%
# Detect .csv files in the directory
csv_files = [file for file in os.listdir(path) if file.endswith('.csv')]

# Store the last MSE values
last_mse_train = []
last_mse_valid = []

# Iterate through each .csv file and extract the last MSE values
for file in csv_files:
    df = pd.read_csv(file)
    if 'MeanSquaredError' in df.columns and 'val_MeanSquaredError' in df.columns:
        last_mse_train.append(df['MeanSquaredError'].iloc[-1])
        last_mse_valid.append(df['val_MeanSquaredError'].iloc[-1])

# Plot the last MSE values
plt.figure(figsize=(10, 6))
plt.plot(last_mse_train, label='Last MSE_TRAIN', linewidth=2, marker='o')
plt.plot(last_mse_valid, label='Last MSE_VALID', linewidth=2, marker='o')
plt.title('Last Mean Squared Error for Each CSV File', fontsize=16)
plt.xlabel('File Index', fontsize=14)
plt.ylabel('Mean Squared Error', fontsize=14)
plt.legend(fontsize=12)
plt.xticks(ticks=np.arange(len(csv_files)), labels=[f'File {i+1}' for i in range(len(csv_files))], rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('..\last_mse_plot_cnnae.png', dpi=300, bbox_inches='tight')
plt.show()
