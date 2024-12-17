import numpy as np
import pickle
import pandas as pd

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

##########################################-.LOAD PARAMS AND DATASETS.-########################################

version = 3
with open(f'../version_{version}/encoded_dataset.pkl', 'rb') as f:
    encoded_dataset = pickle.load(f)
encoded_dataset = np.array(encoded_dataset, dtype=object) 

with open(f"../version_{version}/train_dataset.pkl", "rb") as f:
    train_dataset = pickle.load(f)
    
u_final_train = np.array(list(zip(*train_dataset))[1])

decoded_final_states_rk4 = np.load(f'../version_{version}/decoded_final_states_rk4.npy')
encoded_final_states_rk4 = np.load(f'../version_{version}/encoded_final_states_rk4.npy')

##########################################-.TRAINING LOG.-########################################

training_log = pd.read_csv(f'../version_{version}/training_log.csv')



plt.figure(figsize=(10, 6))
plt.plot(training_log['Validation Loss'], 'r-', linewidth=3, label='Validation Loss', alpha = 0.7)
plt.plot(training_log['Training Loss'], 'b-', markersize=10, label='Training Loss', alpha = 0.6)
plt.yscale('log')


plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss (log scale)', fontsize=14)
plt.title('Training and Validation Loss Over Epochs', fontsize=16)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(loc='upper right', fontsize=12)
plt.savefig(f'./training_log_version_{version}')
plt.tight_layout()
plt.show()


##########################################-.PLOTS.-########################################

sample_idx = 6

u_tilde_final_states = decoded_final_states_rk4[sample_idx]

# Spectrum
u_hat_final_states =np.fft.rfft(u_tilde_final_states, axis=0)
u_hat_final_states_og = np.fft.rfft(u_final_train[sample_idx], axis=0)
difference = np.abs(np.fft.rfft(u_tilde_final_states, axis=0) - np.fft.rfft(u_final_train[sample_idx], axis=0))
# Energy
node_energy = np.sum(np.square(decoded_final_states_rk4), axis=1)
train_energy = np.sum(np.square(u_final_train), axis=1)

fig, axs = plt.subplots(2, 2, figsize=(20, 12))

# 1. Decoded comparison plot
axs[0,0].plot(decoded_final_states_rk4[sample_idx], 'r-', linewidth=2, label=r'$\tilde{u}(\tau)$')
axs[0,0].plot(u_final_train[sample_idx], 'b.', markersize=8, label=r'$u(\tau)$')
axs[0,0].set_title(f'Comparación NODE vs Train version_{version}', fontsize=16)
axs[0,0].set_xlabel('x', fontsize=14)
axs[0,0].set_ylabel(r'$u(x, \tau)$', fontsize=14)
axs[0,0].legend(fontsize=12, loc='best')
axs[0,0].grid(True, alpha=0.3)

# 2. Encoded comparison plot
axs[0,1].plot(encoded_final_states_rk4[sample_idx], 'r-', linewidth=2, label=r'$\tilde{h}(\tau)$')
axs[0,1].plot(encoded_dataset[sample_idx][1], 'b--', markersize=8, label=r'$h(\tau)$')
axs[0,1].set_title(f'Comparación NODE vs Train Encoded version_{version}', fontsize=16)
axs[0,1].set_xlabel('x', fontsize=14)
axs[0,1].set_ylabel(r'$h(x, 0)$', fontsize=14)
axs[0,1].legend(fontsize=12, loc='best')
axs[0,1].grid(True, alpha=0.3)

# 3. Spectrum plot
axs[1,0].loglog(np.abs(u_hat_final_states)**2, 'r-', label='NODE Spectrum')
axs[1,0].loglog(np.abs(u_hat_final_states_og)**2, 'b--', label='Original Spectrum (Train)')
axs[1,0].loglog(difference, 'g-',label=r'$|\tilde{u}(k) - u(k)|$')
axs[1,0].set_title(f'Densidad de energía version_{version}', fontsize=16)
axs[1,0].set_xlabel('k', fontsize=14)
axs[1,0].set_ylabel('Densidad de energía ($|u(k)|^2$)', fontsize=14)
axs[1,0].legend(fontsize=12, loc='best')
axs[1,0].grid(True, alpha=0.3)

# 4. Energy plot
axs[1,1].plot(node_energy, 'r-', label='NODE Energy', alpha=0.7)
axs[1,1].plot(train_energy, 'b--', label='Train dataset Energy', alpha=0.5)
axs[1,1].set_title(f'Energía total version_{version}', fontsize=16)
axs[1,1].set_xlabel('Tiempo (a.u.)', fontsize=14)
axs[1,1].set_ylabel(r'$\sum_{columnas} u^2$', fontsize=14)
axs[1,1].legend(fontsize=12, loc='best')
axs[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'./all_in_one_version_{version}.png')
plt.show()

