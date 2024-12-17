import json
import numpy as np

import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt5')

##########################################-.LOAD PARAMS AND DATASETS.-########################################

with open('files/architecture.json') as f:
    params = json.load(f)

dataset = params['dataset']
path = f'../data_KS/Convolutional_dataset{dataset}/'
u_train = np.load(path + 'u_train.npy', mmap_mode='r')
u_test = np.load(path + 'u_test.npy', mmap_mode='r')

u_train_autoencoder = np.load('files/encoded_u_train.npy').T
u_test_autoencoder = np.load('files/encoded_u_test.npy').T

################################################-.COMPUTE MSE.-#################################################

mse_train = np.mean((u_train - u_train_autoencoder) ** 2, axis=0)
mse_test = np.mean((u_test - u_test_autoencoder) ** 2, axis=0)

min_mse_train_idx = np.argmin(mse_train)
max_mse_train_idx = np.argmax(mse_train)

min_mse_test_idx = np.argmin(mse_test)
max_mse_test_idx = np.argmax(mse_test)

print("Train dataset:")
print(f"Minimum MSE: {mse_train[min_mse_train_idx]} at snapshot {min_mse_train_idx}")
print(f"Maximum MSE: {mse_train[max_mse_train_idx]} at snapshot {max_mse_train_idx}")

print("\nTest dataset:")
print(f"Minimum MSE: {mse_test[min_mse_test_idx]} at snapshot {min_mse_test_idx}")
print(f"Maximum MSE: {mse_test[max_mse_test_idx]} at snapshot {max_mse_test_idx}")

######################################-.Plotting snapshots with minimum MSE.-#################################

fig1 = plt.figure(1, figsize=(14, 8))

plt.plot(u_train[:, min_mse_train_idx], 'g.', label='u_train (Train)', alpha=0.8)
plt.plot(u_train_autoencoder[:, min_mse_train_idx], 'g-', label='u_train_autoencoder (Train)', alpha=0.8)

plt.plot(u_test[:, min_mse_test_idx], 'm.', label='u_test (Test)', alpha=0.8)
plt.plot(u_test_autoencoder[:, min_mse_test_idx], 'm-', label='u_test_autoencoder (Test)', alpha=0.8)

plt.title('Mejor snapshot $u(x,t_i)$ Train and Test', fontsize=30)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$u(x,t_i)$', fontsize=20)

plt.xticks(size=14)
plt.yticks(size=14)
plt.grid('on')
plt.legend(fontsize=14, loc='best')

plt.savefig('validation_plots/minimum_mse_snapshots.png', dpi=300)

######################################-.Plotting snapshots with maximum MSE.-#################################

fig2 = plt.figure(2, figsize=(14, 8))

plt.plot(u_train[:, max_mse_train_idx], 'g.', label='u_train (Train)', alpha=0.8)
plt.plot(u_train_autoencoder[:, max_mse_train_idx], 'g-', label='u_train_autoencoder (Train)', alpha=0.8)

plt.plot(u_test[:, max_mse_test_idx], 'm.', label='u_test (Test)', alpha=0.8)
plt.plot(u_test_autoencoder[:, max_mse_test_idx], 'm-', label='u_test_autoencoder (Test)', alpha=0.8)

plt.title('Peor snapshot $u(x,t_i)$ Train and Test', fontsize=30)
plt.xlabel('$x$', fontsize=20)
plt.ylabel('$u(x,t_i)$', fontsize=20)

plt.xticks(size=14)
plt.yticks(size=14)
plt.grid('on')
plt.legend(fontsize=14, loc='best')

plt.savefig('validation_plots/maximum_mse_snapshots.png', dpi=300)

########################################-.Plotting the MSE distribution.-###################################

fig3 = plt.figure(3, figsize=(14, 8))

bins = np.logspace(np.log10(min(mse_train.min(), mse_test.min())), 
                   np.log10(max(mse_train.max(), mse_test.max())), 
                   50)

fig3 = plt.figure(3, figsize=(14, 8))

plt.hist(mse_train, bins=bins, color='green', alpha=0.7, label='Train MSE', linewidth=2)
plt.hist(mse_test, bins=bins, color='magenta', alpha=0.5, label='Test MSE', linewidth=2)

plt.title('Distribución del Error Cuadrático Medio (MSE)', fontsize=30)
plt.xlabel('MSE', fontsize=20)
plt.ylabel('Frecuencia', fontsize=20)

plt.xscale('log')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(visible=True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=14, loc='best')

plt.savefig('validation_plots/mse_distribution_logbin.png', dpi=300)
plt.show()

########################################-.Spectrum and difference in spectrum.-###################################

u_train_spectrum = np.fft.rfft(u_train, axis=0)
u_train_autoencoder_spectrum = np.fft.rfft(u_train_autoencoder, axis=0)
difference_train = np.abs(u_train_spectrum - u_train_autoencoder_spectrum)

u_test_spectrum = np.fft.rfft(u_test, axis=0)
u_test_autoencoder_spectrum = np.fft.rfft(u_test_autoencoder, axis=0)
difference_test = np.abs(u_test_spectrum - u_test_autoencoder_spectrum)

fig4 = plt.figure(4, figsize=(12, 8))

plt.loglog(np.abs(u_train_spectrum[:, -1]) ** 2, label='Original Spectrum (Train)', color='green', linestyle='-', linewidth=2, alpha=0.7)
plt.loglog(difference_train[:, -1], label=r'$|\tilde{u}(k) - u(k)|$ (Train)', color='green', linestyle='-.', linewidth=2)

plt.loglog(np.abs(u_test_spectrum[:, -1]) ** 2, label='Original Spectrum (Test)', color='magenta', linestyle='-', linewidth=2, alpha=0.7)
plt.loglog(difference_test[:, -1], label=r'$|\tilde{u}(k) - u(k)|$ (Test)', color='magenta', linestyle='-.', linewidth=2)

plt.title('Comparación de Espectro Train and Test', fontsize=22)
plt.xlabel('$k$', fontsize=18)
plt.ylabel('$|u(k)|^2$', fontsize=18)

plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(visible=True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
plt.legend(fontsize=16, loc='best')

plt.savefig('validation_plots/spectrum_comparison.png', dpi=300)