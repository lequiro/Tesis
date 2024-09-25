from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
dataset = 3
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\data_KS\Convolutional_dataset{dataset}'
os.chdir(path)
get_ipython().run_line_magic('matplotlib', 'qt5')
del path
del dataset
#%%
prueba = 1
v = 3
df_cnnae = pd.read_csv(f'prueba{prueba}/v{v}.csv')



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


plt.figure(figsize=(10, 6))
plt.semilogy(df_cnnae['epoch'], df_cnnae['val_MeanSquaredError'])
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Validation MSE', fontsize=14)
plt.title('Validation MSE over Epochs', fontsize=18)
plt.xticks(fontsize=12)
# desired_yticks = [10**-2, 10**-1, 10**-0.5]
# plt.yticks(desired_yticks, fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()
#%%
#cargo los archivos train data_sets
u_test = np.load('u_test.npy')
u_hat_test = np.load('u_hat_test.npy')

u_preds = np.load('u_test.npy')
u_hat_preds = np.load('u_hat_test.npy')

data = np.load('data.npz')
L = data['L']
T = data['T']
nx = data['nx']
k = data['k']
#%%
u= u_test.copy()
u_hat = u_hat_test.copy()
#%%
# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot first graph
im = axs[0, 0].imshow(u, extent=[0, T, 0, L], aspect='auto', origin='lower')
axs[0, 0].set_title('u')
axs[0, 0].set_xlabel('Tiempo')
axs[0, 0].set_ylabel('Espacio')
fig.colorbar(im, ax=axs[0, 0])

# Plot second graph
axs[0, 1].loglog(k, np.abs(u_hat[:, -1]))
axs[0, 1].set_xlabel('k')
axs[0, 1].set_ylabel('Densidad de energía $|u(k)|$')
axs[0, 1].set_title('Densidad de energía')

# Plot third graph
seleccion_x = np.sort(np.random.choice(nx, size=10, replace=False))
for i in seleccion_x:
    axs[1, 0].plot(u[i, :])
axs[1, 0].set_xlabel('Tiempo')
axs[1, 0].set_ylabel('Amplitud')
axs[1, 0].set_title('20 selecciones aleatorias de u')

# Plot fourth graph
real_space = np.sum(np.square(u), axis=0)
axs[1, 1].plot(real_space)
axs[1, 1].set_xlabel('Tiempo')
axs[1, 1].set_ylabel('Energía total $\sum_{columnas} u^2$')
axs[1, 1].set_title('Energía total')

plt.tight_layout()
plt.show()
#%%

plt.loglog(k, np.abs(u_hat[:, 2]), 'b', label='Valid t=2e-3')
plt.loglog(k, np.abs(u_hat[:, 1000]), 'b', label='Valid t=1')
plt.loglog(k, np.abs(u_hat[:, 2000]), 'b', label='Valid t=2')

# Plot predictions (in red)
plt.loglog(k, np.abs(u_hat_preds[:, 2]), 'r--', label='Preds t=2e-3')
plt.loglog(k, np.abs(u_hat_preds[:, 1000]), 'r--', label='Preds t=1')
plt.loglog(k, np.abs(u_hat_preds[:, 2000]), 'r--', label='Preds t=2')

plt.xlabel('k', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.title('Valid Data vs Predictions', fontsize=16)
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()