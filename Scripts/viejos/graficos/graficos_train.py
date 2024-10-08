import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import os
from scipy.fft import rfft
dataset = 1
path = rf'C:\Users\Luis Quispe\Desktop\Tesis\sakura_runs\data_KS\Convolutional_dataset{dataset}'
os.chdir(path)
get_ipython().run_line_magic('matplotlib', 'qt5')
del path
del dataset
#%%
#cargo los archivos train data_sets
u = np.load('u_train.npy') 
u_hat = np.load('u_hat_train.npy')

data = np.load('data.npz')
L = data['L']
T = data['T']
t = data['t']
nx = data['nx']
k = data['k']

#%%
# Create subplots
plt.close('all')
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
seleccion_x = np.sort(np.random.choice(u.shape[1], size=10, replace=False))
for i in seleccion_x:
    axs[1, 0].plot(u[:,i])
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