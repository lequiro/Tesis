''' Analysis script '''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

for which in [1, 2, 3]:

    plt.figure(1)
    df = pd.read_csv(f'{which:02}/output.csv')
    plt.semilogy(df['val_mean_squared_error'], label=which)

    plt.figure(10)
    plt.semilogy(df['val_grad_loss'], label=which)
    
    plt.figure(2)
    u = np.load(f'{which:02}/preds00.npy')
    plt.plot(u, label=which)

    try:
        plt.figure(3)
        u = np.load(f'{which:02}/preds12.npy')
        plt.plot(u, label=which)
    except FileNotFoundError:
        pass

plt.figure(1)
plt.legend()

plt.figure(10)
plt.legend()

plt.figure(2)
plt.legend()

plt.figure(3)
plt.legend()

plt.show()
