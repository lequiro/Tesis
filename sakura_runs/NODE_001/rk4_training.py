import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from rk4_functions import RK4Model

version = 1
json_files = sorted([f for f in os.listdir('../autoencoder/files') if f.endswith('.json')])

with open(f'../autoencoder/files/{json_files[0]}') as f:
    params = json.load(f)


base_model = Sequential([
    Input(shape=(8,)),
    Dense(16, activation='tanh'),
    Dense(16, activation='tanh'),
    Dense(8)
])

rk4_model = RK4Model(base_model)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()
patience = 1
factor = 0.5
min_lr = 1e-6
best_loss = float('inf')
wait = 0
rk4_model.summary()


with open(f'version_{version}/encoded_dataset.pkl', "rb") as f:
    version_loaded = pickle.load(f)

dt = 1e-4
batch_size = 64
num_epochs = 10

def data_generator():
    for initial_condition, final_condition, num_steps in version_loaded:
        yield tf.cast(initial_condition, tf.float32), tf.cast(final_condition, tf.float32), num_steps

output_signature = (
    tf.TensorSpec(shape=(8,), dtype=tf.float32),      # Initial condition
    tf.TensorSpec(shape=(8,), dtype=tf.float32),      # Final condition
    tf.TensorSpec(shape=(), dtype=tf.int32)           # Number of steps
)

dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=output_signature
).shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    epoch_loss = 0.0
    num_batches = 0

    for batch_num, batch in enumerate(iter(dataset)):
        
        initial_conditions_batch, final_conditions_batch, num_steps_batch = batch
        with tf.GradientTape() as tape:
            predicted_final_states = rk4_model.forward_rk4_nn(initial_conditions_batch, num_steps_batch, dt)
            loss = loss_fn(final_conditions_batch, predicted_final_states)

        gradients = tape.gradient(loss, rk4_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rk4_model.trainable_variables))

        epoch_loss += loss.numpy()
        num_batches += 1
        


        if batch_num % 100 == 0:
            print(f"Batch {batch_num}, Loss: {loss.numpy()}, Batch Size: {initial_conditions_batch.shape[0]}")

            batch_weights_save_path = f'version_{version}/rk4_model_epoch{epoch + 1}_batch{batch_num + 1}.weights.h5'
            rk4_model.save_weights(batch_weights_save_path)
            print(f"Weights saved at: {batch_weights_save_path}")
            
    epoch_loss /= num_batches
    print(f"Epoch {epoch + 1} Loss: {epoch_loss}")

    # Learning rate scheduler
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            new_lr = max(optimizer.learning_rate.numpy() * factor, min_lr)
            optimizer.learning_rate.assign(new_lr)
            print(f"Reducing learning rate to {new_lr}")
            wait = 0

weights_save_path = os.path.join(f'version_{version}/rk4_model.weights.h5')
rk4_model.save_weights(weights_save_path)
print(f"Weights saved at: {weights_save_path}")
