import os
import json
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
import csv
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

def data_generator(data):
    for initial_condition, final_condition, num_steps in data:
        yield tf.cast(initial_condition, tf.float32), tf.cast(final_condition, tf.float32), num_steps

output_signature = (
    tf.TensorSpec(shape=(8,), dtype=tf.float32),   # Initial condition
    tf.TensorSpec(shape=(8,), dtype=tf.float32),   # Final condition
    tf.TensorSpec(shape=(), dtype=tf.int32)        # Number of steps
)

train_data, val_data = train_test_split(list(version_loaded), test_size=0.2, random_state=42)

# Prepare datasets
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_data),
    output_signature=output_signature
).shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_data),
    output_signature=output_signature
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Create directory for saving weights if it doesn't exist
weights_dir = f'version_{version}'
os.makedirs(weights_dir, exist_ok=True)

# Prepare CSV logger
csv_file_path = f'{weights_dir}/training_log.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Epoch', 'Training Loss', 'Validation Loss'])

global_batch_counter = 0

# Training loop with validation
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0

    # Training
    for batch in train_dataset:
        global_batch_counter += 1
        initial_conditions_batch, final_conditions_batch, num_steps_batch = batch

        with tf.GradientTape() as tape:
            predicted_final_states = rk4_model.forward_rk4_nn(initial_conditions_batch, num_steps_batch, dt)
            loss = loss_fn(final_conditions_batch, predicted_final_states)

        gradients = tape.gradient(loss, rk4_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, rk4_model.trainable_variables))

        epoch_loss += loss.numpy()
        num_batches += 1

        # Save weights every 100 batches
        if global_batch_counter % 100 == 0:
            batch_weights_save_path = os.path.join(weights_dir, f'rk4_model_batch_{global_batch_counter}.weights.h5')
            rk4_model.save_weights(batch_weights_save_path)
            print(f"Saved weights at batch {global_batch_counter} to {batch_weights_save_path}")

    epoch_loss /= num_batches

    # Validation
    val_loss = 0.0
    val_batches = 0
    for batch in val_dataset:
        initial_conditions_batch, final_conditions_batch, num_steps_batch = batch
        predicted_final_states = rk4_model.forward_rk4_nn(initial_conditions_batch, num_steps_batch, dt)
        loss = loss_fn(final_conditions_batch, predicted_final_states)
        val_loss += loss.numpy()
        val_batches += 1

    val_loss /= val_batches

    # Print combined training and validation loss in one line
    print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.6f}, Validation Loss: {val_loss:.6f}")

    # Log losses to CSV
    with open(csv_file_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([epoch + 1, epoch_loss, val_loss])

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

# Save the final model weights
final_weights_save_path = os.path.join(weights_dir, 'rk4_model_final.weights.h5')
rk4_model.save_weights(final_weights_save_path)
print(f"Weights saved at: {final_weights_save_path}")
