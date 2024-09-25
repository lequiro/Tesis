import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Reshape, Flatten, Conv1DTranspose
from tensorflow.keras.models import Model

@tf.function
def grad_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, tf.shape(y_true))
    dx = 1
    dy_true = tf.experimental.numpy.diff(y_true, axis=1)*dx
    dy_pred = tf.experimental.numpy.diff(y_pred, axis=1)*dx
    loss_deriv = tf.keras.losses.mse(dy_true, dy_pred)
    return loss_deriv

# Custom loss function
@tf.function
def custom_loss(y_true, y_pred,alpha):
    y_pred = tf.reshape(y_pred, tf.shape(y_true))
    loss = tf.keras.losses.mse(y_true, y_pred)
    
    loss_deriv = grad_loss(y_true, y_pred)
    
    loss = alpha * loss + (1 - alpha) * loss_deriv
    
    return loss

def neural_configuration(neuronas_encoder, kernel, strides):
    encoder = []
    decoder = []
    neuronas_decoder = sorted(neuronas_encoder.copy())
    strides_decoder = sorted(strides.copy())
    for i in range(len(neuronas_encoder)):
        encoder.append((neuronas_encoder[i], kernel[i], strides[i]))
    for m in range(1,len(neuronas_decoder)):
        decoder.append((neuronas_decoder[m], kernel[m], strides_decoder[m-1]))

    decoder.append((neuronas_decoder[-1],kernel[0],strides_decoder[-1]))
    return encoder, decoder

def build_model(N, encoder_layers, decoder_layers, dh, lr):
    input_shape = (N, 1)

    # Encoder
    inputs = Input(shape=input_shape)
    x = inputs
    for filters, kernel_size, stride in encoder_layers:
        x = Conv1D(filters, kernel_size, activation='relu', padding='same', strides=stride)(x)
    encoded_shape = x.shape
    x = Flatten()(x)
    
    encoded = Dense(dh)(x)

    # Decoder
    x = Dense(encoded_shape[1]*encoded_shape[2], activation='relu')(encoded)
    x = Reshape((encoded_shape[1], encoded_shape[2]))(x)
    x = Conv1DTranspose(encoded_shape[2], decoder_layers[0][1], activation='relu', padding='same', name = 'jeje', strides= 1)(x)
    for filters, kernel_size, stride in decoder_layers:
        x = Conv1DTranspose(filters, kernel_size, activation='relu', padding='same', strides=stride)(x)
    decoded = Conv1DTranspose(1, 3, padding='same', name='decoded')(x)

    # Autoencoder model
    autoencoder = Model(inputs, decoded)

    return autoencoder