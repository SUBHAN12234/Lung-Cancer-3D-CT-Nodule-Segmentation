import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv3D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv3D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def encoder_block(x, filters):
    f = conv_block(x, filters)
    p = layers.MaxPool3D(pool_size=2)(f)
    return f, p

def decoder_block(x, skip, filters):
    us = layers.Conv3DTranspose(filters, 2, strides=2, padding='same')(x)
    concat = layers.Concatenate()([us, skip])
    return conv_block(concat, filters)

def build_3d_unet(input_shape=(None, 128, 128, 1)):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    s1, p1 = encoder_block(inputs, 32)
    s2, p2 = encoder_block(p1, 64)
    s3, p3 = encoder_block(p2, 128)

    # Bottleneck
    b = conv_block(p3, 256)

    # Decoder
    d1 = decoder_block(b, s3, 128)
    d2 = decoder_block(d1, s2, 64)
    d3 = decoder_block(d2, s1, 32)

    # Output
    outputs = layers.Conv3D(1, 1, activation='sigmoid')(d3)

    return models.Model(inputs, outputs)
