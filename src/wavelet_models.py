import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_wavelet_net(
    input_shape: tuple[int, int] = (180, 4),
    use_blstm: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    l2_reg: float = 0.0005,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)

    subband_weights = layers.Conv1D(
        filters=8,
        kernel_size=1,
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='wavelet_subband_weighting',
    )(inputs)

    x = layers.Conv1D(
        filters=32,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
    )(subband_weights)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(
        filters=64,
        kernel_size=5,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    if use_blstm:
        x = layers.Bidirectional(layers.LSTM(32, return_sequences=False))(x)
    else:
        x = layers.Flatten()(x)

    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='wavelet_net')

    if use_focal_loss:
        loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=focal_gamma)
    else:
        loss = 'binary_crossentropy'

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=['accuracy'],
    )
    return model
