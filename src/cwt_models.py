import tensorflow as tf
from tensorflow.keras import layers, models, regularizers


def build_cwt_2d_cnn(
    input_shape: tuple[int, int, int] = (32, 32, 1),
    l2_reg: float = 0.0005,
    learning_rate: float = 0.001,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(
            16, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(
            32, (3, 3), padding='same', activation='relu',
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid'),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=focal_gamma) if use_focal_loss else 'binary_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
    )
    return model
