import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from .config import DEFAULT_L2_REG, DEFAULT_WINDOW_SIZE


def build_csnet_binary_model(
    window_size: int = DEFAULT_WINDOW_SIZE,
    l2_reg: float = DEFAULT_L2_REG,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    inputs = Input(shape=(window_size, 1), name='ecg_input')

    x = Conv1D(filters=32, kernel_size=5, activation='relu', name='conv1d_1')(inputs)
    x = BatchNormalization(name='bn_1')(x)
    x = MaxPooling1D(pool_size=2, name='pool_1')(x)

    x = Conv1D(filters=64, kernel_size=5, activation='relu', name='conv1d_2')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = MaxPooling1D(pool_size=2, name='pool_2')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(128, activation='relu', name='dense_features')(x)
    x = Dropout(0.5, name='dropout')(x)

    outputs = Dense(1, kernel_regularizer=l2(l2_reg), activation='linear', name='svm_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='csnet_binary')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='squared_hinge', metrics=['accuracy'])
    return model


def build_hybrid_feature_csnet(
    window_size: int = DEFAULT_WINDOW_SIZE,
    n_features: int = 13,
    l2_reg: float = DEFAULT_L2_REG,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    input_wave = Input(shape=(window_size, 1), name='waveform_input')
    x1 = Conv1D(32, 5, activation='relu', name='conv1d_1')(input_wave)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Conv1D(64, 5, activation='relu', name='conv1d_2')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(128, activation='relu')(x1)
    x1 = Dropout(0.4)(x1)

    input_feat = Input(shape=(n_features,), name='feature_input')
    x2 = Dense(32, activation='relu')(input_feat)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(0.2)(x2)

    merged = Concatenate()([x1, x2])
    merged = Dense(64, activation='relu')(merged)
    merged = Dropout(0.3)(merged)

    output = Dense(1, kernel_regularizer=l2(l2_reg), activation='linear', name='svm_output')(merged)

    model = Model(inputs=[input_wave, input_feat], outputs=output, name='hybrid_feature_csnet')
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='squared_hinge', metrics=['accuracy'])
    return model
