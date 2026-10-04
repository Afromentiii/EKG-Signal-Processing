import os
import tensorflow as tf


def setup_accelerator_and_memory() -> dict:
    gpus = tf.config.list_physical_devices('GPU')
    info = {
        'gpu_available': len(gpus) > 0,
        'gpu_count': len(gpus),
        'device_type': 'GPU' if len(gpus) > 0 else 'CPU',
        'mixed_precision': False,
    }

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            info['mixed_precision'] = True
        except RuntimeError as exc:
            print(f"Konfiguracja GPU nie powiodla sie: {exc}")
    else:
        # Optymalizacja wielowatkowosci procesora przy braku GPU w srodowisku TF
        num_cores = os.cpu_count() or 4
        tf.config.threading.set_intra_op_parallelism_threads(num_cores)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    return info
