import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .config import DEFAULT_FS


def compute_gradcam_1d(
    model: tf.keras.Model,
    ecg_sample: np.ndarray,
    feat_sample: np.ndarray = None,
    target_layer_name: str = 'conv1d_2',
) -> np.ndarray:
    if ecg_sample.ndim == 1:
        wave_tensor = tf.convert_to_tensor(ecg_sample.reshape(1, -1, 1), dtype=tf.float32)
    elif ecg_sample.ndim == 2:
        wave_tensor = tf.convert_to_tensor(
            ecg_sample.reshape(1, ecg_sample.shape[0], ecg_sample.shape[1]),
            dtype=tf.float32,
        )
    else:
        wave_tensor = tf.convert_to_tensor(ecg_sample, dtype=tf.float32)

    is_multimodal = isinstance(model.input, list)
    if is_multimodal:
        if feat_sample is None:
            n_feats = model.input[1].shape[1]
            feat_tensor = tf.zeros((1, n_feats), dtype=tf.float32)
        else:
            feat_tensor = tf.convert_to_tensor(
                feat_sample.reshape(1, -1),
                dtype=tf.float32,
            )
        model_inputs = [wave_tensor, feat_tensor]
    else:
        model_inputs = wave_tensor

    target_layer = model.get_layer(target_layer_name)
    grad_model = tf.keras.Model(inputs=model.input, outputs=[target_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(model_inputs)
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=1)
    heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads[0], axis=-1)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    heatmap_norm = (heatmap / (max_val + 1e-10)).numpy()

    window_size = wave_tensor.shape[1]
    cam_1d = np.interp(
        np.linspace(0, 1, window_size),
        np.linspace(0, 1, len(heatmap_norm)),
        heatmap_norm,
    )
    return cam_1d


def plot_model_explanations(
    model: tf.keras.Model,
    X_wave: np.ndarray,
    y: np.ndarray,
    X_feat: np.ndarray = None,
    target_layer_name: str = 'conv1d_2',
    figsize: tuple[int, int] = (14, 4),
) -> None:
    idx_neg = np.where(y == -1)[0]
    idx_pos = np.where(y == 1)[0]

    chosen_indices = []
    if len(idx_neg) > 0:
        chosen_indices.append(idx_neg[0])
    if len(idx_pos) > 0:
        chosen_indices.append(idx_pos[0])

    if not chosen_indices:
        raise ValueError("Brak probek do wyjasnienia modelu.")

    fig, axes = plt.subplots(1, len(chosen_indices), figsize=figsize, sharey=True)
    if len(chosen_indices) == 1:
        axes = [axes]

    window_size = X_wave.shape[1]
    time_ms = np.arange(window_size) / DEFAULT_FS * 1000.0

    scatter = None
    is_multimodal = isinstance(model.input, list)

    for ax, idx in zip(axes, chosen_indices):
        wave_sample = X_wave[idx : idx + 1]
        feat_sample = X_feat[idx : idx + 1] if X_feat is not None else None
        true_lbl = y[idx]

        if is_multimodal:
            if feat_sample is None:
                n_feats = model.input[1].shape[1]
                feat_sample = np.zeros((1, n_feats), dtype=np.float32)
            pred_prob = float(model.predict([wave_sample, feat_sample], verbose=0)[0][0])
        else:
            pred_prob = float(model.predict(wave_sample, verbose=0)[0][0])

        pred_lbl = 1 if pred_prob > 0 else -1

        cam = compute_gradcam_1d(
            model,
            wave_sample,
            feat_sample=feat_sample,
            target_layer_name=target_layer_name,
        )
        signal_vals = wave_sample[0, :, 0]

        ax.plot(time_ms, signal_vals, color='black', linewidth=1.5, label='Sygnal EKG')

        scatter = ax.scatter(
            time_ms,
            signal_vals,
            c=cam,
            cmap='plasma',
            s=22,
            zorder=3,
            label='Waga Grad-CAM (istotnosc)',
        )
        ax.axvline(
            x=time_ms[window_size // 2],
            color='red',
            linestyle='--',
            alpha=0.6,
            label='Zalamek R',
        )

        lbl_map = {-1: 'Norma (-1)', 1: 'Arytmia/Stym (+1)'}
        ax.set_title(
            f"Rzeczywista: {lbl_map.get(true_lbl)} | Predykcja: {lbl_map.get(pred_lbl)} (score: {pred_prob:.2f})"
        )
        ax.set_xlabel("Czas [ms]")
        ax.set_ylabel("Z-score amplitudy")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    plt.tight_layout()
    if scatter is not None:
        fig.colorbar(scatter, ax=axes, orientation='vertical', fraction=0.02, pad=0.02, label='Aktywacja Grad-CAM')

    plt.show()
