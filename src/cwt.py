import cv2
import matplotlib.pyplot as plt
import numpy as np
import pywt
from .preprocessing import extract_heartbeats


def compute_cwt_scalogram(
    segment: np.ndarray,
    wavelet: str = 'morl',
    scales: np.ndarray = None,
    target_size: tuple[int, int] = (32, 32),
) -> np.ndarray:
    if scales is None:
        scales = np.arange(1, 33)

    coefs, _ = pywt.cwt(segment, scales, wavelet)
    scalogram = np.abs(coefs)

    resized = cv2.resize(scalogram, target_size, interpolation=cv2.INTER_AREA)
    norm = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
    return norm.astype(np.float32)


def build_cwt_dataset_with_groups(
    record_ids: list[str],
    wavelet: str = 'morl',
    scales: np.ndarray = None,
    target_size: tuple[int, int] = (32, 32),
    window_size: int = 180,
    normalize_method: str = 'record_zscore',
    max_beats_per_record: int = None,
    pn_dir: str = 'mitdb',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if scales is None:
        scales = np.arange(1, 33)

    X_list, y_list, group_list = [], [], []

    for rec in record_ids:
        beats, y_arr = extract_heartbeats(
            rec,
            window_size=window_size,
            normalize_method=normalize_method,
            pn_dir=pn_dir,
        )
        if len(y_arr) == 0:
            continue

        if max_beats_per_record is not None and len(y_arr) > max_beats_per_record:
            pos_idx = np.where(y_arr == 1)[0]
            neg_idx = np.where(y_arr == -1)[0]
            n_pos = min(len(pos_idx), max_beats_per_record // 2)
            n_neg = min(len(neg_idx), max_beats_per_record - n_pos)
            sel_pos = np.random.choice(pos_idx, n_pos, replace=False) if len(pos_idx) > 0 else np.array([], dtype=int)
            sel_neg = np.random.choice(neg_idx, n_neg, replace=False) if len(neg_idx) > 0 else np.array([], dtype=int)
            sel_idx = np.sort(np.concatenate([sel_pos, sel_neg]).astype(int))
            beats = beats[sel_idx]
            y_arr = y_arr[sel_idx]

        for beat in beats:
            scalo = compute_cwt_scalogram(
                beat, wavelet=wavelet, scales=scales, target_size=target_size
            )
            X_list.append(scalo)

        y_list.extend(y_arr)
        group_list.extend([str(rec)] * len(y_arr))

    X = np.expand_dims(np.array(X_list, dtype=np.float32), -1)
    y = np.array(y_list, dtype=int)
    groups = np.array(group_list)
    return X, y, groups


def plot_cwt_comparison(
    record_id: str = '106',
    wavelets: tuple[str, str] = ('morl', 'mexh'),
    scales: np.ndarray = None,
    target_size: tuple[int, int] = (64, 64),
    window_size: int = 180,
    pn_dir: str = 'mitdb',
) -> None:
    if scales is None:
        scales = np.arange(1, 41)

    beats, y = extract_heartbeats(
        record_id, window_size=window_size, normalize_method='record_zscore', pn_dir=pn_dir
    )
    norm_idx = np.where(y == -1)[0][0]
    arr_idx = np.where(y == 1)[0][0]

    norm_beat = beats[norm_idx]
    arr_beat = beats[arr_idx]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8))

    axes[0, 0].plot(norm_beat, color='steelblue', lw=1.5)
    axes[0, 0].set_title(f'Normal Beat (-1) - Record {record_id}')
    axes[0, 0].set_xlabel('Sample')
    axes[0, 0].set_ylabel('Amplitude (Z-score)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(arr_beat, color='crimson', lw=1.5)
    axes[0, 1].set_title(f'Arrhythmia Beat (+1) - Record {record_id}')
    axes[0, 1].set_xlabel('Sample')
    axes[0, 1].set_ylabel('Amplitude (Z-score)')
    axes[0, 1].grid(True, alpha=0.3)

    for col, (beat, title_suffix) in enumerate([(norm_beat, 'Normal'), (arr_beat, 'Arrhythmia')]):
        for row, w in enumerate(wavelets, start=1):
            coefs, _ = pywt.cwt(beat, scales, w)
            scalo = np.abs(coefs)
            resized = cv2.resize(scalo, target_size)
            norm_scalo = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
            im = axes[row, col].imshow(norm_scalo, cmap='viridis', aspect='auto', origin='lower')
            axes[row, col].set_title(f'CWT: {w.upper()} Scalogram ({title_suffix})')
            axes[row, col].set_xlabel('Time (normalized)')
            axes[row, col].set_ylabel('Scale index')
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
