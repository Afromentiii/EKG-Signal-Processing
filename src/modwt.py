import matplotlib.pyplot as plt
import numpy as np
import pywt

from .preprocessing import extract_heartbeats


def compute_modwt_subbands(
    segment: np.ndarray,
    wavelet: str = 'db4',
    level: int = 3,
) -> np.ndarray:
    length = len(segment)
    target_length = int(np.ceil(length / (2 ** level))) * (2 ** level)
    pad_len = target_length - length
    pad_left = pad_len // 2
    pad_right = pad_len - pad_left

    padded = np.pad(segment, (pad_left, pad_right), mode='edge')
    coeffs = pywt.swt(padded, wavelet, level=level)

    subbands = [coeffs[0][0]]
    for _, detail in coeffs:
        subbands.append(detail)

    stacked = np.stack(subbands, axis=-1)
    if pad_len > 0:
        stacked = stacked[pad_left:pad_left + length, :]

    return stacked.astype(np.float32)


def build_modwt_dataset_with_groups(
    record_ids: list[str],
    wavelet: str = 'db4',
    level: int = 3,
    window_size: int = 180,
    normalize_method: str = 'record_zscore',
    max_beats_per_record: int = None,
    pn_dir: str = 'mitdb',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_list, y_list, group_list = [], [], []

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
            sub = compute_modwt_subbands(beat, wavelet=wavelet, level=level)
            x_list.append(sub)

        y_list.extend(y_arr)
        group_list.extend([str(rec)] * len(y_arr))

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    groups = np.array(group_list)
    return x, y, groups


def plot_modwt_subbands(
    record_id: str = '106',
    wavelet: str = 'db4',
    level: int = 3,
    window_size: int = 180,
    pn_dir: str = 'mitdb',
) -> None:
    beats, y = extract_heartbeats(
        record_id, window_size=window_size, normalize_method='record_zscore', pn_dir=pn_dir
    )
    norm_idx = np.where(y == -1)[0][0]
    arr_idx = np.where(y == 1)[0][0]

    norm_beat = beats[norm_idx]
    arr_beat = beats[arr_idx]

    sub_norm = compute_modwt_subbands(norm_beat, wavelet=wavelet, level=level)
    sub_arr = compute_modwt_subbands(arr_beat, wavelet=wavelet, level=level)

    channel_names = [f'A{level} (Trend / ST-T)'] + [f'D{level - i} (Detale / QRS)' for i in range(level)]

    fig, axes = plt.subplots(level + 2, 2, figsize=(11, 2.2 * (level + 2)), sharex=True)

    axes[0, 0].plot(norm_beat, color='steelblue', lw=1.5)
    axes[0, 0].set_title(f'Pobudzenie normalne (-1) - Rekord {record_id}')
    axes[0, 0].set_ylabel('Amplituda')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(arr_beat, color='crimson', lw=1.5)
    axes[0, 1].set_title(f'Pobudzenie arytmiczne (+1) - Rekord {record_id}')
    axes[0, 1].set_ylabel('Amplituda')
    axes[0, 1].grid(True, alpha=0.3)

    for i, name in enumerate(channel_names):
        row = i + 1
        axes[row, 0].plot(sub_norm[:, i], color='teal', lw=1.2)
        axes[row, 0].set_ylabel(name)
        axes[row, 0].grid(True, alpha=0.3)

        axes[row, 1].plot(sub_arr[:, i], color='darkorange', lw=1.2)
        axes[row, 1].set_ylabel(name)
        axes[row, 1].grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel('Probka')
    axes[-1, 1].set_xlabel('Probka')
    fig.suptitle(f'Dekompozycja wielopodpasmowa MODWT ({wavelet.upper()}, poziom {level})', fontsize=12, y=1.005)
    plt.tight_layout()
    plt.show()
