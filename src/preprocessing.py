import concurrent.futures
import numpy as np
import wfdb
from scipy.signal import butter, filtfilt

from .config import (
    ALL_RECORDS,
    BINARY_LABEL_MAPPING,
    DEFAULT_FS,
    DEFAULT_HIGHCUT,
    DEFAULT_LOWCUT,
    DEFAULT_ORDER,
    DEFAULT_WINDOW_SIZE,
    SAMPLE_RECORDS,
)

_SIGNAL_RAM_CACHE: dict[str, tuple[np.ndarray, wfdb.Annotation, float]] = {}
_HEARTBEATS_RAM_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
    fs: float = DEFAULT_FS,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def load_record_to_ram(
    record_id: str,
    pn_dir: str = 'mitdb',
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
    order: int = DEFAULT_ORDER,
    normalize_record_zscore: bool = True,
) -> tuple[np.ndarray, wfdb.Annotation, float]:
    cache_key = f"{record_id}_zscore_{normalize_record_zscore}"
    if cache_key in _SIGNAL_RAM_CACHE:
        return _SIGNAL_RAM_CACHE[cache_key]

    record = wfdb.rdrecord(record_id, pn_dir=pn_dir)
    annotation = wfdb.rdann(record_id, 'atr', pn_dir=pn_dir)
    raw_signal = record.p_signal[:, 0]
    filtered_signal = butter_bandpass_filter(
        raw_signal,
        lowcut=lowcut,
        highcut=highcut,
        fs=record.fs,
        order=order,
    )

    if normalize_record_zscore:
        rec_mean = np.mean(filtered_signal)
        rec_std = np.std(filtered_signal)
        if rec_std > 0:
            filtered_signal = (filtered_signal - rec_mean) / rec_std

    data_tuple = (filtered_signal, annotation, float(record.fs))
    _SIGNAL_RAM_CACHE[cache_key] = data_tuple
    return data_tuple


def extract_heartbeats(
    record_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    label_mapping: dict = None,
    lowcut: float = DEFAULT_LOWCUT,
    highcut: float = DEFAULT_HIGHCUT,
    order: int = DEFAULT_ORDER,
    normalize_method: str = 'record_zscore',
    window_range: tuple[int, int] = None,
    suppress_pacing_spike: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (record_id, window_size, normalize_method, window_range, suppress_pacing_spike)
    if cache_key in _HEARTBEATS_RAM_CACHE:
        X_cached, y_cached = _HEARTBEATS_RAM_CACHE[cache_key]
        return X_cached.copy(), y_cached.copy()

    if label_mapping is None:
        label_mapping = BINARY_LABEL_MAPPING

    use_rec_zscore = normalize_method in ('record_zscore', 'record')
    filtered_signal, annotation, fs = load_record_to_ram(
        record_id,
        pn_dir=pn_dir,
        lowcut=lowcut,
        highcut=highcut,
        order=order,
        normalize_record_zscore=use_rec_zscore,
    )

    if window_range is not None:
        n_pre = abs(window_range[0])
        n_post = window_range[1]
    else:
        n_pre = window_size // 2
        n_post = window_size // 2

    r_peaks = annotation.sample
    symbols = annotation.symbol

    X_list, y_list = [], []

    spike_pre = int(0.025 * fs)
    spike_post = max(1, int(0.003 * fs))

    for peak, sym in zip(r_peaks, symbols):
        if sym not in label_mapping:
            continue

        if peak - n_pre < 0 or peak + n_post >= len(filtered_signal):
            continue

        segment = filtered_signal[peak - n_pre : peak + n_post].copy()

        if suppress_pacing_spike and n_pre >= spike_pre:
            s_start = n_pre - spike_pre
            s_end = n_pre - spike_post
            segment[s_start:s_end] = np.linspace(segment[s_start], segment[s_end], s_end - s_start)

        if normalize_method == 'beat_zscore':
            std_val = np.std(segment)
            if std_val == 0:
                continue
            normalized_segment = (segment - np.mean(segment)) / std_val
        else:
            normalized_segment = segment - np.median(segment)

        X_list.append(normalized_segment)
        y_list.append(label_mapping[sym])

    X_arr, y_arr = np.array(X_list), np.array(y_list)
    _HEARTBEATS_RAM_CACHE[cache_key] = (X_arr, y_arr)
    return X_arr.copy(), y_arr.copy()


def balance_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_aux: np.ndarray = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    classes, counts = np.unique(y_train, return_counts=True)
    if len(classes) < 2 or counts[0] == counts[1]:
        return (X_train, y_train) if X_aux is None else (X_train, X_aux, y_train)

    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]

    min_idx = np.where(y_train == min_class)[0]
    maj_idx = np.where(y_train == maj_class)[0]

    rng = np.random.default_rng(random_state)

    target_min_count = max(200, len(min_idx))
    sampled_min_idx = rng.choice(
        min_idx,
        size=target_min_count,
        replace=(len(min_idx) < target_min_count),
    )

    target_maj_count = min(len(maj_idx), target_min_count * 2)
    sampled_maj_idx = rng.choice(maj_idx, size=target_maj_count, replace=False)

    combined = np.concatenate([sampled_min_idx, sampled_maj_idx])
    rng.shuffle(combined)

    if X_aux is not None:
        return X_train[combined], X_aux[combined], y_train[combined]
    return X_train[combined], y_train[combined]


def preload_records_to_ram(
    record_ids: list[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    max_workers: int = 8,
) -> None:
    if record_ids is None:
        record_ids = ALL_RECORDS

    def _worker(rec):
        try:
            extract_heartbeats(rec, window_size=window_size, pn_dir=pn_dir)
            return rec, True
        except Exception as exc:
            return rec, False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_worker, record_ids))

    succeeded = sum(1 for _, ok in results if ok)
    print(f"Zaladowano do RAM: {succeeded}/{len(record_ids)} rekordow w trybie wielowatkowym.")


def clear_ram_cache() -> None:
    _SIGNAL_RAM_CACHE.clear()
    _HEARTBEATS_RAM_CACHE.clear()


def build_cnn_dataset_with_groups(
    record_ids: list[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    label_mapping: dict = None,
    window_range: tuple[int, int] = None,
    suppress_pacing_spike: bool = False,
    normalize_method: str = 'record_zscore',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_X, all_y, all_groups = [], [], []

    for rec in record_ids:
        try:
            X_patient, y_patient = extract_heartbeats(
                rec,
                window_size=window_size,
                pn_dir=pn_dir,
                label_mapping=label_mapping,
                window_range=window_range,
                suppress_pacing_spike=suppress_pacing_spike,
                normalize_method=normalize_method,
            )
            if len(y_patient) == 0:
                continue

            all_X.append(X_patient)
            all_y.append(y_patient)
            all_groups.append(np.full((len(y_patient),), rec))
        except Exception as exc:
            print(f"Blad podczas przetwarzania rekordu {rec}: {exc}")

    if not all_X:
        raise ValueError("Nie udalo sie zaladowac danych dla podanej listy pacjentow.")

    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    groups_final = np.concatenate(all_groups, axis=0)

    X_final = X_final.reshape((X_final.shape[0], X_final.shape[1], 1))
    return X_final, y_final, groups_final
