import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from .config import BINARY_LABEL_MAPPING, DEFAULT_WINDOW_SIZE
from .preprocessing import load_record_to_ram

FEATURE_NAMES = [
    'Pre-RR [s]',
    'Post-RR [s]',
    'Stosunek RR (Pre/Post)',
    'Pole zespolu QRS [mV·s]',
    'Amplituda Vpp [mV]',
    'Energia sygnalu pobudzenia',
    'Skosnosc (Skewness)',
    'Kurtoza (Kurtosis)',
    'Maksymalna stromosc dV/dt (pre-R) [mV/s]',
    'Dominujaca czestotliwosc [Hz]',
    'Srodek ciezkosci widma [Hz]',
    'Udzial energii 0.5-4 Hz',
    'Udzial energii 4-15 Hz (QRS)',
    'Udzial energii 15-40 Hz',
]

_FEATURES_RAM_CACHE: dict[tuple[str, int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def extract_engineered_features(
    record_id: str,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    label_mapping: dict = None,
    normalize_method: str = 'record_zscore',
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cache_key = (record_id, window_size, normalize_method)
    if cache_key in _FEATURES_RAM_CACHE:
        waves, feats, y_c = _FEATURES_RAM_CACHE[cache_key]
        return waves.copy(), feats.copy(), y_c.copy()

    if label_mapping is None:
        label_mapping = BINARY_LABEL_MAPPING

    use_rec_zscore = normalize_method in ('record_zscore', 'record')
    filtered_signal, annotation, fs = load_record_to_ram(
        record_id,
        pn_dir=pn_dir,
        normalize_record_zscore=use_rec_zscore,
    )
    r_peaks = annotation.sample
    symbols = annotation.symbol

    half_window = window_size // 2
    rr_diffs = np.diff(r_peaks) / fs

    waveforms = []
    features_list = []
    labels_list = []

    slope_window = int(0.050 * fs)

    for i in range(1, len(r_peaks) - 1):
        peak = r_peaks[i]
        sym = symbols[i]

        if sym not in label_mapping:
            continue
        if peak - half_window < 0 or peak + half_window >= len(filtered_signal):
            continue

        seg = filtered_signal[peak - half_window : peak + half_window]
        if normalize_method == 'beat_zscore':
            std_val = np.std(seg)
            if std_val == 0:
                continue
            norm_seg = (seg - np.mean(seg)) / std_val
        else:
            norm_seg = seg - np.median(seg)

        pre_rr = float(rr_diffs[i - 1])
        post_rr = float(rr_diffs[i])
        rr_ratio = float(pre_rr / (post_rr + 1e-6))

        qrs_half = int(0.08 * fs)
        qrs_seg = seg[half_window - qrs_half : half_window + qrs_half]
        qrs_area = float(np.trapz(np.abs(qrs_seg), dx=1.0 / fs))
        v_pp = float(np.ptp(seg))
        energy = float(np.sum(seg**2))

        sk = float(skew(seg))
        kt = float(kurtosis(seg))

        pre_diff = np.diff(seg[max(0, half_window - slope_window) : half_window]) * fs
        max_pre_slope = float(np.max(np.abs(pre_diff))) if len(pre_diff) > 0 else 0.0

        fft_vals = np.abs(np.fft.rfft(seg))
        freqs = np.fft.rfftfreq(len(seg), 1.0 / fs)

        dom_freq = float(freqs[np.argmax(fft_vals[1:]) + 1])
        spec_centroid = float(np.sum(freqs * fft_vals) / (np.sum(fft_vals) + 1e-6))

        band_low = float(np.sum(fft_vals[(freqs >= 0.5) & (freqs < 4)]))
        band_qrs = float(np.sum(fft_vals[(freqs >= 4) & (freqs < 15)]))
        band_high = float(np.sum(fft_vals[(freqs >= 15) & (freqs <= 40)]))
        total_band = band_low + band_qrs + band_high + 1e-6

        feat_vector = [
            pre_rr,
            post_rr,
            rr_ratio,
            qrs_area,
            v_pp,
            energy,
            sk,
            kt,
            max_pre_slope,
            dom_freq,
            spec_centroid,
            band_low / total_band,
            band_qrs / total_band,
            band_high / total_band,
        ]

        waveforms.append(norm_seg)
        features_list.append(feat_vector)
        labels_list.append(label_mapping[sym])

    W = np.array(waveforms)
    F = np.array(features_list)
    Y = np.array(labels_list)

    _FEATURES_RAM_CACHE[cache_key] = (W, F, Y)
    return W.copy(), F.copy(), Y.copy()


def build_multimodal_dataset_with_groups(
    record_ids: list[str],
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    standardize_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_waves, all_feats, all_y, all_groups = [], [], [], []

    for rec in record_ids:
        try:
            W, F, Y = extract_engineered_features(rec, window_size=window_size, pn_dir=pn_dir)
            if len(Y) == 0:
                continue
            all_waves.append(W)
            all_feats.append(F)
            all_y.append(Y)
            all_groups.append(np.full((len(Y),), rec))
        except Exception as exc:
            print(f"Blad podczas ekstrakcji cech z rekordu {rec}: {exc}")

    if not all_waves:
        raise ValueError("Nie udalo sie wyodrebnic cech dla podanej listy rekordow.")

    X_waves = np.concatenate(all_waves, axis=0)
    X_feats = np.concatenate(all_feats, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    groups_final = np.concatenate(all_groups, axis=0)

    X_waves = X_waves.reshape((X_waves.shape[0], X_waves.shape[1], 1))

    if standardize_features:
        scaler = StandardScaler()
        X_feats = scaler.fit_transform(X_feats)

    return X_waves, X_feats, y_final, groups_final


def get_feature_importance_table(
    X_features: np.ndarray,
    y: np.ndarray,
    feature_names: list[str] = None,
) -> pd.DataFrame:
    if feature_names is None:
        feature_names = FEATURE_NAMES

    f_values, p_values = f_classif(X_features, y)
    df = pd.DataFrame({
        'Cecha biomedyczna': feature_names,
        'Wskaznik ANOVA F': np.round(f_values, 2),
        'Wartosc p': [f"{p:.2e}" for p in p_values],
    })
    return df.sort_values(by='Wskaznik ANOVA F', ascending=False).reset_index(drop=True)


def get_feature_importance_per_group(
    patient_groups: dict[str, list[str]] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if patient_groups is None:
        from .config import SAMPLE_PATIENT_GROUPS
        patient_groups = SAMPLE_PATIENT_GROUPS

    results = {'Cecha biomedyczna': FEATURE_NAMES}

    for grp_name, rec_ids in patient_groups.items():
        all_F, all_Y = [], []
        for r in rec_ids:
            try:
                _, F, Y = extract_engineered_features(r, window_size=window_size, pn_dir=pn_dir)
                if len(Y) > 0:
                    all_F.append(F)
                    all_Y.append(Y)
            except Exception:
                pass
        if all_F:
            F_arr = np.vstack(all_F)
            Y_arr = np.concatenate(all_Y)
            if len(np.unique(Y_arr)) > 1:
                f_vals, _ = f_classif(F_arr, Y_arr)
                results[f"ANOVA F ({grp_name})"] = np.round(f_vals, 2)
            else:
                results[f"ANOVA F ({grp_name})"] = 0.0

    df_res = pd.DataFrame(results)
    sort_col = [c for c in df_res.columns if 'ANOVA' in c]
    if sort_col:
        df_res = df_res.sort_values(by=sort_col[0], ascending=False)
    return df_res.reset_index(drop=True)


RELATIONAL_QRS_FEATURE_NAMES = [
    'Wariancja stromosci Var(dV/dt) [mV^2/s^2]',
    'Asymetria zbocza narastajacego do opadajacego',
    'Szerokosc w polowie wysokosci (FWHM) [ms]',
    'Pole pod zespolem QRS [mV·s]',
]


def extract_relational_qrs_features(
    segment: np.ndarray,
    fs: float = 360.0,
    n_pre: int = 18,
) -> list[float]:
    area = float(np.trapz(np.abs(segment), dx=1.0 / fs))

    half_max = float(np.max(segment)) / 2.0
    above_half = np.where(segment >= half_max)[0]
    fwhm = float((above_half[-1] - above_half[0]) / fs * 1000.0) if len(above_half) > 1 else 0.0

    diff = np.diff(segment) * fs
    var_dvdt = float(np.var(diff))

    rise_slope = float(np.max(np.abs(diff[:n_pre]))) if n_pre > 0 else 1.0
    fall_slope = float(np.max(np.abs(diff[n_pre:]))) if len(diff) > n_pre else 1.0
    slope_asym = float(rise_slope / (fall_slope + 1e-6))

    return [var_dvdt, slope_asym, fwhm, area]


def get_relational_qrs_importance_table(
    record_ids: list[str] = None,
    pn_dir: str = 'mitdb',
    window_range: tuple[int, int] = (-18, 36),
) -> pd.DataFrame:
    if record_ids is None:
        from .config import SAMPLE_RECORDS
        record_ids = SAMPLE_RECORDS

    from .preprocessing import build_cnn_dataset_with_groups

    X_asym, y_all, _ = build_cnn_dataset_with_groups(
        record_ids, window_range=window_range, pn_dir=pn_dir
    )
    X_beats = X_asym.squeeze()

    n_pre = abs(window_range[0])
    feats = np.array([extract_relational_qrs_features(b, n_pre=n_pre) for b in X_beats])

    f_vals, p_vals = f_classif(feats, y_all)
    df = pd.DataFrame({
        'Cecha relacyjna (okno 150 ms)': RELATIONAL_QRS_FEATURE_NAMES,
        'Wskaznik ANOVA F': np.round(f_vals, 2),
        'Wartosc p': [f"{p:.2e}" for p in p_vals],
    })
    return df.sort_values(by='Wskaznik ANOVA F', ascending=False).reset_index(drop=True)


def run_permutation_shuffle_test(
    record_ids: list[str] = None,
    pn_dir: str = 'mitdb',
    window_range: tuple[int, int] = (-18, 36),
    random_state: int = 42,
) -> pd.DataFrame:
    if record_ids is None:
        from .config import SAMPLE_RECORDS
        record_ids = SAMPLE_RECORDS

    from .preprocessing import build_cnn_dataset_with_groups

    X_asym, y_all, groups = build_cnn_dataset_with_groups(
        record_ids, window_range=window_range, pn_dir=pn_dir
    )
    X_beats = X_asym.squeeze()
    n_pre = abs(window_range[0])
    feats = np.array([extract_relational_qrs_features(b, n_pre=n_pre) for b in X_beats])

    f_orig, _ = f_classif(feats, y_all)

    rng = np.random.default_rng(random_state)
    y_shuffled = rng.permutation(y_all)
    f_shuffled, p_shuffled = f_classif(feats, y_shuffled)

    unique_p = np.unique(groups)
    perm_p = rng.permutation(unique_p)
    p_map = dict(zip(unique_p, perm_p))
    y_block = np.zeros_like(y_all)
    for i, g in enumerate(groups):
        candidates = y_all[groups == p_map[g]]
        y_block[i] = rng.choice(candidates)

    f_block, _ = f_classif(feats, y_block)

    df = pd.DataFrame({
        'Cecha relacyjna': RELATIONAL_QRS_FEATURE_NAMES,
        'ANOVA F (Rzeczywiste y)': np.round(f_orig, 2),
        'ANOVA F (Permutacja losowa)': np.round(f_shuffled, 2),
        'Wartosc p (Permutacja)': [f"{p:.2e}" for p in p_shuffled],
        'ANOVA F (Permutacja blokowa pacjentow)': np.round(f_block, 2),
    })
    return df.sort_values(by='ANOVA F (Rzeczywiste y)', ascending=False).reset_index(drop=True)
