import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .config import DEFAULT_FS, SAMPLE_PATIENT_GROUPS, SAMPLE_RECORDS
from .metadata import get_patient_metadata
from .preprocessing import load_record_to_ram


def compute_record_signal_metrics(
    record_id: str,
    pn_dir: str = 'mitdb',
) -> dict:
    meta = get_patient_metadata(record_id, pn_dir=pn_dir)
    filtered_signal, annotation, fs = load_record_to_ram(record_id, pn_dir=pn_dir)

    r_peaks = annotation.sample
    rr_intervals = np.diff(r_peaks) / fs * 1000.0
    valid_rr = rr_intervals[(rr_intervals >= 200.0) & (rr_intervals <= 2500.0)]

    mean_rr = float(np.mean(valid_rr)) if len(valid_rr) > 0 else 0.0
    bpm = (60000.0 / mean_rr) if mean_rr > 0 else 0.0
    sdnn = float(np.std(valid_rr)) if len(valid_rr) > 0 else 0.0

    v_pp = float(np.ptp(filtered_signal))
    rms = float(np.sqrt(np.mean(filtered_signal**2)))
    sk = float(skew(filtered_signal))
    kt = float(kurtosis(filtered_signal))

    # Analiza morfologiczna i czasowa zespolu QRS (okno +/- 80 ms wokol zalamka R)
    half_qrs = int(0.08 * fs)
    qrs_widths = []
    r_amps = []
    qrs_areas = []

    sampled_peaks = r_peaks[5:105] if len(r_peaks) >= 105 else r_peaks
    for p in sampled_peaks:
        if p - half_qrs < 0 or p + half_qrs >= len(filtered_signal):
            continue
        seg = filtered_signal[p - half_qrs : p + half_qrs]
        r_amps.append(np.max(seg))
        qrs_areas.append(np.trapz(np.abs(seg), dx=1.0 / fs) * 1000.0)

        thresh = 0.2 * np.max(np.abs(seg))
        above = np.where(np.abs(seg) > thresh)[0]
        if len(above) > 0:
            width_ms = (above[-1] - above[0]) / fs * 1000.0
            qrs_widths.append(width_ms)

    mean_r_amp = float(np.mean(r_amps)) if r_amps else 0.0
    mean_qrs_width = float(np.mean(qrs_widths)) if qrs_widths else 0.0
    mean_qrs_area = float(np.mean(qrs_areas)) if qrs_areas else 0.0

    return {
        'ID': record_id,
        'Grupa': meta['Grupa'],
        'Tetno [BPM]': round(bpm, 1),
        'RR [ms]': round(mean_rr, 1),
        'SDNN [ms]': round(sdnn, 1),
        'Amplituda R [mV]': round(mean_r_amp, 2),
        'Szerokosc QRS [ms]': round(mean_qrs_width, 1),
        'Pole QRS [mV·ms]': round(mean_qrs_area, 1),
        'Vpp [mV]': round(v_pp, 2),
        'Kurtoza': round(kt, 1),
    }


def get_signal_statistics_table(
    record_ids: list[str] = None,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if record_ids is None:
        record_ids = SAMPLE_RECORDS

    metrics_list = [compute_record_signal_metrics(r, pn_dir=pn_dir) for r in record_ids]
    return pd.DataFrame(metrics_list)


def get_group_signal_statistics_table(
    patient_groups: dict[str, list[str]] = None,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if patient_groups is None:
        patient_groups = SAMPLE_PATIENT_GROUPS

    rows = []
    for grp_name, rec_ids in patient_groups.items():
        grp_metrics = [compute_record_signal_metrics(r, pn_dir=pn_dir) for r in rec_ids]
        df_grp = pd.DataFrame(grp_metrics)

        rows.append({
            'Grupa Pacjentow': grp_name,
            'Liczba Pacjentow': len(rec_ids),
            'Tetno [BPM]': f"{df_grp['Tetno [BPM]'].mean():.1f} (+/- {df_grp['Tetno [BPM]'].std():.1f})",
            'RR [ms]': f"{df_grp['RR [ms]'].mean():.1f} (+/- {df_grp['RR [ms]'].std():.1f})",
            'SDNN [ms]': f"{df_grp['SDNN [ms]'].mean():.1f} (+/- {df_grp['SDNN [ms]'].std():.1f})",
            'Amplituda R [mV]': f"{df_grp['Amplituda R [mV]'].mean():.2f} (+/- {df_grp['Amplituda R [mV]'].std():.2f})",
            'Szerokosc QRS [ms]': f"{df_grp['Szerokosc QRS [ms]'].mean():.1f} (+/- {df_grp['Szerokosc QRS [ms]'].std():.1f})",
            'Pole QRS [mV·ms]': f"{df_grp['Pole QRS [mV·ms]'].mean():.1f} (+/- {df_grp['Pole QRS [mV·ms]'].std():.1f})",
            'Vpp [mV]': f"{df_grp['Vpp [mV]'].mean():.2f} (+/- {df_grp['Vpp [mV]'].std():.2f})",
            'Kurtoza': f"{df_grp['Kurtoza'].mean():.1f} (+/- {df_grp['Kurtoza'].std():.1f})",
        })

    return pd.DataFrame(rows)
