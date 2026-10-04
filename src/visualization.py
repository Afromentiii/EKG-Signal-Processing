import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

from .config import DEFAULT_FS, DEFAULT_WINDOW_SIZE, SAMPLE_PATIENT_GROUPS
from .preprocessing import build_cnn_dataset_with_groups, butter_bandpass_filter, extract_heartbeats


def plot_ecg_signal(
    record_id: str,
    start_second: float = 0.0,
    end_second: float = 10.0,
    pn_dir: str = 'mitdb',
    figsize: tuple[int, int] = (15, 4),
) -> None:
    try:
        record_info = wfdb.rdrecord(record_id, pn_dir=pn_dir, sampto=1)
        fs = record_info.fs

        start_sample = int(start_second * fs)
        end_sample = int(end_second * fs)

        record = wfdb.rdrecord(
            record_id,
            pn_dir=pn_dir,
            sampfrom=start_sample,
            sampto=end_sample,
        )
        annotation = wfdb.rdann(
            record_id,
            'atr',
            pn_dir=pn_dir,
            sampfrom=start_sample,
            sampto=end_sample,
            shift_samps=True,
        )

        wfdb.plot_wfdb(
            record=record,
            annotation=annotation,
            time_units='seconds',
            title=f'Analiza sygnalu EKG - Pacjent: {record_id}',
            figsize=figsize,
        )
    except Exception as exc:
        print(f"Blad rysowania rekordu {record_id}: {exc}")


def plot_random_beats_side_by_side(
    record_id: str = '102',
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    random_state: int = None,
    figsize: tuple[int, int] = (14, 4),
) -> None:
    X, y = extract_heartbeats(record_id, window_size=window_size, pn_dir=pn_dir)
    if len(X) < 2:
        raise ValueError(f"Rekord {record_id} posiada mniej niz 2 wyekstrahowane uderzenia.")

    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=2, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    time_ms = np.arange(window_size) / DEFAULT_FS * 1000.0

    label_desc = {1: 'Arytmia / Stymulacja (+1)', -1: 'Norma (-1)'}

    for idx, ax in zip(indices, axes):
        ax.plot(time_ms, X[idx], color='tab:blue', linewidth=1.5)
        ax.axvline(
            x=time_ms[window_size // 2],
            color='red',
            linestyle='--',
            alpha=0.7,
            label='Punkt R / Impuls',
        )
        ax.set_title(f"Pacjent {record_id} - Pobudzenie #{idx} ({label_desc.get(y[idx], y[idx])})")
        ax.set_xlabel("Czas [ms]")
        ax.set_ylabel("Z-score amplitudy")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_class_distribution(
    stats_df: pd.DataFrame,
    figsize: tuple[int, int] = (9, 4),
) -> None:
    records = stats_df['ID'].tolist()
    norm = stats_df['Norma (-1)'].tolist()
    arr = stats_df['Arytmia / Stym (+1)'].tolist()

    x = np.arange(len(records))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, norm, width, label='Norma (-1)', color='#2b5c8f')
    ax.bar(x + width / 2, arr, width, label='Arytmia / Stym (+1)', color='#d95f02')

    ax.set_xticks(x)
    ax.set_xticklabels([f"Pacjent {r}" for r in records])
    ax.set_ylabel("Liczba pobudzen")
    ax.set_title("Rozklad klas binarnych pobudzen per pacjent")
    ax.legend()
    ax.grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_group_mean_waveforms(
    patient_groups: dict[str, list[str]] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    pn_dir: str = 'mitdb',
    figsize: tuple[int, int] = (12, 4),
) -> None:
    if patient_groups is None:
        patient_groups = SAMPLE_PATIENT_GROUPS

    fig, axes = plt.subplots(1, len(patient_groups), figsize=figsize, sharey=True)
    if len(patient_groups) == 1:
        axes = [axes]

    time_ms = np.arange(window_size) / DEFAULT_FS * 1000.0

    for ax, (grp_name, rec_ids) in zip(axes, patient_groups.items()):
        X, _, _ = build_cnn_dataset_with_groups(rec_ids, window_size=window_size, pn_dir=pn_dir)
        beats = X[:, :, 0]
        mean_beat = np.mean(beats, axis=0)
        std_beat = np.std(beats, axis=0)

        ax.plot(time_ms, mean_beat, color='#1f77b4', linewidth=2, label='Sredni przebieg')
        ax.fill_between(
            time_ms,
            mean_beat - std_beat,
            mean_beat + std_beat,
            color='#1f77b4',
            alpha=0.25,
            label='±1 SD',
        )
        ax.axvline(
            x=time_ms[window_size // 2],
            color='red',
            linestyle='--',
            alpha=0.7,
            label='Punkt R',
        )
        ax.set_title(f"Grupa: {grp_name}")
        ax.set_xlabel("Czas [ms]")
        ax.set_ylabel("Z-score amplitudy")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_qrs_spectrum_comparison(
    record_standard: str = '100',
    record_paced: str = '102',
    beat_index: int = 20,
    pn_dir: str = 'mitdb',
    figsize: tuple[int, int] = (12, 6),
) -> None:
    rec100 = wfdb.rdrecord(record_standard, pn_dir=pn_dir)
    ann100 = wfdb.rdann(record_standard, 'atr', pn_dir=pn_dir)
    rec102 = wfdb.rdrecord(record_paced, pn_dir=pn_dir)
    ann102 = wfdb.rdann(record_paced, 'atr', pn_dir=pn_dir)

    fs = rec100.fs
    sig100 = butter_bandpass_filter(rec100.p_signal[:, 0], fs=fs)
    sig102 = butter_bandpass_filter(rec102.p_signal[:, 0], fs=fs)

    p100 = ann100.sample[beat_index]
    p102 = ann102.sample[beat_index]
    half_w = int(0.12 * fs)

    seg100 = sig100[p100 - half_w : p100 + half_w]
    seg102 = sig102[p102 - half_w : p102 + half_w]
    t_ms = np.linspace(-120, 120, len(seg100))

    fft100 = np.abs(np.fft.rfft(seg100))
    fft102 = np.abs(np.fft.rfft(seg102))
    freqs = np.fft.rfftfreq(len(seg100), 1.0 / fs)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    axes[0, 0].plot(t_ms, seg100, color='tab:blue', linewidth=1.5)
    axes[0, 0].set_title(f'Zespol QRS - Norma (Pacjent {record_standard})')
    axes[0, 0].set_xlabel('Czas [ms]')
    axes[0, 0].set_ylabel('Amplituda [mV]')
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)

    axes[0, 1].plot(t_ms, seg102, color='tab:red', linewidth=1.5)
    axes[0, 1].set_title(f'Zespol QRS - Stymulator (Pacjent {record_paced})')
    axes[0, 1].set_xlabel('Czas [ms]')
    axes[0, 1].set_ylabel('Amplituda [mV]')
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)

    axes[1, 0].plot(freqs, fft100, color='tab:blue', linewidth=1.5)
    axes[1, 0].set_title('Widmo zespolu QRS - Norma')
    axes[1, 0].set_xlabel('Czestotliwosc [Hz]')
    axes[1, 0].set_ylabel('|F(f)|')
    axes[1, 0].set_xlim(0, 100)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)

    axes[1, 1].plot(freqs, fft102, color='tab:red', linewidth=1.5)
    axes[1, 1].set_title('Widmo zespolu QRS - Stymulator')
    axes[1, 1].set_xlabel('Czestotliwosc [Hz]')
    axes[1, 1].set_ylabel('|F(f)|')
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()
