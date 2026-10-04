from collections import Counter
import pandas as pd
import wfdb

from .config import BINARY_LABEL_MAPPING, SAMPLE_PATIENT_GROUPS, SAMPLE_RECORDS


def get_patient_metadata(record_id: str, pn_dir: str = 'mitdb') -> dict:
    header = wfdb.rdheader(record_id, pn_dir=pn_dir)
    comments = header.comments

    first_line = comments[0].split() if comments else []
    age = first_line[0] if len(first_line) > 0 else 'N/A'
    sex = first_line[1] if len(first_line) > 1 else 'N/A'
    meds = comments[1] if len(comments) > 1 else 'Brak'
    notes = ' '.join(comments[2:]) if len(comments) > 2 else ('Brak' if len(comments) <= 1 else '')

    is_paced = any('pace' in c.lower() for c in comments)

    return {
        'ID Rekordu': record_id,
        'Wiek': age,
        'Plec': sex,
        'Leki': meds,
        'Stymulator (Pacemaker)': 'Tak' if is_paced else 'Nie',
        'Grupa': 'Pacemaker' if is_paced else 'Standard',
        'Uwagi kliniczne': notes if notes else 'Rytm zatokowy / standardowy',
    }


def get_patient_metadata_table(
    record_ids: list[str] = None,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if record_ids is None:
        record_ids = SAMPLE_RECORDS

    records_meta = [get_patient_metadata(rec_id, pn_dir=pn_dir) for rec_id in record_ids]
    return pd.DataFrame(records_meta)


def get_patient_statistics_table(
    record_ids: list[str] = None,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if record_ids is None:
        record_ids = SAMPLE_RECORDS

    rows = []
    for rec_id in record_ids:
        meta = get_patient_metadata(rec_id, pn_dir=pn_dir)
        ann = wfdb.rdann(rec_id, 'atr', pn_dir=pn_dir)
        counts = Counter(ann.symbol)

        norm_count = sum(counts[s] for s in counts if BINARY_LABEL_MAPPING.get(s) == -1)
        arr_count = sum(counts[s] for s in counts if BINARY_LABEL_MAPPING.get(s) == 1)
        total_valid = norm_count + arr_count

        pct_positive = (arr_count / total_valid * 100.0) if total_valid > 0 else 0.0

        rows.append({
            'ID': rec_id,
            'Grupa': meta['Grupa'],
            'Wiek': meta['Wiek'],
            'Plec': meta['Plec'],
            'Stymulator': meta['Stymulator (Pacemaker)'],
            'Norma (-1)': norm_count,
            'Arytmia / Stym (+1)': arr_count,
            'Razem': total_valid,
            'Udzial (+1)': f"{pct_positive:.1f}%",
        })

    return pd.DataFrame(rows)


def get_group_statistics_table(
    patient_groups: dict[str, list[str]] = None,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if patient_groups is None:
        patient_groups = SAMPLE_PATIENT_GROUPS

    rows = []
    for grp_name, rec_ids in patient_groups.items():
        norm_total = 0
        arr_total = 0

        for rec_id in rec_ids:
            ann = wfdb.rdann(rec_id, 'atr', pn_dir=pn_dir)
            counts = Counter(ann.symbol)
            norm_total += sum(counts[s] for s in counts if BINARY_LABEL_MAPPING.get(s) == -1)
            arr_total += sum(counts[s] for s in counts if BINARY_LABEL_MAPPING.get(s) == 1)

        total_beats = norm_total + arr_total
        pct = (arr_total / total_beats * 100.0) if total_beats > 0 else 0.0

        rows.append({
            'Grupa Pacjentow': grp_name,
            'Liczba Pacjentow': len(rec_ids),
            'Pobudzenia Norma (-1)': norm_total,
            'Pobudzenia Arytmia/Stym (+1)': arr_total,
            'Razem Pobudzen': total_beats,
            'Udzial Klasy (+1)': f"{pct:.1f}%",
        })

    return pd.DataFrame(rows)


def get_aami_statistics_table(
    patient_groups: dict[str, list[str]] = None,
    pn_dir: str = 'mitdb',
) -> pd.DataFrame:
    if patient_groups is None:
        from .config import FULL_PATIENT_GROUPS
        patient_groups = FULL_PATIENT_GROUPS

    from .config import AAMI_MAPPING

    rows = []
    for grp_name, rec_ids in patient_groups.items():
        counts_total = Counter()
        for rec_id in rec_ids:
            ann = wfdb.rdann(rec_id, 'atr', pn_dir=pn_dir)
            for s in ann.symbol:
                if s in AAMI_MAPPING:
                    counts_total[AAMI_MAPPING[s]] += 1

        total = sum(counts_total.values())
        rows.append({
            'Grupa Pacjentow': grp_name,
            'Liczba Pacjentow': len(rec_ids),
            'N (Norma / Bloki)': f"{counts_total['N']} ({counts_total['N']/total*100:.1f}%)" if total else "0",
            'S (Nadkomorowe)': f"{counts_total['S']} ({counts_total['S']/total*100:.1f}%)" if total else "0",
            'V (Komorowe)': f"{counts_total['V']} ({counts_total['V']/total*100:.1f}%)" if total else "0",
            'F (Zlane)': f"{counts_total['F']} ({counts_total['F']/total*100:.1f}%)" if total else "0",
            'Q (Stymulator / Nieznane)': f"{counts_total['Q']} ({counts_total['Q']/total*100:.1f}%)" if total else "0",
            'Razem Pobudzen': total,
        })

    return pd.DataFrame(rows)
