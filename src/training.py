import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import tensorflow as tf

from .evaluation import compute_detailed_metrics
from .models import build_csnet_binary_model, build_hybrid_feature_csnet
from .preprocessing import balance_training_data, build_cnn_dataset_with_groups


def evaluate_group_kfold(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    epochs: int = 3,
    batch_size: int = 128,
    l2_reg: float = 0.01,
    learning_rate: float = 0.001,
    balance_classes: bool = True,
    verbose: int = 0,
) -> tuple[pd.DataFrame, dict, tf.keras.Model, tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Liczba unikalnych grup ({len(unique_groups)}) jest mniejsza niz n_splits ({n_splits}). "
            f"Zmniejsz n_splits do maksymalnie {len(unique_groups)}."
        )

    gkf = GroupKFold(n_splits=n_splits)
    metrics_records = []
    all_y_true = []
    all_y_pred = []

    fold_no = 1
    window_size = X.shape[1]
    last_model = None

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        print(f"\n--- Rozpoczynam Fold {fold_no}/{n_splits} ---")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        test_patients = np.unique(groups[test_idx])
        print(f"  Pacjenci weryfikacyjni: {test_patients}")

        if balance_classes:
            X_train_fit, y_train_fit = balance_training_data(X_train, y_train)
        else:
            X_train_fit, y_train_fit = X_train, y_train

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train_fit, y_train_fit))
            .cache()
            .shuffle(buffer_size=min(len(X_train_fit), 10000))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = (
            tf.data.Dataset.from_tensor_slices(X_test)
            .cache()
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        model = build_csnet_binary_model(
            window_size=window_size,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )

        model.fit(
            train_ds,
            epochs=epochs,
            verbose=verbose,
        )
        last_model = model

        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred = np.where(y_pred_probs > 0, 1, -1).flatten()

        fold_metrics = compute_detailed_metrics(y_test, y_pred)
        metrics_records.append(fold_metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(
            f"  Wyniki foldu -> Acc: {fold_metrics['accuracy']:.3f}, "
            f"Bal-Acc: {fold_metrics['balanced_accuracy']:.3f}, "
            f"Czulosc (+1): {fold_metrics['sensitivity_pos']:.3f}, "
            f"Specyficznosc (-1): {fold_metrics['specificity_neg']:.3f}"
        )
        fold_no += 1

    df_m = pd.DataFrame(metrics_records)
    results_df = pd.DataFrame([{
        'Model': 'CS-NET (CNN + SVM)',
        'Balansowanie': 'Wlaczone (Resampling)' if balance_classes else 'Wylaczone (Surowe)',
        'Accuracy': f"{df_m['accuracy'].mean():.4f} (+/- {df_m['accuracy'].std():.4f})",
        'Balanced Accuracy': f"{df_m['balanced_accuracy'].mean():.4f} (+/- {df_m['balanced_accuracy'].std():.4f})",
        'Czulosc (+1)': f"{df_m['sensitivity_pos'].mean():.4f} (+/- {df_m['sensitivity_pos'].std():.4f})",
        'Specyficznosc (-1)': f"{df_m['specificity_neg'].mean():.4f} (+/- {df_m['specificity_neg'].std():.4f})",
        'F1-Score (macro)': f"{df_m['f1_macro'].mean():.4f} (+/- {df_m['f1_macro'].std():.4f})",
        'MCC (Absolute)': f"{df_m['mcc_abs'].mean():.4f} (+/- {df_m['mcc_abs'].std():.4f})",
    }])

    return results_df, df_m.to_dict(orient='list'), last_model, (np.array(all_y_true), np.array(all_y_pred))


def evaluate_hybrid_feature_csnet(
    X_waves: np.ndarray,
    X_feats: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    epochs: int = 3,
    batch_size: int = 128,
    l2_reg: float = 0.01,
    learning_rate: float = 0.001,
    balance_classes: bool = True,
    verbose: int = 0,
) -> tuple[pd.DataFrame, tf.keras.Model, tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Liczba unikalnych grup ({len(unique_groups)}) jest mniejsza niz n_splits ({n_splits})."
        )

    gkf = GroupKFold(n_splits=n_splits)
    metrics_records = []
    all_y_true = []
    all_y_pred = []

    fold_no = 1
    window_size = X_waves.shape[1]
    n_features = X_feats.shape[1]
    last_model = None

    for train_idx, test_idx in gkf.split(X_waves, y, groups=groups):
        print(f"\n--- Hybrydowy CS-NET: Fold {fold_no}/{n_splits} ---")
        X_w_train, X_w_test = X_waves[train_idx], X_waves[test_idx]
        X_f_train, X_f_test = X_feats[train_idx], X_feats[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if balance_classes:
            X_w_fit, X_f_fit, y_fit = balance_training_data(X_w_train, y_train, X_aux=X_f_train)
        else:
            X_w_fit, X_f_fit, y_fit = X_w_train, X_f_train, y_train

        train_ds = (
            tf.data.Dataset.from_tensor_slices((
                {'waveform_input': X_w_fit, 'feature_input': X_f_fit},
                y_fit,
            ))
            .cache()
            .shuffle(buffer_size=min(len(y_fit), 10000))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        model = build_hybrid_feature_csnet(
            window_size=window_size,
            n_features=n_features,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )

        model.fit(
            train_ds,
            epochs=epochs,
            verbose=verbose,
        )
        last_model = model

        y_pred_probs = model.predict([X_w_test, X_f_test], batch_size=batch_size, verbose=0)
        y_pred = np.where(y_pred_probs > 0, 1, -1).flatten()

        fold_metrics = compute_detailed_metrics(y_test, y_pred)
        metrics_records.append(fold_metrics)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        print(
            f"  Wyniki foldu -> Acc: {fold_metrics['accuracy']:.3f}, "
            f"Bal-Acc: {fold_metrics['balanced_accuracy']:.3f}, "
            f"Czulosc (+1): {fold_metrics['sensitivity_pos']:.3f}, "
            f"Specyficznosc (-1): {fold_metrics['specificity_neg']:.3f}"
        )
        fold_no += 1

    df_m = pd.DataFrame(metrics_records)
    results_df = pd.DataFrame([{
        'Model': 'Hybrydowy CS-NET (Fale + 13 Cech)',
        'Balansowanie': 'Wlaczone (Resampling)' if balance_classes else 'Wylaczone',
        'Accuracy': f"{df_m['accuracy'].mean():.4f} (+/- {df_m['accuracy'].std():.4f})",
        'Balanced Accuracy': f"{df_m['balanced_accuracy'].mean():.4f} (+/- {df_m['balanced_accuracy'].std():.4f})",
        'Czulosc (+1)': f"{df_m['sensitivity_pos'].mean():.4f} (+/- {df_m['sensitivity_pos'].std():.4f})",
        'Specyficznosc (-1)': f"{df_m['specificity_neg'].mean():.4f} (+/- {df_m['specificity_neg'].std():.4f})",
        'F1-Score (macro)': f"{df_m['f1_macro'].mean():.4f} (+/- {df_m['f1_macro'].std():.4f})",
        'MCC (Absolute)': f"{df_m['mcc_abs'].mean():.4f} (+/- {df_m['mcc_abs'].std():.4f})",
    }])

    return results_df, last_model, (np.array(all_y_true), np.array(all_y_pred))


def evaluate_by_patient_groups(
    patient_groups: dict[str, list[str]],
    window_size: int = 300,
    epochs: int = 3,
    batch_size: int = 128,
    l2_reg: float = 0.01,
    learning_rate: float = 0.001,
    balance_classes: bool = True,
    verbose: int = 0,
    return_artifacts: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, dict]]:
    rows = []
    artifacts = {}

    for group_name, record_ids in patient_groups.items():
        print(f"\n=======================================================")
        print(f" Trening dla grupy: {group_name} (liczba rekordow: {len(record_ids)}) | Balans: {balance_classes}")
        print(f"=======================================================")

        X, y, groups = build_cnn_dataset_with_groups(record_ids, window_size=window_size)
        n_splits = min(len(set(groups)), 5)

        group_results, _, model, (y_true_all, y_pred_all) = evaluate_group_kfold(
            X,
            y,
            groups,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
            balance_classes=balance_classes,
            verbose=verbose,
        )

        row = group_results.iloc[0].to_dict()
        row['Grupa Pacjentow'] = group_name
        row['Liczba Pacjentow'] = len(set(groups))
        row['Pobudzenia'] = len(y)
        rows.append(row)

        artifacts[group_name] = {
            'model': model,
            'y_true': y_true_all,
            'y_pred': y_pred_all,
            'X': X,
            'y': y,
        }

    cols = [
        'Grupa Pacjentow',
        'Balansowanie',
        'Pobudzenia',
        'Accuracy',
        'Balanced Accuracy',
        'Czulosc (+1)',
        'Specyficznosc (-1)',
        'F1-Score (macro)',
        'MCC (Absolute)',
    ]
    summary_df = pd.DataFrame(rows)[cols]
    if return_artifacts:
        return summary_df, artifacts
    return summary_df


def compare_balancing_impact(
    records_or_groups: list[str] | dict[str, list[str]] = None,
    epochs: int = 3,
    batch_size: int = 128,
) -> pd.DataFrame:
    if records_or_groups is None:
        record_ids = ['100', '101']
    elif isinstance(records_or_groups, dict):
        record_ids = [r for recs in records_or_groups.values() for r in recs]
    else:
        record_ids = list(records_or_groups)

    X, y, groups = build_cnn_dataset_with_groups(record_ids)
    n_splits = min(len(set(groups)), 5)

    df_raw, _, _, _ = evaluate_group_kfold(
        X, y, groups, n_splits=n_splits, epochs=epochs, batch_size=batch_size, balance_classes=False
    )
    df_raw.insert(0, 'Strategia Treningu', 'Brak balansu (surowe dane)')

    df_bal, _, _, _ = evaluate_group_kfold(
        X, y, groups, n_splits=n_splits, epochs=epochs, batch_size=batch_size, balance_classes=True
    )
    df_bal.insert(0, 'Strategia Treningu', 'Zbalansowany fold treningowy (1:1)')

    return pd.concat([df_raw, df_bal], ignore_index=True)


def compare_window_shapes(
    record_ids: list[str] = None,
    epochs: int = 3,
    batch_size: int = 128,
) -> pd.DataFrame:
    if record_ids is None:
        record_ids = ['100', '101', '102', '104']

    # 1. Symetryczne okno 300 probek (ok. 833 ms)
    X_sym, y_sym, g_sym = build_cnn_dataset_with_groups(record_ids, window_size=300)
    n_splits = min(len(set(g_sym)), 4)
    df_sym, _, _, _ = evaluate_group_kfold(
        X_sym, y_sym, g_sym, n_splits=n_splits, epochs=epochs, batch_size=batch_size, balance_classes=True
    )
    df_sym.insert(0, 'Ksztalt Okna', 'Symetryczne okno (300 probek / 833 ms)')

    # 2. Asymetryczne okno QRS (-50 ms do +100 ms / 54 probki)
    X_asym, y_asym, g_asym = build_cnn_dataset_with_groups(
        record_ids, window_range=(-18, 36), suppress_pacing_spike=False
    )
    df_asym, _, _, _ = evaluate_group_kfold(
        X_asym, y_asym, g_asym, n_splits=n_splits, epochs=epochs, batch_size=batch_size, balance_classes=True
    )
    df_asym.insert(0, 'Ksztalt Okna', 'Asymetryczne QRS (-50 ms do +100 ms / 54 probki)')

    # 3. Asymetryczne okno QRS + Supresja iglicy stymulacji
    X_supp, y_supp, g_supp = build_cnn_dataset_with_groups(
        record_ids, window_range=(-18, 36), suppress_pacing_spike=True
    )
    df_supp, _, _, _ = evaluate_group_kfold(
        X_supp, y_supp, g_supp, n_splits=n_splits, epochs=epochs, batch_size=batch_size, balance_classes=True
    )
    df_supp.insert(0, 'Ksztalt Okna', 'Asymetryczne QRS + Supresja iglicy')

    return pd.concat([df_sym, df_asym, df_supp], ignore_index=True)


def evaluate_threshold_tuning(
    record_ids: list[str] = None,
    window_range: tuple[int, int] = (-18, 36),
    thresholds: list[float] = None,
    epochs: int = 2,
    batch_size: int = 128,
    return_breakdown: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    if record_ids is None:
        from .config import SAMPLE_RECORDS
        record_ids = SAMPLE_RECORDS

    if thresholds is None:
        thresholds = [0.2, 0.0, -0.2, -0.4, -0.6]

    X, y, groups = build_cnn_dataset_with_groups(record_ids, window_range=window_range)
    n_splits = min(len(set(groups)), 5)
    gkf = GroupKFold(n_splits=n_splits)

    oof_scores = np.zeros(len(y))
    oof_true = np.zeros(len(y))

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train_bal, y_train_bal = balance_training_data(X_train, y_train)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_train_bal, y_train_bal))
            .shuffle(buffer_size=len(y_train_bal))
            .batch(batch_size)
        )
        test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)

        model = build_csnet_binary_model(window_size=X.shape[1])
        model.fit(train_ds, epochs=epochs, verbose=0)

        scores = model.predict(test_ds, verbose=0).flatten()
        oof_scores[test_idx] = scores
        oof_true[test_idx] = y_test

    rows = []
    for th in thresholds:
        y_pred_th = np.where(oof_scores > th, 1, -1)
        m = compute_detailed_metrics(oof_true, y_pred_th)
        prob_equiv = 1.0 / (1.0 + np.exp(-th))
        rows.append({
            'Prog SVM (score)': th,
            'Prawdopodobienstwo (P)': f"{prob_equiv:.2f}",
            'Accuracy': f"{m['accuracy']*100:.2f}%",
            'Balanced Accuracy': f"{m['balanced_accuracy']*100:.2f}%",
            'Czulosc (+1)': f"{m['sensitivity_pos']*100:.2f}%",
            'Specyficznosc (-1)': f"{m['specificity_neg']*100:.2f}%",
            'F1-Score': f"{m['f1_macro']:.4f}",
            'MCC': f"{m['mcc_abs']:.4f}",
        })

    thresh_df = pd.DataFrame(rows)
    if return_breakdown:
        from .evaluation import get_patient_prediction_breakdown
        pred_opt = np.where(oof_scores > -0.2, 1, -1)
        breakdown_df = get_patient_prediction_breakdown(oof_true, pred_opt, groups)
        return thresh_df, breakdown_df

    return thresh_df


def evaluate_de_chazal_benchmark(
    ds1_records: list[str] = None,
    ds2_records: list[str] = None,
    window_range: tuple[int, int] = (-18, 36),
    epochs: int = 3,
    batch_size: int = 128,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from .config import DE_CHAZAL_DS1, DE_CHAZAL_DS2
    from .evaluation import get_patient_prediction_breakdown

    if ds1_records is None:
        ds1_records = DE_CHAZAL_DS1
    if ds2_records is None:
        ds2_records = DE_CHAZAL_DS2

    X_train, y_train, _ = build_cnn_dataset_with_groups(ds1_records, window_range=window_range)
    X_test, y_test, test_groups = build_cnn_dataset_with_groups(ds2_records, window_range=window_range)

    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_train_bal, y_train_bal))
        .shuffle(buffer_size=len(y_train_bal))
        .batch(batch_size)
    )
    test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)

    model = build_csnet_binary_model(window_size=X_train.shape[1])
    model.fit(train_ds, epochs=epochs, verbose=0)

    test_scores = model.predict(test_ds, verbose=0).flatten()
    test_pred = np.where(test_scores > -0.2, 1, -1)

    m = compute_detailed_metrics(y_test, test_pred)
    res_df = pd.DataFrame([{
        'Protokol': 'de Chazal DS1 (Train) -> DS2 (Test)',
        'Liczba pacjentow testowych': len(set(test_groups)),
        'Pobudzenia testowe': len(y_test),
        'Accuracy': f"{m['accuracy']*100:.2f}%",
        'Balanced Accuracy': f"{m['balanced_accuracy']*100:.2f}%",
        'Czulosc (+1)': f"{m['sensitivity_pos']*100:.2f}%",
        'Specyficznosc (-1)': f"{m['specificity_neg']*100:.2f}%",
        'F1-Score': f"{m['f1_macro']:.4f}",
        'MCC': f"{m['mcc_abs']:.4f}",
    }])

    patient_df = get_patient_prediction_breakdown(y_test, test_pred, test_groups)
    return res_df, patient_df
