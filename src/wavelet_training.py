import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
import tensorflow as tf

from .evaluation import compute_detailed_metrics
from .modwt import build_modwt_dataset_with_groups
from .preprocessing import balance_training_data
from .wavelet_models import build_wavelet_net


def evaluate_wavelet_net_group_kfold(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    wavelet_name: str = 'db4',
    use_blstm: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    n_splits: int = None,
    epochs: int = 3,
    batch_size: int = 64,
    l2_reg: float = 0.0005,
    learning_rate: float = 0.001,
    balance_classes: bool = True,
    verbose: int = 0,
) -> tuple[pd.DataFrame, dict, tf.keras.Model, tuple[np.ndarray, np.ndarray]]:
    unique_groups = np.unique(groups)
    if n_splits is None:
        n_splits = len(unique_groups)

    gkf = GroupKFold(n_splits=n_splits)
    metrics_records = []
    all_y_true, all_y_pred = [], []
    last_model = None

    for train_idx, test_idx in gkf.split(x, y, groups=groups):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        if balance_classes:
            x_train, y_train = balance_training_data(x_train, y_train)

        y_train_bin = np.where(y_train == 1, 1, 0)

        train_ds = (
            tf.data.Dataset.from_tensor_slices((x_train, y_train_bin))
            .shuffle(buffer_size=len(y_train_bin))
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = (
            tf.data.Dataset.from_tensor_slices(x_test)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        model = build_wavelet_net(
            input_shape=x.shape[1:],
            use_blstm=use_blstm,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )

        model.fit(train_ds, epochs=epochs, verbose=verbose)
        last_model = model

        probs = model.predict(test_ds, verbose=0).flatten()
        y_pred = np.where(probs >= 0.5, 1, -1)

        m = compute_detailed_metrics(y_test, y_pred)
        metrics_records.append(m)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

    df_m = pd.DataFrame(metrics_records)
    model_desc = 'WaveletNet (MODWT + 1D CNN)' if not use_blstm else 'WaveletNet (MODWT + BLSTM)'
    if use_focal_loss:
        model_desc += ' + Focal Loss'

    summary_df = pd.DataFrame([{
        'Falka (Wavelet)': wavelet_name.upper(),
        'Model': model_desc,
        'Liczba Foldow': n_splits,
        'Accuracy': f"{df_m['accuracy'].mean():.4f} (+/- {df_m['accuracy'].std():.4f})",
        'Precision (macro)': f"{df_m['precision_macro'].mean():.4f} (+/- {df_m['precision_macro'].std():.4f})",
        'Recall (macro)': f"{df_m['recall_macro'].mean():.4f} (+/- {df_m['recall_macro'].std():.4f})",
        'F1-Score (macro)': f"{df_m['f1_macro'].mean():.4f} (+/- {df_m['f1_macro'].std():.4f})",
        'MCC (Absolute)': f"{df_m['mcc_abs'].mean():.4f} (+/- {df_m['mcc_abs'].std():.4f})",
    }])

    return summary_df, df_m.to_dict(orient='list'), last_model, (np.array(all_y_true), np.array(all_y_pred))


def compare_modwt_wavelets(
    record_ids: list[str] = None,
    wavelets: tuple[str, str] = ('db4', 'sym4'),
    level: int = 3,
    window_size: int = 180,
    use_blstm: bool = False,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    epochs: int = 3,
    batch_size: int = 64,
    max_beats_per_record: int = None,
    pn_dir: str = 'mitdb',
    verbose: int = 0,
) -> tuple[pd.DataFrame, dict]:
    if record_ids is None:
        record_ids = ['100', '106', '109', '119']

    summary_rows = []
    artifacts = {}

    for w in wavelets:
        x, y, groups = build_modwt_dataset_with_groups(
            record_ids=record_ids,
            wavelet=w,
            level=level,
            window_size=window_size,
            max_beats_per_record=max_beats_per_record,
            pn_dir=pn_dir,
        )

        res_df, fold_dict, model, (y_true, y_pred) = evaluate_wavelet_net_group_kfold(
            x,
            y,
            groups,
            wavelet_name=w,
            use_blstm=use_blstm,
            use_focal_loss=use_focal_loss,
            focal_gamma=focal_gamma,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        summary_rows.append(res_df.iloc[0])

        artifacts[w] = {
            'model': model,
            'y_true': y_true,
            'y_pred': y_pred,
            'fold_metrics': fold_dict,
            'x': x,
            'y': y,
        }

    comparison_df = pd.DataFrame(summary_rows).reset_index(drop=True)
    return comparison_df, artifacts


def plot_wavelet_net_confusion_matrices(
    artifacts: dict,
    figsize: tuple[int, int] = (10, 4),
) -> None:
    fig, axes = plt.subplots(1, len(artifacts), figsize=figsize)
    if len(artifacts) == 1:
        axes = [axes]

    labels = ('Norma (-1)', 'Arytmia (+1)')

    for ax, (w_name, data) in zip(axes, artifacts.items()):
        y_true = data['y_true']
        y_pred = data['y_pred']
        cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
        total = np.sum(cm)

        ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(2):
            for j in range(2):
                cnt = cm[i, j]
                pct = (cnt / total * 100.0) if total > 0 else 0.0
                col = 'darkgreen' if i == j else 'darkred'
                ax.text(
                    j, i, f"{cnt}\n({pct:.1f}%)",
                    ha='center', va='center', color=col, fontsize=10, fontweight='bold'
                )

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Przewidywana klasa', fontweight='bold')
        ax.set_ylabel('Rzeczywista klasa', fontweight='bold')
        ax.set_title(f'Macierz pomylek - MODWT: {w_name.upper()}', pad=12, fontweight='bold')

    plt.tight_layout()
    plt.show()
