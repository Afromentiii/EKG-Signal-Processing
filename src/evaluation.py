import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_detailed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average='macro', zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    mcc_abs = float(abs(matthews_corrcoef(y_true, y_pred)))

    sens_positive = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    spec_negative = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'sensitivity_pos': sens_positive,
        'specificity_neg': spec_negative,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'mcc_abs': mcc_abs,
        'confusion_matrix': cm,
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Macierz pomylek (Confusion Matrix)",
    labels: tuple[str, str] = ('Norma (-1)', 'Arytmia / Stym (+1)'),
    figsize: tuple[int, int] = (6, 5),
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[-1, 1])
    total = np.sum(cm)

    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = (count / total * 100.0) if total > 0 else 0.0
            color = 'darkgreen' if i == j else 'darkred'
            ax.text(
                j,
                i,
                f"{count}\n({pct:.1f}%)",
                ha='center',
                va='center',
                color=color,
                fontsize=11,
                fontweight='bold',
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Przewidywana klasa (Predicted)', fontweight='bold')
    ax.set_ylabel('Rzeczywista klasa (True Target)', fontweight='bold')
    ax.set_title(title, pad=15, fontweight='bold')

    plt.colorbar(cax)
    plt.tight_layout()
    plt.show()


def get_patient_prediction_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
) -> pd.DataFrame:
    rows = []
    unique_groups = np.unique(groups)
    for g in sorted(unique_groups):
        mask = (groups == g)
        y_t = y_true[mask]
        y_p = y_pred[mask]
        m = compute_detailed_metrics(y_t, y_p)
        cm = m['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        n_pos = int(np.sum(y_t == 1))
        n_neg = int(np.sum(y_t == -1))
        rows.append({
            'Pacjent': str(g),
            'Pobudzenia': len(y_t),
            'Norma (-1)': n_neg,
            'Arytmia (+1)': n_pos,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            'Czulosc (+1)': f"{m['sensitivity_pos']*100:.1f}%",
            'Specyficznosc (-1)': f"{m['specificity_neg']*100:.1f}%",
            'Balanced Accuracy': f"{m['balanced_accuracy']*100:.1f}%",
            'MCC': f"{m['mcc_abs']:.4f}",
        })
    return pd.DataFrame(rows)
