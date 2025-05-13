from ignite.contrib.metrics import AveragePrecision, ROC_AUC, RocCurve, PrecisionRecallCurve
from ignite.metrics import Accuracy, RootMeanSquaredError
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    mean_absolute_error,
    precision_recall_curve,
    roc_curve,
    r2_score,
    mean_squared_error,
    precision_score, 
    recall_score,
)
from torchmetrics.classification import (
    AUROC,
    AveragePrecision as TorchMetricsAveragePrecision,
    PrecisionRecallCurve as TorchMetricsPrecisionRecallCurve,
    CalibrationError,
    F1Score,
)
from enum import Enum
from icu_benchmarks.models.custom_metrics import (
    CalibrationCurve,
    BalancedAccuracy,
    MAE,
    MAPE,
    JSD,
    BinaryFairnessWrapper,
)
import numpy as np


def compute_precision(y_true, y_score, threshold=0.5):
    """
    Compute Precision at a given threshold.

    Parameters:
    y_true (array-like): True binary labels (0 or 1).
    y_score (array-like): Predicted scores (probabilities or confidence scores).
    threshold (float, optional): Threshold to convert scores into binary predictions. Default is 0.5.

    Returns:
    float: Precision score.
    """
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return precision_score(y_true, y_pred)

def compute_recall(y_true, y_score, threshold=0.5):
    """
    Compute Recall at a given threshold.

    Parameters:
    y_true (array-like): True binary labels (0 or 1).
    y_score (array-like): Predicted scores (probabilities or confidence scores).
    threshold (float, optional): Threshold to convert scores into binary predictions. Default is 0.5.

    Returns:
    float: Recall score.
    """
    y_pred = (np.array(y_score) >= threshold).astype(int)
    return recall_score(y_true, y_pred)


class MLMetrics:
    BINARY_CLASSIFICATION = {
        "AUC": roc_auc_score,
        "Calibration_Curve": calibration_curve,
        "AUPRC": average_precision_score,
        "PR_Curve": precision_recall_curve,
        "RO_Curve": roc_curve,
        "Precision": compute_precision,
        "Recall": compute_recall,
    }

    MULTICLASS_CLASSIFICATION = {
        "Accuracy": accuracy_score,
        "AUC": roc_auc_score,
        "Balanced_Accuracy": balanced_accuracy_score,
        "AUPRC": average_precision_score,
    }

    REGRESSION = {
        "MAE": mean_absolute_error,
        "R2": r2_score,
        "RMSE": mean_squared_error,
    }


# TODO: add support for confusion matrix
class DLMetrics:
    BINARY_CLASSIFICATION = {
        "AUC": ROC_AUC,
        "Calibration_Curve": CalibrationCurve,
        "PR": AveragePrecision,
        "PR_Curve": PrecisionRecallCurve,
        "RO_Curve": RocCurve,
    }

    BINARY_CLASSIFICATION_TORCHMETRICS = {
        "AUC": AUROC(task="binary"),
        "AUPRC": TorchMetricsAveragePrecision(task="binary"),
        "PrecisionRecallCurve": TorchMetricsPrecisionRecallCurve(task="binary"),
        "Calibration_Error": CalibrationError(task="binary", n_bins=10),
        "F1": F1Score(task="binary", num_classes=2),
        "Binary_Fairness": BinaryFairnessWrapper(num_groups=2, task="demographic_parity", group_name="sex"),
    }

    MULTICLASS_CLASSIFICATION = {
        "Accuracy": Accuracy,
        "BalancedAccuracy": BalancedAccuracy,
    }

    REGRESSION = {
        "MAE": MAE,
    }

    IMPUTATION = {
        "rmse": RootMeanSquaredError,
        "mae": MAE,
        "jsd": JSD,
        "mape": MAPE
    }


class ImputationInit(str, Enum):
    """Type of initialization to use for the imputation model."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    XAVIER = "xavier"
    KAIMING = "kaiming"
    ORTHOGONAL = "orthogonal"
