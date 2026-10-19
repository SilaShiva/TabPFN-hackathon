"""thalscreen: CBC-only beta-thalassaemia carrier screening benchmark.

Compares TabPFN-3.5 against the published CBC discrimination indices and
modern ML baselines on 12,339 real patients (Kaggle HPLC screening data).
"""
from .dataset import load_dataset, published_indices, FEATURES, CBC_FEATURES
from .baselines import index_auc, oof_predict, MODELS, OOFResult
from .evaluate import recall_at_specificity, operating_table, learning_curve

__version__ = "0.1.0"
__all__ = [
    "load_dataset", "published_indices", "FEATURES", "CBC_FEATURES",
    "index_auc", "oof_predict", "MODELS", "OOFResult",
    "recall_at_specificity", "operating_table", "learning_curve",
]
