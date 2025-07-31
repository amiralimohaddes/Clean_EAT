#!/usr/bin/env python3
# fan_anomaly_eat.py
from pathlib import Path
from load_dataset import load_data
import sys

# ready-made helper
from to_install.unsupervised_anomalie_detection.sklearn_pipes import get_deep_feat_cls_eat_all_model




# 1) load data -------------------------------------------------------------
root      = Path("./dataset")
data      = load_data(root)

print("Train (normal):", len(data["bearing"]["train"]))
print("Test  (normal and anomalous):", len(data["bearing"]["test"]))

X_train   = [wav for _, wav in data["bearing"]["train"] if wav is not None]


# 2) build pipeline --------------------------------------------------------
pipe, _   = get_deep_feat_cls_eat_all_model(
                feat_model_type   = "EAT",   # <- real Efficient Audio Transformer
                feat_model_depth  = 12,      # matches your checkpoint depth
                n_components      = 10,       # K-means clusters / PCA dims / GMM comps
                model_type        = "kmeansNN"   # or 'pca_recon' / 'GMMpb'
            )



# 3) train & score ---------------------------------------------------------
print(f"Fitting on {len(X_train)} healthy bearing clips …")
print("Progress: [", end="", flush=True)
pipe.fit(X_train)
print("##########] done.")

# ----- 1. keep only usable clips *and* remember their paths -------------
test_pairs = [(p, wav) for p, wav in data["bearing"]["test"] if wav is not None]

paths_test = [p   for p, _ in test_pairs]   # length N
X_test     = [wav for _, wav in test_pairs] # length N

# sanity-check
assert len(paths_test) == len(X_test), "path/clip count mismatch"

print(f"Generating anomaly scores for {len(X_test)} test clips …")
print("Progress: [", end="", flush=True)
    
scores = pipe.transform(X_test).ravel()

assert len(scores) == len(paths_test)
print("##########] done.")

print("Anomaly scores:\n", scores)

from metrics import compute_and_save_metrics
out_dir = Path("./results_bearing")           # or any folder you like
metrics = compute_and_save_metrics(
    paths=paths_test,
    scores=scores,
    out_dir=out_dir,
)

print(f"\nAUC  = {metrics['auc']:.4f}")
print(f"pAUC = {metrics['pauc']:.4f} (saved to {out_dir})")
