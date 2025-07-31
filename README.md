# Acoustic‑Anomaly Detection – **Without fine tunning** demo

*Model: Efficient Audio Transformer (EAT)  •  Distance detector: k‑means NN*

---

## 1  Repository layout

```text
├── dataset/                     # 16‑kHz WAVs in the official DCASE folder layout
│   ├── fan/
│   │   ├── train/               # 1000 wavs normals
│   │   └── test/                # 200 wavs   (normal + anomaly)
│   └── bearing/ …               # second machine type
├── model_weights/
│   └── EAT-base_epoch30_pt.pt   # original PyTorch checkpoint (depth=12)
├── eat_base_export.onnx         # **generated** ONNX export – see step 3
│
├── main.py                      # end‑to‑end pipeline – train ➟ score ➟ metrics
├── metrics.py                   # AUC / pAUC calculation + .csv + ROC plot
├── load_dataset.py              # lightweight loader that returns lists of (path, wav)
└── example_model_onnx_conversion.py   #  script that exports the PT ckpt to ONNX
```

---

## 2  Quick start (⏱ 3 commands)

- Download and extract the dataset.
- Download the EAT checkpoint.
Then:

```bash

# 1 one‑time: export the EAT checkpoint to ONNX  (≈30 s CPU) 
python example_model_onnx_conversion.py

# 2 train on the 1 000 source‑normal clips and score the 200‑file dev set
python main.py
```

Outputs land in `./results_fan/`:

| file          | content                                |
| ------------- | -------------------------------------- |
| `scores.csv`  | `path,score,label` for every test clip |
| `metrics.txt` | `AUC` and `pAUC@10 % FPR`              |
| `roc.png`     | ROC curve with the 10 % FPR line       |

---

## 3  How the pipeline works (`main.py`)

```text
WAV ➟ resample(16 kHz) ➟ frame ➟ magnitude FFT ➟ 128‑mel filter‑bank
    ➟ LinearScaler (match EAT training stats)
    ➟ ONNXFeatureExtractor (EAT‑base)           -> (B, L, 768)
       • single mel‑channel is replicated → 3
       • mean‑pool over patches         → (B, 768)
    ➟ StandardScaler
    ➟ KMeansNN  (nearest‑centroid distance, k = n_components)
    ➟ MinMaxScaler  (maps train‑distance range → [0,1])
    → anomaly score  (0 ≈ normal, 1 ≈ far from any centroid)
```

* **`get_deep_feat_cls_eat_all_model()`** builds the entire scikit‑learn
  `Pipeline` and swaps in the ONNX file you exported.
* `metrics.compute_and_save_metrics()` calculates ROC‑AUC and pAUC
  (partial AUC up to 10 % FPR) and saves artefacts.

---

## 4  ONNX export details (`example_model_onnx_conversion.py`)

```python
depth      = 12             # full EAT‑base

dynamic_axes = {"input":  {0: "batch", 3: "time"},
                "emb":    {0: "batch", 1: "patch"}}
```

* Height 128 and channels 3 are **fixed**; width (time) is dynamic.
* Output tensor `"emb"` is `(B, patch, 768)`; the pipeline does a
  `mean(axis=1)` to obtain one embedding per clip.

---



