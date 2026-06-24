# HOAM Project

論文模型 HOAM & HOAMV2 之 PyTorch 實作（SMT 元件極性分類）。結合 SubCenter ArcFace、Triplet、Center loss（HybridMarginLoss），以及 Orthogonal Fusion、Learnable Edge Layer 等元件，並提供訓練、評估、KNN/Match 推論功能。

## 目錄
- [環境安裝](#環境安裝)
- [快速開始](#快速開始)
- [命令行介面 (CLI)](#命令行介面-cli)
- [專案架構](#專案架構)

## 環境安裝

需要 Python >= 3.12。建議用 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/CJDaniel96/hoam.git
cd hoam
uv sync                 # 建立 .venv 並安裝相依套件 + 本套件（含 `hoam` 指令）
```

或使用 pip（建議在虛擬環境中）：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # 可編輯安裝，會註冊 `hoam` 指令
```

> 註：CUDA 版 PyTorch 由 `pyproject.toml` 的 `[tool.uv.index]` 在 Linux/Windows 上指向 cu128 wheel；macOS 使用 CPU 版。

## 快速開始

### 1. 準備資料
影像依類別放入 `train/`、`val/` 子資料夾，並把 `configs/config.yaml` 的 `data.data_dir` 指向資料根目錄：

```bash
data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

### 2. 訓練模型

```bash
# 使用 configs/config.yaml 預設值（console 指令）
hoam train

# 以 Hydra 覆寫參數（透過模組進入點）
python -m hoam.train data.data_dir=/path/to/data training.max_epochs=50 training.lr=5e-4
```

訓練輸出：`checkpoints/best.pt`、`checkpoints/last.pt`（model state_dict）、`config_used.yaml`、`mean_std.json`，TensorBoard 紀錄於 `logs/`。可用 `configs/config.yaml` 的 `experiment.seed` 控制可重現性。

### 3. 評估模型

```bash
hoam evaluate \
  --model-path checkpoints/best.pt \
  --model-structure HOAM \
  --test-data /path/to/val \
  --mean-std-file checkpoints/mean_std.json \
  --save-dir eval_out \
  --batch-size 64
```

- `--model-path` 是訓練匯出的 `best.pt`（state_dict，非 `.ckpt`）。
- `--mean-std-file` 建議指向訓練時存下的 `mean_std.json`，以重現相同正規化。
- 輸出：`eval_out/test_metrics.json`、`test_metrics.csv`（retrieval + 分類 scalar）、`classification_report.csv`（每類別 P/R/F1）、`confusion_matrix.csv`。

### 4. KNN 推論

```bash
hoam infer \
  --mode knn \
  --model-structure HOAM \
  --model-path checkpoints/best.pt \
  --dataset-pkl dataset.pkl \
  --data imgs/ \
  --save-dir knn_out/ \
  --k 5
```

輸出 `knn_out/top1.json` 與 Top-1 圖片。

### 5. 匹配推論

```bash
hoam infer \
  --mode match \
  --model-structure HOAM \
  --model-path checkpoints/best.pt \
  --query-image query.jpg \
  --data imgs/ \
  --save-dir match_out/ \
  --threshold 0.8
```

圖片依「OK」/「NG」放入對應子資料夾。

## 命令行介面 (CLI)

安裝後提供 `hoam` 指令：

| 指令 | 說明 |
| --- | --- |
| `hoam train` | 以 Hydra config + PyTorch Lightning 訓練（`--config-dir`、`--config-name`） |
| `hoam evaluate` | 在測試集上計算 retrieval 與分類指標 |
| `hoam infer` | KNN / Match 推論 |

`hoam <cmd> --help` 可查看各指令選項。

## 專案架構

```bash
hoam/
├── configs/
│   └── config.yaml          # Hydra config（experiment/data/model/loss/training/knn）
├── src/hoam/
│   ├── cli.py               # `hoam` CLI：train / evaluate / infer
│   ├── train.py             # Lightning 訓練（DataModule + LightningModule + Trainer）
│   ├── evaluate.py          # retrieval + 分類指標
│   ├── inference.py         # KNN / Match 推論
│   ├── utils.py             # load_model、set_seed、UnNormalize
│   ├── metrics.py           # k-NN 分類指標（accuracy / P / R / F1 / confusion）
│   ├── models/              # hoam.py（HOAM/HOAMV2）、base.py（building blocks）
│   ├── losses/              # hybrid_margin.py（HybridMarginLoss）
│   └── data/                # transforms.py、statistics.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
