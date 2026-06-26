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
source .venv/bin/activate
```

或使用 pip（建議在虛擬環境中）：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .                 # 可編輯安裝，會註冊 `hoam` 指令
```

> 註：CUDA 版 PyTorch 由 `pyproject.toml` 的 `[tool.uv.index]` 在 Linux/Windows 上指向 cu128 wheel；macOS 使用 CPU 版。
> 若不想啟用虛擬環境，可把下方 `hoam ...` 改成 `uv run hoam ...`，把 `python -m ...` 改成 `uv run python -m ...`。

## 快速開始

### 1. 準備資料
影像依類別放入 `train/`、`val/` 子資料夾。訓練時 `data.data_dir` 要指向資料根目錄；評估時 `--test-data` 則指向要評估的 class-folder 目錄，例如 `data/val`。

```bash
data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

`train/` 用來訓練與計算 normalization 統計，`val/` 用來 validation / evaluate。每個類別資料夾至少要有影像檔，且評估資料需包含至少兩個類別，才能計算 retrieval / clustering 指標。

### 2. 訓練模型

```bash
# 方法 A：先修改 configs/config.yaml 的 data.data_dir，再使用 console 指令
hoam train

# 方法 B：不修改 config，直接用 Hydra 覆寫參數
python -m hoam.train \
  data.data_dir=/path/to/data \
  training.max_epochs=50 \
  training.lr=5e-4
```

預設會訓練 `HOAM`。若要訓練 `HOAMV2`，建議同步指定 HOAMV2 預設 backbone，方便後續 CLI 載入：

```bash
python -m hoam.train \
  data.data_dir=/path/to/data \
  model.structure=HOAMV2 \
  model.backbone=efficientnetv2_rw_s
```

訓練輸出位於 `training.checkpoint_dir`，預設為 `checkpoints/`：

- `best.pt`、`last.pt`：可供 evaluate / infer 使用的 model state_dict。
- `config_used.yaml`：本次訓練實際使用的設定。
- `mean_std.json`：由訓練資料計算出的 normalization 統計，評估與推論請使用同一份。
- TensorBoard 紀錄位於 `logs/`。

可用 `configs/config.yaml` 的 `experiment.seed` 控制可重現性。

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
- `--test-data` 必須是 class-folder 目錄，例如 `data/val`，其下一層才是各類別資料夾。
- 若訓練的是 HOAMV2，請把 `--model-structure` 改成 `HOAMV2`。
- 輸出：`eval_out/test_metrics.json`、`test_metrics.csv`（retrieval + 分類 scalar）、`classification_report.csv`（每類別 P/R/F1）、`confusion_matrix.csv`。

### 4. KNN 推論

KNN 模式需要先建立 reference index。訓練時打開 `knn.enable=true` 後，流程會用 train set 建立 `knn.index` 與 `dataset.pkl`：

```bash
python -m hoam.train \
  data.data_dir=/path/to/data \
  knn.enable=true
```

完成後使用同一個 checkpoint 目錄中的 `best.pt`、`mean_std.json`、`knn.index`、`dataset.pkl`：

```bash
hoam infer \
  --mode knn \
  --model-structure HOAM \
  --model-path checkpoints/best.pt \
  --mean-std-file checkpoints/mean_std.json \
  --dataset-pkl checkpoints/dataset.pkl \
  --faiss-index checkpoints/knn.index \
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
  --mean-std-file checkpoints/mean_std.json \
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

> `hoam infer` 是 pass-through wrapper；完整 inference 參數請用 `python -m hoam.inference --help` 查看。

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
