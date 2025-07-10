# HOAM Project
 
論文模型 HOAM & HOAMV2 之 PyTorch 實作，結合 ArcFace, GeM, Laplacian 等元件，並提供訓練、推論、評估與 KNN 功能。
 
## 目錄
- [環境安裝](#環境安裝)
- [快速開始](#快速開始)
- [命令行介面 (CLI)](#命令行介面-cli)
- [專案架構](#專案架構)
 
## 環境安裝
```bash
git clone https://github.com/CJDaniel96/hoam.git
cd hoam
python -m venv venv
source venv/bin/activate    # Linux & MacOS 
venv\Scripts\activate       # Windows

# 建議使用 virtualenv 或 conda
pip install -r requirements.txt
```

## 快速開始

1. 準備資料
- 將影像資料按類別放入 `train/`、`val/` 資料夾，結構如下：
- 修改 `configs/config.yaml` 中的 `data.data_dir` 路徑指向 `data/`
```bash
data/
├── train/
│   ├── class1/
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

2. 訓練模型
```bash
# 使用預設配置
python -m src.hoam.train
# 覆寫配置參數範例
python -m src.hoam.train data.data_dir="/path/to/data" training.epochs=50 training.lr=5e-4
```

3. 評估模型
```bash
hoam evaluate \
  --model-path checkpoints/best.ckpt \
  --test-data /path/to/val \
  --save-dir eval_out \
  --batch-size 64
```

- 結果保存在 `eval_out/test_metrics.json` 與 `eval_out/test_metrics.csv`

4. KNN 推論
```bash
hoam infer \
  --mode knn \
  --model-structure HOAM \
  --model-path checkpoints/best.ckpt \
  --dataset-pkl dataset.pkl \
  --data imgs/ \
  --save-dir knn_out/ \
  --k 5
```

- 輸出 `knn_out/top1.json` 以及 Top-1 圖片於子資料夾中

5. 匹配推論
```bash
hoam infer \
  --mode match \
  --model-structure HOAM \
  --model-path checkpoints/best.ckpt \
  --query-image query.jpg \
  --data imgs/ \
  --save-dir match_out/ \
  --threshold 0.8
```

- 圖片將依「OK」或「NG」放入對應子資料夾

## 專案架構
```bash
hoam/
├── configs/           # Hydra 配置檔 (config.yaml)
├── src/hoam/          # 原始碼
│   ├── cli.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   ├── models/
│   ├── losses/
│   └── data/
├── requirements.txt
├── pyproject.toml
└── README.md
```