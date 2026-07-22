# Churn Prediction Benchmark — Revised Pipeline

Bộ code này dựng lại toàn bộ thực nghiệm cho bài báo *"Benchmarking Deep
Learning Models for Predicting Customer Churn in Banking with Explainable
AI"* theo đúng góp ý của Reviewer #2, #3, #5 (Major Revision, Neural
Computing and Applications).

**Dataset dùng:** `data/churn.csv` — bộ 14 cột, 10.000 dòng, đúng như mô tả
trong bản thảo và mục Data Availability (Kaggle: `mathchi/churn-for-bank-customers`).
Bộ này **không có** biến `Complain` — nếu bạn thấy tài liệu nào nhắc tới
`Complain` là đang nói tới một bộ Kaggle khác, không dùng ở đây.

---

## 1. Cài đặt (Windows/Mac/Linux + VS Code)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

> ⚠️ `torch` + `pytorch-tabnet` khá nặng (vài trăm MB). Nếu máy yếu hoặc
> mạng chậm, có thể chạy trước với `--skip-deep` (xem bên dưới) để có ngay
> kết quả baseline ML, rồi cài thêm deep learning sau.

## 2. Chạy nhanh để kiểm tra mọi thứ hoạt động (~5-10 phút)

```bash
python main.py --quick --skip-deep
```

Lệnh này chạy 1 seed, hyperparameter search nhỏ, **bỏ qua MLP/TabNet/TabR**
— dùng để chắc chắn code không lỗi trước khi chạy full.

## 3. Chạy đầy đủ cho kết quả cuối cùng nộp bài (vài giờ, tuỳ máy)

```bash
python main.py
```

Chạy: 5 seed × 5-fold = 25 lần đánh giá cho mỗi model, đầy đủ MLP + TabNet +
TabR + 5 baseline ML, tuning, SHAP, ablation, kiểm định thống kê.

Nếu muốn bỏ qua deep learning (chỉ có baseline ML):
```bash
python main.py --skip-deep
```

## 4. Minh chứng leakage trước/sau sửa (dùng cho response letter)

```bash
python leakage_demo.py
```

So sánh trực tiếp: pipeline lỗi (SMOTE trước khi chia train/test) vs
pipeline đã sửa (SMOTE chỉ trên train). Kết quả lưu ở
`outputs/tables/leakage_before_after.csv` — nên trích trực tiếp bảng này
vào response letter khi trả lời Reviewer 3 & 5.

## 5. Cấu trúc project

```
churn-benchmark/
├── data/churn.csv                  # dataset gốc
├── config.py                       # chỉnh seed, hyperparameter, đường dẫn ở đây
├── main.py                         # chạy toàn bộ pipeline
├── leakage_demo.py                 # minh chứng before/after leakage
├── requirements.txt
├── src/
│   ├── data_pipeline.py            # load data + leakage audit + preprocessing
│   ├── baseline_models.py          # LogReg, RF, XGBoost, LightGBM, CatBoost
│   ├── deep_models.py              # MLP, TabNet, TabR (PyTorch)
│   ├── evaluate.py                 # metrics, repeated CV, kiểm định thống kê
│   └── shap_analysis.py            # SHAP cho tất cả model + đo độ ổn định
└── outputs/
    ├── tables/                     # mọi bảng số liệu (CSV) — copy thẳng vào Word
    └── figures/                    # biểu đồ SHAP (PNG)
```

## 6. Đầu ra map trực tiếp vào bản thảo

| File output | Dùng cho phần nào của bài |
|---|---|
| `table_leakage_audit.csv` | Bảng 3.x — Leakage Audit and Feature Availability Assessment |
| `leakage_before_after.csv` | Minh chứng trong response letter + có thể thêm vào Discussion |
| `table_baseline_cv_summary.csv`, `table_deep_cv_summary.csv`, `table_ALL_MODELS_summary.csv` | Bảng kết quả chính (Results), thay thế Bảng 2 cũ |
| `table_pairwise_wilcoxon.csv`, `friedman_test.json` | Phần kiểm định thống kê so sánh model |
| `table_imbalance_ablation_summary.csv` | Ablation study — so sánh none/class_weight/SMOTE |
| `shap_importance_*.csv`, `shap_*.png` | Phần Explainability, cho từng model |
| `baseline_best_hyperparameters.json` | Bảng Hyperparameter tuning trong Methodology (mục tái lập) |

## 7. Ghi chú quan trọng khi viết lại bản thảo

- **Bỏ hoàn toàn** mọi đoạn văn/hình về `Complain_1` — biến này không tồn
  tại trong dataset 14 cột thật sự dùng trong bài.
- Model TabR ở đây là **bản tự cài đặt đơn giản hoá** (encoder + attention
  retrieval), không phải code gốc của tác giả TabR — cần ghi rõ điều này
  trong Methods để tránh hiểu nhầm giống lỗi "RealTabR" cũ mà Reviewer 3
  đã chỉ ra.
- Mọi con số trong bảng đều có mean ± std qua 25 lần đánh giá (5 seed × 5
  fold) — nhớ báo cáo cả hai, không chỉ giá trị trung bình.
- Trước khi tuyên bố model nào "tốt nhất", luôn kiểm tra
  `table_pairwise_wilcoxon.csv` xem khác biệt có ý nghĩa thống kê
  (`significant_at_0.05`) hay không.

## 8. Nếu gặp lỗi khi chạy

- `ImportError: torch` / `pytorch_tabnet` → chưa cài, chạy lại
  `pip install -r requirements.txt`, hoặc dùng `--skip-deep` tạm thời.
- Chạy chậm → giảm `SEEDS` trong `config.py` xuống `[42]` hoặc giảm
  `BASELINE_SEARCH_ITER`.
- Muốn đổi kiến trúc MLP/TabNet/TabR → sửa trực tiếp trong `config.py`
  (mục `MLP_PARAMS`, `TABNET_PARAMS`, `TABR_PARAMS`).
