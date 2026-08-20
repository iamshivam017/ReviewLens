# 🔍 ReviewLens — Sentiment Analysis & Insight Dashboard

> **Capstone Project — B.Tech Data Science & AI**  
> Classifies real customer reviews as positive, negative, or neutral using
> TF-IDF + Logistic Regression, trained on **real Kaggle review data**.

---

## 📌 Problem Statement

Businesses receive thousands of product/service reviews every day across
multiple platforms. Manually reading and categorising them is impossible
at scale. ReviewLens automates sentiment classification and surfaces
actionable insights — what customers love, what they hate, and why.

---

## 🌍 Real-World Use Case

| Industry | Application |
|---|---|
| E-commerce | Classify product reviews from Amazon / Flipkart |
| EdTech | Analyse course feedback to improve content |
| Mobile / SaaS | Detect negative app-store reviews before they go viral |
| Restaurants | Monitor sentiment across Zomato / survey channels |

---

## 📊 Dataset — Real Data, Not Synthetic

ReviewLens is trained on **~30,800 real reviews**, merged from **13 Kaggle
datasets** covering 6 categories and 8 platforms:

| Category | Platforms |
|---|---|
| Electronics | Amazon |
| App | App Store, Google Play Store |
| E-commerce | Flipkart, generic e-commerce sites |
| Restaurant | Zomato, customer surveys |
| Course | Coursera |
| Movie | IMDB |

Every source file has a different schema (different column names, ratings
vs. star-text vs. binary "liked" flags). `src/data_sources.py` maps each
one onto a single unified schema, and `src/data_loader.py` merges them —
see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full pipeline.

**The processed dataset (`data/raw/reviews.csv`, ~11MB) ships with this
repo**, so `train.py` works immediately — no need to re-download anything.
The much larger raw Kaggle CSVs (~250MB) are *not* included; see
[Regenerating the Dataset](#-regenerating-the-dataset-optional) below if
you want to rebuild it from scratch or add new sources.

---

## ✨ Dashboard Features

The Streamlit dashboard has 8 pages, each one file in `src/pages/`:

| Page | What it does |
|---|---|
| 📊 **Overview** | Snapshot of the active dataset (incl. date range, avg. confidence) and the trained model |
| 🤖 **Predict Sentiment** | Paste one review → instant sentiment + confidence + keywords |
| 🔍 **Review Explorer** | Upload **any** CSV, confirm the auto-detected column mapping, then search/filter/browse |
| 📈 **Trends & Charts** | Sentiment distribution, top keywords, word clouds, time trend, confidence distribution, rating-vs-sentiment mismatch detector |
| ⚖️ **Product Comparison** | Compare sentiment across categories and sources |
| 📄 **Summary** | Plain-language auto-summary — sentiment verdict, top praise/complaints, best/worst category, and whether sentiment is improving or declining over time |
| 🧠 **Model Performance** | Accuracy, F1, tuned hyperparameters, confusion matrix, per-class precision/recall/F1, top influential words |
| 📥 **Export Reports** | Download the annotated dataset and a text summary report |

Whatever CSV you upload in **Review Explorer** becomes the "active
dataset" for every other page — Trends, Comparison, Summary, and Export
all update automatically. Before you upload anything, the dashboard shows
a live sample of the real training data so it's never empty.

### Flexible CSV upload

The uploader does **not** require an exact schema. Any valid CSV works,
regardless of column names, column count, or row count:

- **Column auto-detection:** `src/csv_mapper.py` scans the uploaded
  file's column names and guesses which one holds the review text,
  rating, date, source, and category — using keyword matching against
  common naming patterns (`review`, `comment`, `feedback`, `stars`,
  `score`, `platform`, `purchased`, …). If no column name is
  recognisable, it falls back to picking the text-heaviest column
  (longest average string length).
- **Data-type validation, not just name matching:** the selected
  columns are checked against their *actual content* — a column
  guessed as "rating" is validated to actually be mostly numeric, a
  "date" column is validated to actually parse as dates, before
  analysis runs. A clear ✅/⚠️ line is shown for each mapped field.
- **Every guess is shown before analysis runs**, in dropdowns the user
  can correct in one click if it's wrong.
- **Robust error handling** for real-world failure modes: empty files,
  files with headers but no rows, non-UTF-8 encoding (common from
  Excel exports), malformed/corrupted CSVs, and files with no
  detectable text column all produce a specific, actionable message —
  never a raw crash.
- **Upload limits**, always shown in-app before uploading:
  - Max file size: **50 MB**
  - Hard row cap: **150,000 rows** (rejected with a clear message above this)
  - Recommended: under **20,000 rows** for the fastest results — larger
    files still work, processed automatically in chunks with a live
    progress bar
- Unmapped optional fields (rating/date/source/category) are simply
  left out — the dashboard's charts already adapt to missing columns.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| ML Model | scikit-learn — TF-IDF + Logistic Regression |
| Baseline | Multinomial Naive Bayes |
| NLP | NLTK (stopwords, lemmatization) |
| Dashboard | Streamlit (multi-page) |
| Charts | Plotly, Matplotlib |
| Word Cloud | wordcloud |
| Serialization | joblib |
| Testing | pytest + Streamlit AppTest |

---

## 📁 Folder Structure

```
reviewlens/
│
├── app.py                    # Streamlit entry point (thin router)
├── train.py                  # Model training script
├── prepare_dataset.py        # Builds data/raw/reviews.csv from Kaggle sources
├── ReviewLens_Notebook.ipynb # Full pipeline walkthrough — EDA → training → explainability
├── build_notebook.py         # Script that generates the notebook above
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── reviews.csv       # Unified real dataset (ships with repo)
│   ├── processed/
│   └── kaggle_raw/           # Raw Kaggle CSVs (not shipped — see below)
│
├── models/
│   ├── pipeline.pkl          # Trained sklearn Pipeline
│   └── metadata.json         # Training metrics, confusion matrix, top features
│
├── src/
│   ├── preprocess.py         # Text cleaning pipeline
│   ├── features.py           # TF-IDF utilities
│   ├── train_model.py        # Pipeline factories + evaluation
│   ├── evaluate.py           # Metrics helpers
│   ├── explain.py            # Coefficient-based explainability
│   ├── utils.py              # Paths, constants, helpers
│   ├── data_sources.py       # Config: which Kaggle CSV maps to which schema
│   ├── data_loader.py        # Generic loader that merges all sources
│   ├── csv_mapper.py         # Auto-detects columns in any uploaded CSV
│   ├── dashboard_utils.py    # Shared prediction/chart helpers for the app
│   └── pages/                # One file per dashboard sidebar item
│       ├── overview.py
│       ├── predict.py
│       ├── explorer.py
│       ├── trends.py
│       ├── comparison.py
│       ├── summary.py
│       ├── performance.py
│       └── export.py
│
├── tests/
│   ├── test_preprocess.py
│   ├── test_data_loader.py
│   ├── test_csv_mapper.py
│   ├── test_model.py
│   ├── test_app.py
│   └── test_dashboard.py
│
└── docs/
    ├── PROJECT_REPORT.md
    └── ARCHITECTURE.md
```

---

## ⚙️ Installation

```bash
git clone <repo-url>
cd reviewlens
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 Quick Start

The processed dataset is already included, so you only need two commands:

```bash
python train.py                 # Step 1 — train the model
streamlit run app.py            # Step 2 — launch the dashboard
```

Open your browser at **http://localhost:8501**

---

## ☁️ Hosting from GitHub (Streamlit Cloud)

`models/pipeline.pkl` is intentionally not stored in Git, so a fresh hosted
deployment may start without a trained model file.

ReviewLens now auto-trains once on first launch (using `data/raw/reviews.csv`),
then continues normally after the model is generated.

If auto-training cannot run (for example, required data files are missing), run:

```bash
python prepare_dataset.py
python train.py
```

Then redeploy.

---

## 📓 Jupyter Notebook (Viva Walkthrough)

`ReviewLens_Notebook.ipynb` is a full, narrated walkthrough of the entire
pipeline — EDA, preprocessing, hyperparameter search, training, evaluation,
and explainability — in one place, with outputs already embedded so it can
be read without running anything first.

It imports directly from `src/`, the same code `train.py` and `app.py` use,
so results in the notebook are guaranteed to match the shipped model rather
than being a separate, potentially inconsistent re-implementation.

```bash
jupyter notebook ReviewLens_Notebook.ipynb
```

To regenerate it from scratch (e.g. after retraining with different data):

```bash
python build_notebook.py    # rebuilds the notebook structure
jupyter nbconvert --to notebook --execute --inplace ReviewLens_Notebook.ipynb
```

---

## 🤖 How Training Works

```bash
python train.py
```

1. Loads `data/raw/reviews.csv` (~30,800 real reviews)
2. Preprocesses text (lowercase, HTML/URL removal, stopwords, lemmatization)
3. Splits 80/20 with stratification
4. Trains **Logistic Regression** across a small hyperparameter grid
   (cross-validated on macro F1) and a **Naive Bayes** baseline
5. Selects the best configuration
6. Saves `models/pipeline.pkl` and `models/metadata.json`

**Actual results on this dataset** (see `models/metadata.json` for the
exact numbers from your own training run):
```
Model      : LogisticRegression
Accuracy   : 78.2%
Macro F1   : 0.609
Weighted F1: 0.782
```

These are honest numbers from real, noisy, imbalanced review data — not
inflated synthetic-data numbers. The "neutral" class remains the hardest
to classify, which matches published sentiment-analysis literature:
neutral reviews are inherently ambiguous and are a small minority of real
review data (~7% here). See [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md)
for the full breakdown of what was tried to improve this.

---

## 🔄 Regenerating the Dataset (optional)

Only needed if you want to add/remove a data source or rebuild from
scratch. The raw Kaggle CSVs are not included in this repo (≈250MB).

1. Download the source datasets (see file names listed in `src/data_sources.py`)
2. Place them in `data/kaggle_raw/`
3. Run:
   ```bash
   python prepare_dataset.py
   ```
4. This rewrites `data/raw/reviews.csv`; re-run `python train.py` afterward.

To add a new source, add one `SourceConfig` entry to
`src/data_sources.py` — no other code changes needed.

---

## 🖥️ Using the Dashboard

```bash
streamlit run app.py
```

- **Predict Sentiment** — paste any review, click Analyze, see the
  predicted label, confidence, and which words drove the prediction.
- **Review Explorer** — upload **any** CSV. The app auto-detects which
  column holds the review text (and rating/date/source/category, if
  present) and lets you confirm or correct the mapping before analysis.
- All other pages (Trends, Comparison, Summary, Export) analyse
  whatever the active dataset currently is.

---

## 📝 Example Single Review

**Input:**
```
The phone camera is excellent, but battery backup is very poor.
```

**Output:**
```
Sentiment  : Positive
Positive   : excellent, backup, battery backup, phone camera, camera, battery
Negative   : poor, phone
```

The model correctly identifies both signals — "excellent" pulling toward
positive and "poor" pulling toward negative — which is exactly why mixed
reviews like this get a more moderate confidence score than a review
that's clearly one-sided.

---

## 📋 CSV Upload — No Fixed Format Required

Any CSV can be uploaded. Column names don't need to match anything
specific — the app detects the right columns automatically:

```csv
Comments,Stars,Timestamp,Platform
"Great product! Very happy with my purchase.",5,2024-01-15,Amazon
"Terrible quality. Stopped working after a week.",1,2024-02-20,Flipkart
```

This would be auto-mapped as `Comments → review_text`, `Stars → rating`,
`Timestamp → date`, `Platform → source` — no renaming needed. A minimal
CSV with only a text column also works fine:

```csv
feedback
"Great product! Very happy with my purchase."
"Terrible quality. Stopped working after a week."
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

- `test_preprocess.py` — text cleaning pipeline (no model needed)
- `test_data_loader.py` — label-mapping logic + source config sanity checks
- `test_csv_mapper.py` — flexible column-detection logic for CSV upload
- `test_model.py` — trained model correctness (skipped if not trained)
- `test_app.py` — module/page import checks
- `test_dashboard.py` — actually runs the Streamlit app and every page
  using Streamlit's `AppTest` framework, catching real render-time bugs

---

## ⚠️ Limitations

1. **English only** — preprocessing and stopwords are English-specific
2. **Neutral class is weak** (precision ≈ 28%) — inherently ambiguous
   in real review data. The dashboard shows an explicit warning
   whenever a prediction is neutral, and the explanation panel shows
   dedicated hedging-language keywords (e.g. "okay", "nothing
   special") rather than just leftover positive/negative signals —
   see [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) §5d for why
   this distinction matters
3. **No deep learning** — TF-IDF + LR is CPU-friendly and interpretable,
   but a transformer model (BERT etc.) would likely score higher
4. **No aspect-level sentiment** — can't separate "camera good, battery bad"
   into two separate scores within one review
5. **Domain shift** — the merged dataset spans very different domains
   (movies, courses, electronics); a model fine-tuned on one domain only
   would likely perform better within that domain
6. **Column auto-detection is heuristic** — very unusually-named columns
   may need manual correction in the mapping dropdown

---

## 🔮 Future Improvements

- Fine-tune a transformer model (DistilBERT) for higher accuracy
- Aspect-based sentiment analysis (separate scores per product feature)
- Multi-language support
- REST API (FastAPI) for integration with other systems
- Active learning — feed user corrections back into retraining

---

## ✅ Project Checklist

- [x] Trained on real Kaggle data (not synthetic)
- [x] 13 data sources merged into one unified schema
- [x] Model trains (`python train.py`) with hyperparameter search
- [x] Single review analysis works
- [x] CSV upload accepts **any** column structure via auto-mapping
- [x] Word clouds, keyword charts, time trend, category/source comparison
- [x] Model performance page with confusion matrix
- [x] Annotated CSV + summary report download
- [x] 139 automated tests (unit + dashboard integration)
- [x] Upload limits (file size, row count) enforced and clearly displayed
- [x] Robust error handling for corrupted/invalid/oversized CSV files
- [x] Rating-vs-sentiment mismatch detector and confidence distribution
- [x] Source-aware trend analysis (avoids misleading mixed-platform trends)
- [x] Documentation (README, ARCHITECTURE, PROJECT_REPORT)
