"""
==========================================================================
Streamlit App — Analisis Sentimen Hybrid SVM (TF-IDF Word + Char N-gram)
Dataset: Tweet COVID-19 dari London (Step_one, Step_two, Step_three)
Validasi: Stratified K-Fold Cross-Validation (5-Fold)

Cara menjalankan:
    streamlit run app.py

File CSV (Step_one.csv, Step_two.csv, Step_three.csv) HARUS berada
di folder yang sama dengan app.py ini.
==========================================================================
"""
import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import seaborn as sns
import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score)
from scipy.sparse import hstack

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analisis Sentimen Hybrid SVM",
    page_icon=None,
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS — TEMA FORMAL DARK/LIGHT
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>

/* ════════════════════════════════════════════════════════════════
   DARK THEME  (default — Streamlit dark mode)
   Selector: [data-theme="dark"]  atau tidak ada data-theme
   ════════════════════════════════════════════════════════════════ */

/* ── Variabel dark ── */
[data-theme="dark"],
[data-theme="dark"] * {
    --bg-base:        #18122B;
    --bg-surface:     #393053;
    --bg-card:        #443C68;
    --bg-accent:      #635985;
    --text-primary:   #F8FAFC;
    --text-secondary: #D9EAFD;
    --text-muted:     #BCCCDC;
    --text-subtle:    #9AA6B2;
    --border-color:   #635985;
}

/* ── Background utama dark ── */
[data-theme="dark"] .stApp,
[data-theme="dark"] section[data-testid="stMain"],
[data-theme="dark"] section[data-testid="stMain"] > div,
[data-theme="dark"] .stMainBlockContainer,
[data-theme="dark"] .block-container {
    background-color: #18122B !important;
}

[data-theme="dark"] section[data-testid="stSidebar"],
[data-theme="dark"] section[data-testid="stSidebar"] > div {
    background-color: #393053 !important;
}

/* ── Teks dark ── */
[data-theme="dark"] html,
[data-theme="dark"] body,
[data-theme="dark"] p,
[data-theme="dark"] span,
[data-theme="dark"] label,
[data-theme="dark"] div,
[data-theme="dark"] .stMarkdown,
[data-theme="dark"] .stText {
    color: #F8FAFC !important;
}

[data-theme="dark"] h1,
[data-theme="dark"] h2,
[data-theme="dark"] h3,
[data-theme="dark"] h4,
[data-theme="dark"] h5,
[data-theme="dark"] h6 {
    color: #D9EAFD !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

[data-theme="dark"] .stApp h1 {
    color: #F8FAFC !important;
    border-bottom: 2px solid #635985;
    padding-bottom: 0.4rem;
}

[data-theme="dark"] .stCaption,
[data-theme="dark"] [data-testid="stCaptionContainer"] p {
    color: #9AA6B2 !important;
    font-size: 0.85rem;
}

/* ── Metric dark ── */
[data-theme="dark"] [data-testid="stMetric"] {
    background-color: #443C68 !important;
    border: 1px solid #635985 !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
[data-theme="dark"] [data-testid="stMetricLabel"] p {
    color: #BCCCDC !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-theme="dark"] [data-testid="stMetricValue"] {
    color: #F8FAFC !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Tombol dark ── */
[data-theme="dark"] .stButton > button {
    background-color: #635985 !important;
    color: #F8FAFC !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease;
}
[data-theme="dark"] .stButton > button:hover {
    background-color: #443C68 !important;
}
[data-theme="dark"] [data-testid="stDownloadButton"] > button {
    background-color: #393053 !important;
    color: #D9EAFD !important;
    border: 1px solid #635985 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
[data-theme="dark"] [data-testid="stDownloadButton"] > button:hover {
    background-color: #443C68 !important;
}

/* ── Alert boxes dark ── */
[data-theme="dark"] [data-testid="stInfo"] {
    background-color: rgba(99,89,133,0.22) !important;
    border-left: 4px solid #635985 !important;
    border-radius: 6px !important;
}
[data-theme="dark"] [data-testid="stInfo"] p { color: #D9EAFD !important; }

[data-theme="dark"] [data-testid="stSuccess"] {
    background-color: rgba(68,60,104,0.35) !important;
    border-left: 4px solid #5C8A6E !important;
    border-radius: 6px !important;
}
[data-theme="dark"] [data-testid="stSuccess"] p { color: #F8FAFC !important; }

[data-theme="dark"] [data-testid="stError"] {
    background-color: rgba(130,50,50,0.30) !important;
    border-left: 4px solid #C0392B !important;
    border-radius: 6px !important;
}
[data-theme="dark"] [data-testid="stError"] p { color: #F8FAFC !important; }

[data-theme="dark"] [data-testid="stWarning"] {
    background-color: rgba(120,90,40,0.30) !important;
    border-left: 4px solid #D4A017 !important;
    border-radius: 6px !important;
}
[data-theme="dark"] [data-testid="stWarning"] p { color: #F8FAFC !important; }

/* ── Expander dark ── */
[data-theme="dark"] details {
    background-color: #393053 !important;
    border: 1px solid #635985 !important;
    border-radius: 8px !important;
}
[data-theme="dark"] details summary { color: #D9EAFD !important; font-weight: 600 !important; }
[data-theme="dark"] details > div    { color: #BCCCDC !important; }

/* ── Tabs dark ── */
[data-theme="dark"] [data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #635985 !important;
}
[data-theme="dark"] [data-testid="stTabs"] [role="tab"] {
    color: #9AA6B2 !important;
    font-weight: 500 !important;
}
[data-theme="dark"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #F8FAFC !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #D9EAFD !important;
    background-color: #443C68 !important;
}

/* ── DataFrame dark ── */
[data-theme="dark"] [data-testid="stDataFrame"] {
    border: 1px solid #635985 !important;
    border-radius: 8px !important;
}
[data-theme="dark"] [data-testid="stDataFrame"] thead tr th {
    background-color: #443C68 !important;
    color: #D9EAFD !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 0.82rem !important;
}
[data-theme="dark"] [data-testid="stDataFrame"] tbody tr td {
    color: #F8FAFC !important;
    font-size: 0.84rem !important;
}

/* ── Code dark ── */
[data-theme="dark"] .stCode,
[data-theme="dark"] code,
[data-theme="dark"] pre {
    background-color: #393053 !important;
    color: #D9EAFD !important;
    border: 1px solid #635985 !important;
    border-radius: 6px !important;
}

[data-theme="dark"] hr { border-color: #635985 !important; opacity: 0.4; }

/* ════════════════════════════════════════════════════════════════
   LIGHT THEME  (Streamlit light mode — data-theme="light")
   Bg menggunakan light palette, teks menggunakan dark palette
   ════════════════════════════════════════════════════════════════ */

/* ── Background utama light ── */
[data-theme="light"] .stApp,
[data-theme="light"] section[data-testid="stMain"],
[data-theme="light"] section[data-testid="stMain"] > div,
[data-theme="light"] .stMainBlockContainer,
[data-theme="light"] .block-container {
    background-color: #F8FAFC !important;
}

[data-theme="light"] section[data-testid="stSidebar"],
[data-theme="light"] section[data-testid="stSidebar"] > div {
    background-color: #D9EAFD !important;
}

/* ── Teks light — gunakan dark palette agar kontras ── */
[data-theme="light"] html,
[data-theme="light"] body,
[data-theme="light"] p,
[data-theme="light"] span,
[data-theme="light"] label,
[data-theme="light"] div,
[data-theme="light"] .stMarkdown,
[data-theme="light"] .stText {
    color: #18122B !important;
}

[data-theme="light"] h1,
[data-theme="light"] h2,
[data-theme="light"] h3,
[data-theme="light"] h4,
[data-theme="light"] h5,
[data-theme="light"] h6 {
    color: #393053 !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

[data-theme="light"] .stApp h1 {
    color: #18122B !important;
    border-bottom: 2px solid #635985;
    padding-bottom: 0.4rem;
}

[data-theme="light"] .stCaption,
[data-theme="light"] [data-testid="stCaptionContainer"] p {
    color: #635985 !important;
    font-size: 0.85rem;
}

/* ── Metric light ── */
[data-theme="light"] [data-testid="stMetric"] {
    background-color: #D9EAFD !important;
    border: 1px solid #BCCCDC !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
}
[data-theme="light"] [data-testid="stMetricLabel"] p {
    color: #443C68 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-theme="light"] [data-testid="stMetricValue"] {
    color: #18122B !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
}

/* ── Tombol light ── */
[data-theme="light"] .stButton > button {
    background-color: #635985 !important;
    color: #F8FAFC !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.8rem !important;
    font-weight: 600 !important;
    transition: background-color 0.2s ease;
}
[data-theme="light"] .stButton > button:hover {
    background-color: #443C68 !important;
}
[data-theme="light"] [data-testid="stDownloadButton"] > button {
    background-color: #BCCCDC !important;
    color: #18122B !important;
    border: 1px solid #635985 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
}
[data-theme="light"] [data-testid="stDownloadButton"] > button:hover {
    background-color: #9AA6B2 !important;
}

/* ── Alert boxes light ── */
[data-theme="light"] [data-testid="stInfo"] {
    background-color: rgba(188,204,220,0.45) !important;
    border-left: 4px solid #635985 !important;
    border-radius: 6px !important;
}
[data-theme="light"] [data-testid="stInfo"] p { color: #18122B !important; }

[data-theme="light"] [data-testid="stSuccess"] {
    background-color: rgba(92,138,110,0.15) !important;
    border-left: 4px solid #5C8A6E !important;
    border-radius: 6px !important;
}
[data-theme="light"] [data-testid="stSuccess"] p { color: #18122B !important; }

[data-theme="light"] [data-testid="stError"] {
    background-color: rgba(192,57,43,0.10) !important;
    border-left: 4px solid #C0392B !important;
    border-radius: 6px !important;
}
[data-theme="light"] [data-testid="stError"] p { color: #18122B !important; }

[data-theme="light"] [data-testid="stWarning"] {
    background-color: rgba(212,160,23,0.12) !important;
    border-left: 4px solid #D4A017 !important;
    border-radius: 6px !important;
}
[data-theme="light"] [data-testid="stWarning"] p { color: #18122B !important; }

/* ── Expander light ── */
[data-theme="light"] details {
    background-color: #D9EAFD !important;
    border: 1px solid #BCCCDC !important;
    border-radius: 8px !important;
}
[data-theme="light"] details summary { color: #393053 !important; font-weight: 600 !important; }
[data-theme="light"] details > div    { color: #18122B !important; }

/* ── Tabs light ── */
[data-theme="light"] [data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #BCCCDC !important;
}
[data-theme="light"] [data-testid="stTabs"] [role="tab"] {
    color: #635985 !important;
    font-weight: 500 !important;
}
[data-theme="light"] [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #18122B !important;
    font-weight: 700 !important;
    border-bottom: 3px solid #443C68 !important;
    background-color: #BCCCDC !important;
}

/* ── DataFrame light ── */
[data-theme="light"] [data-testid="stDataFrame"] {
    border: 1px solid #BCCCDC !important;
    border-radius: 8px !important;
}
[data-theme="light"] [data-testid="stDataFrame"] thead tr th {
    background-color: #D9EAFD !important;
    color: #393053 !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    font-size: 0.82rem !important;
}
[data-theme="light"] [data-testid="stDataFrame"] tbody tr td {
    color: #18122B !important;
    font-size: 0.84rem !important;
}
[data-theme="light"] [data-testid="stDataFrame"] tbody tr:nth-child(even) td {
    background-color: rgba(188,204,220,0.30) !important;
}

/* ── Code light ── */
[data-theme="light"] .stCode,
[data-theme="light"] code,
[data-theme="light"] pre {
    background-color: #D9EAFD !important;
    color: #393053 !important;
    border: 1px solid #BCCCDC !important;
    border-radius: 6px !important;
}

[data-theme="light"] hr { border-color: #BCCCDC !important; opacity: 0.6; }

/* ════════════════════════════════════════════════════════════════
   KOMPONEN UNIVERSAL (berlaku di kedua tema)
   ════════════════════════════════════════════════════════════════ */

/* ── Header separator ── */
.header-divider {
    height: 1px;
    background: linear-gradient(to right, #635985, transparent);
    margin: 0.5rem 0 1.2rem 0;
}

/* ── Font & spacing global ── */
html, body { font-family: "Inter", "Segoe UI", sans-serif; }

/* ── Spinner ── */
[data-theme="dark"]  [data-testid="stSpinner"] p { color: #9AA6B2 !important; }
[data-theme="light"] [data-testid="stSpinner"] p { color: #635985 !important; }

</style>
""", unsafe_allow_html=True)

# Folder berisi script ini = folder data (path relatif, sama seperti
# asumsi script asli yang mengharapkan CSV ada satu folder dengan kode)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_FILES = ["Step_one.csv", "Step_two.csv", "Step_three.csv"]

STOP = {"i","me","my","we","our","you","your","he","him","his","she","her","it","its",
"they","them","their","what","which","who","this","that","these","those","am","is","are",
"was","were","be","been","have","has","had","do","does","did","a","an","the","and","but",
"if","or","of","at","by","for","with","to","from","in","out","on","off","so","not","no",
"than","too","s","t","can","will","just","now","via","amp","rt","get","got","one","also",
"still","even","like","said","go","going","come","know","think","people","make","want",
"see","day","time","covid","coronavirus","uk","london","would","could"}
URL_RE = re.compile(r"http\S+|www\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NONALPHA_RE = re.compile(r"[^a-z\s]")


def preprocess(t):
    t = t.lower()
    for r in (URL_RE, MENTION_RE, HASHTAG_RE):
        t = r.sub(" ", t)
    t = NONALPHA_RE.sub(" ", t)
    return " ".join(w for w in t.split() if w not in STOP and len(w) > 2)


# ──────────────────────────────────────────────────────────────────
# PIPELINE UTAMA (di-cache supaya tidak diulang setiap interaksi UI)
# ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_merge_data():
    missing = [f for f in DATA_FILES if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        return None, missing
    dfs = []
    for f in DATA_FILES:
        df = pd.read_csv(os.path.join(BASE_DIR, f))
        df["source"] = f[:-4]
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)[["tweet", "source"]].dropna(subset=["tweet"]).copy()
    df_all["tweet"] = df_all["tweet"].astype(str)
    return df_all, []


@st.cache_data(show_spinner=False)
def run_preprocessing(df_all):
    df_all = df_all.copy()
    df_all["clean_tweet"] = df_all["tweet"].apply(preprocess)
    df_all = df_all[df_all["clean_tweet"].str.strip() != ""].copy()
    df_all["polarity"] = df_all["tweet"].apply(lambda t: TextBlob(t).sentiment.polarity)
    df_all["sentiment"] = df_all["polarity"].apply(
        lambda p: "Positif" if p > 0.1 else ("Negatif" if p < -0.1 else "Netral"))
    return df_all


@st.cache_data(show_spinner=False)
def build_balanced_sample(df_all, max_per_class=7000, seed=42):
    df_bin = df_all[df_all["sentiment"].isin(["Positif", "Negatif"])].copy()
    n_per_class = min(df_bin["sentiment"].value_counts().min(), max_per_class)
    frames = [grp.sample(min(len(grp), n_per_class), random_state=seed)
              for _, grp in df_bin.groupby("sentiment")]
    df_s = pd.concat(frames).reset_index(drop=True)
    return df_s, n_per_class


@st.cache_resource(show_spinner=False)
def extract_features(texts):
    tfidf_w = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), sublinear_tf=True,
                               min_df=2, max_df=0.95)
    X1 = tfidf_w.fit_transform(texts)

    tfidf_c = TfidfVectorizer(max_features=12000, ngram_range=(3, 6), sublinear_tf=True,
                               min_df=3, max_df=0.95, analyzer="char_wb")
    X2 = tfidf_c.fit_transform(texts)

    X = hstack([X1, X2])
    X_sc = MaxAbsScaler().fit_transform(X)
    return X_sc, tfidf_w, tfidf_c


@st.cache_data(show_spinner=False)
def train_and_evaluate(_cache_key, _X_sc, _y):
    svm = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(svm, _X_sc, _y, cv=cv, n_jobs=-1)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(_X_sc, _y), 1):
        X_tr, X_te = _X_sc[train_idx], _X_sc[test_idx]
        y_tr, y_te = _y[train_idx], _y[test_idx]
        svm_f = LinearSVC(C=1.0, class_weight="balanced", random_state=42, max_iter=3000)
        svm_f.fit(X_tr, y_tr)
        y_fp = svm_f.predict(X_te)
        cm_f = confusion_matrix(y_te, y_fp)
        tn_f, fp_f, fn_f, tp_f = cm_f.ravel() if cm_f.shape == (2, 2) else (0, 0, 0, 0)
        fold_metrics.append({
            "fold": fold_idx, "n_train": len(train_idx), "n_test": len(test_idx),
            "accuracy": accuracy_score(y_te, y_fp),
            "precision": precision_score(y_te, y_fp, average="weighted", zero_division=0),
            "recall": recall_score(y_te, y_fp, average="weighted"),
            "f1": f1_score(y_te, y_fp, average="weighted"),
            "tp": int(tp_f), "tn": int(tn_f), "fp": int(fp_f), "fn": int(fn_f),
        })
    df_fold = pd.DataFrame(fold_metrics)
    return y_pred, df_fold


# ──────────────────────────────────────────────────────────────────
# FUNGSI PEMBUAT FIGURE (figure 1, 2, 3 — identik dengan script asli)
# ──────────────────────────────────────────────────────────────────
def make_figure1(df_all, df_s, df_fold, X_sc, y, y_pred, classes, acc, prec, rec, f1):
    plt.style.use("seaborn-v0_8-whitegrid")
    cm_all = confusion_matrix(y, y_pred)
    tn_all, fp_all, fn_all, tp_all = cm_all.ravel()

    fig1 = plt.figure(figsize=(18, 22))
    fig1.patch.set_facecolor("#f8f9fa")
    fig1.suptitle(
        "CARA KLASIFIKASI & PERHITUNGAN AKURASI\n"
        "Hybrid SVM: TF-IDF Word N-gram (1-3, 20K) + Char N-gram (3-6, 12K) | 5-Fold CV",
        fontsize=15, fontweight="bold", color="#1a252f", y=0.99
    )

    # (A) Pipeline
    ax_pipe = fig1.add_axes([0.03, 0.83, 0.94, 0.13])
    ax_pipe.set_xlim(0, 10); ax_pipe.set_ylim(0, 1); ax_pipe.axis("off")
    ax_pipe.set_title("(A) Pipeline Klasifikasi — Alur dari Tweet hingga Label Prediksi",
                       fontsize=12, fontweight="bold", loc="left", pad=6, color="#1a252f")
    steps = [
        ("① Input\nTweet", "#2c3e50"),
        ("② Pre-\nprocessing", "#16a085"),
        ("③ TextBlob\nLabeling", "#8e44ad"),
        ("④ TF-IDF\nWord+Char", "#2980b9"),
        ("⑤ MaxAbs\nScaler", "#e67e22"),
        ("⑥ Linear\nSVC", "#c0392b"),
        ("⑦ 5-Fold\nCV", "#27ae60"),
        ("⑧ Label\nPrediksi", "#2c3e50"),
    ]
    bw = 1.1; gap = 0.1; start = 0.1
    for i, (label, col) in enumerate(steps):
        x = start + i * (bw + gap)
        rect = mpatches.Rectangle((x, 0.1), bw, 0.8, color=col, alpha=0.88, zorder=2)
        ax_pipe.add_patch(rect)
        ax_pipe.text(x + bw / 2, 0.5, label, ha="center", va="center",
                     fontsize=8.5, fontweight="bold", color="white", zorder=3)
        if i < len(steps) - 1:
            ax_pipe.annotate("", xy=(x + bw + gap, 0.5), xytext=(x + bw, 0.5),
                              arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # (B) Distribusi Polarity
    ax_pol = fig1.add_axes([0.03, 0.65, 0.44, 0.15])
    pol_vals = df_all["polarity"].values
    bins = np.linspace(-1, 1, 60)
    ax_pol.hist(pol_vals[pol_vals > 0.1], bins=bins, color="#2ecc71", alpha=0.8, label="Positif (>+0.1)")
    ax_pol.hist(pol_vals[np.abs(pol_vals) <= 0.1], bins=bins, color="#95a5a6", alpha=0.6, label="Netral (dibuang)")
    ax_pol.hist(pol_vals[pol_vals < -0.1], bins=bins, color="#e74c3c", alpha=0.8, label="Negatif (<-0.1)")
    ax_pol.axvline(0.1, color="#27ae60", lw=1.5, ls="--")
    ax_pol.axvline(-0.1, color="#c0392b", lw=1.5, ls="--")
    ax_pol.set_title("(B) Distribusi Polarity TextBlob & Threshold Labeling",
                      fontsize=11, fontweight="bold", color="#1a252f")
    ax_pol.set_xlabel("Polarity Score"); ax_pol.set_ylabel("Jumlah Tweet")
    ax_pol.legend(fontsize=8); ax_pol.grid(True, alpha=0.3)
    n_pos = (df_all["sentiment"] == "Positif").sum()
    n_neg = (df_all["sentiment"] == "Negatif").sum()
    n_net = (df_all["sentiment"] == "Netral").sum()
    ax_pol.text(0.98, 0.97,
                f"Positif: {n_pos:,}\nNetral (dibuang): {n_net:,}\nNegatif: {n_neg:,}",
                transform=ax_pol.transAxes, va="top", ha="right", fontsize=8,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # (C) Komposisi Fitur
    ax_feat = fig1.add_axes([0.55, 0.65, 0.42, 0.15])
    feat_labels = ["TF-IDF Word\nN-gram (1-3)\n20.000 fitur", "TF-IDF Char\nN-gram (3-6)\n12.000 fitur"]
    feat_sizes = [20000, 12000]
    feat_colors = ["#3498db", "#9b59b6"]
    bars_f = ax_feat.barh(feat_labels, feat_sizes, color=feat_colors, alpha=0.85, height=0.4)
    for bar, sz in zip(bars_f, feat_sizes):
        ax_feat.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
                     f"{sz:,} fitur", va="center", fontsize=10, fontweight="bold")
    ax_feat.axvline(32000, color="#e74c3c", lw=1.5, ls="--", label="Total: 32.000 fitur")
    ax_feat.set_xlim(0, 36000)
    ax_feat.set_title("(C) Komposisi Fitur Hybrid TF-IDF", fontsize=11, fontweight="bold", color="#1a252f")
    ax_feat.set_xlabel("Jumlah Fitur"); ax_feat.legend(fontsize=9)
    ax_feat.text(0.5, -0.22, "hstack([X_word, X_char]) → sparse matrix 32.000 kolom → MaxAbsScaler",
                 transform=ax_feat.transAxes, ha="center", fontsize=8.5, style="italic", color="#555")

    # (D) Tabel Per-Fold
    ax_tbl = fig1.add_axes([0.03, 0.44, 0.94, 0.18])
    ax_tbl.axis("off")
    ax_tbl.set_title("(D) Perhitungan Per Fold — Stratified 5-Fold Cross-Validation",
                      fontsize=12, fontweight="bold", loc="left", pad=6, color="#1a252f")
    col_labels = ["Fold", "N-Train", "N-Test", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1-Score"]
    tbl_data = []
    for _, r in df_fold.iterrows():
        tbl_data.append([
            f"Fold {int(r.fold)}", f"{int(r.n_train):,}", f"{int(r.n_test):,}",
            f"{r.tp:,}", f"{r.tn:,}", f"{r.fp:,}", f"{r.fn:,}",
            f"{r.accuracy:.4f}", f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
        ])
    tbl_data.append([
        "Mean", "", "",
        f"{df_fold.tp.mean():.0f}", f"{df_fold.tn.mean():.0f}",
        f"{df_fold.fp.mean():.0f}", f"{df_fold.fn.mean():.0f}",
        f"{df_fold.accuracy.mean():.4f}", f"{df_fold.precision.mean():.4f}",
        f"{df_fold.recall.mean():.4f}", f"{df_fold.f1.mean():.4f}",
    ])
    tbl_data.append([
        "Std Dev", "", "", "", "", "", "",
        f"{df_fold.accuracy.std():.4f}", f"{df_fold.precision.std():.4f}",
        f"{df_fold.recall.std():.4f}", f"{df_fold.f1.std():.4f}",
    ])
    tbl = ax_tbl.table(cellText=tbl_data, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.55)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c3e50"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    row_colors = ["#eaf4fb", "#fdfefe", "#eaf4fb", "#fdfefe", "#eaf4fb"]
    for i, rc in enumerate(row_colors):
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(rc)
    for j in range(len(col_labels)):
        tbl[6, j].set_facecolor("#d5f5e3"); tbl[6, j].set_text_props(fontweight="bold")
    for j in range(len(col_labels)):
        tbl[7, j].set_facecolor("#fef9e7")

    # (E) Rumus Metrik
    ax_form = fig1.add_axes([0.03, 0.23, 0.44, 0.18])
    ax_form.axis("off")
    ax_form.set_title("(E) Rumus Perhitungan Metrik Evaluasi",
                       fontsize=12, fontweight="bold", loc="left", pad=6, color="#1a252f")
    cm_g = confusion_matrix(y, y_pred)
    tn_g, fp_g, fn_g, tp_g = cm_g.ravel()
    formulas = [
        ("Accuracy", "(TP + TN) / (TP+TN+FP+FN)",
         f"= ({tp_g:,} + {tn_g:,}) / ({tp_g + tn_g + fp_g + fn_g:,})",
         f"= {acc:.4f}  ({acc * 100:.2f}%)", "#1abc9c"),
        ("Precision", "TP / (TP + FP)", f"= {tp_g:,} / ({tp_g:,}+{fp_g:,})", f"= {prec:.4f}", "#3498db"),
        ("Recall", "TP / (TP + FN)", f"= {tp_g:,} / ({tp_g:,}+{fn_g:,})", f"= {rec:.4f}", "#e67e22"),
        ("F1-Score", "2 × (Precision × Recall) / (Precision + Recall)",
         f"= 2 × ({prec:.4f} × {rec:.4f}) / ({prec:.4f}+{rec:.4f})", f"= {f1:.4f}", "#9b59b6"),
    ]
    y_pos = 0.95
    for name, formula, calc, result, col in formulas:
        ax_form.text(0.0, y_pos, f"● {name}:", fontsize=10, fontweight="bold", color=col,
                     transform=ax_form.transAxes, va="top")
        ax_form.text(0.22, y_pos, formula, fontsize=9, color="#2c3e50",
                     transform=ax_form.transAxes, va="top", family="monospace")
        ax_form.text(0.22, y_pos - 0.13, calc, fontsize=8.5, color="#555",
                     transform=ax_form.transAxes, va="top", family="monospace")
        ax_form.text(0.22, y_pos - 0.26, result, fontsize=9, fontweight="bold", color=col,
                     transform=ax_form.transAxes, va="top", family="monospace")
        y_pos -= 0.46

    # (F) Confusion Matrix Global
    ax_cm = fig1.add_axes([0.55, 0.23, 0.42, 0.18])
    im = ax_cm.imshow(cm_g, interpolation="nearest", cmap="Blues")
    ax_cm.set_title("(F) Confusion Matrix Global (semua fold)", fontsize=11, fontweight="bold", color="#1a252f")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(classes, fontsize=11); ax_cm.set_yticklabels(classes, fontsize=11)
    ax_cm.set_xlabel("Prediksi", fontsize=11); ax_cm.set_ylabel("Aktual", fontsize=11)
    thresh = cm_g.max() / 2.
    for i in range(2):
        for j in range(2):
            lbl = ("TP" if (i == 1 and j == 1) else "TN" if (i == 0 and j == 0)
                   else "FP" if (i == 0 and j == 1) else "FN")
            ax_cm.text(j, i - 0.18, lbl, ha="center", va="center", fontsize=9,
                       color="white" if cm_g[i, j] > thresh else "#555", fontstyle="italic")
            ax_cm.text(j, i + 0.18, f"{cm_g[i, j]:,}", ha="center", va="center",
                       fontsize=14, fontweight="bold",
                       color="white" if cm_g[i, j] > thresh else "#1a252f")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    # (G) Per-Fold Accuracy Chart
    ax_cv = fig1.add_axes([0.03, 0.07, 0.44, 0.13])
    fold_nums = df_fold["fold"].tolist()
    fold_accs = df_fold["accuracy"].tolist()
    fold_f1s = df_fold["f1"].tolist()
    ax_cv.plot(fold_nums, fold_accs, "o-", color="#1abc9c", lw=2, ms=8, label="Accuracy")
    ax_cv.plot(fold_nums, fold_f1s, "s--", color="#9b59b6", lw=2, ms=8, label="F1-Score")
    ax_cv.axhline(acc, color="#1abc9c", lw=1, ls=":", alpha=0.6)
    ax_cv.axhline(f1, color="#9b59b6", lw=1, ls=":", alpha=0.6)
    for i, (a, f) in enumerate(zip(fold_accs, fold_f1s)):
        ax_cv.text(fold_nums[i], a + 0.002, f"{a:.4f}", ha="center", fontsize=8, color="#16a085")
        ax_cv.text(fold_nums[i], f - 0.007, f"{f:.4f}", ha="center", fontsize=8, color="#7d3c98")
    ax_cv.set_ylim(min(fold_accs + fold_f1s) - 0.03, max(fold_accs + fold_f1s) + 0.03)
    ax_cv.set_xticks(fold_nums); ax_cv.set_xlabel("Fold ke-")
    ax_cv.set_title("(G) Performa Tiap Fold (Accuracy & F1)", fontsize=11, fontweight="bold", color="#1a252f")
    ax_cv.legend(fontsize=9); ax_cv.grid(True, alpha=0.3)

    # (H) Ringkasan
    ax_sum = fig1.add_axes([0.55, 0.07, 0.42, 0.13])
    ax_sum.axis("off")
    ax_sum.set_facecolor("#eaf4fb")
    summary_lines = [
        f"  Dataset       : {len(df_all):,} tweet (setelah preprocessing)",
        f"  Label Positif : {n_pos:,} | Negatif: {n_neg:,} | Netral dibuang: {n_net:,}",
        f"  Sampel latih  : {len(df_s):,} tweet (balanced, maks 7.000/kelas)",
        f"  Fitur hybrid  : {X_sc.shape[1]:,} (Word 20K + Char 12K)",
        f"  Model         : LinearSVC (C=1.0, class_weight=balanced)",
        f"  Validasi      : Stratified 5-Fold Cross-Validation",
        f"  TP={tp_g:,}  TN={tn_g:,}  FP={fp_g:,}  FN={fn_g:,}",
        f"  Accuracy      : {acc * 100:.2f}%   F1={f1:.4f}   Precision={prec:.4f}   Recall={rec:.4f}",
    ]
    ax_sum.set_title("(H) Ringkasan Hasil Klasifikasi", fontsize=11, fontweight="bold", loc="left", pad=4, color="#1a252f")
    for i, line in enumerate(summary_lines):
        color = "#c0392b" if "Accuracy" in line else "#2c3e50"
        fw = "bold" if "Accuracy" in line else "normal"
        ax_sum.text(0.01, 0.93 - i * 0.115, line, transform=ax_sum.transAxes,
                    fontsize=8.8, va="top", color=color, fontweight=fw, family="monospace")

    return fig1


def make_figure2(df_all, y, classes, y_pred, acc, prec, rec, f1):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Analisis Sentimen Tweet COVID-19 (London)\n"
        "Hybrid SVM: TF-IDF Word N-gram (1-3, 20K) + Char N-gram (3-6, 12K) | 5-Fold CV",
        fontsize=13, fontweight="bold", y=0.99)

    ax1 = axes[0, 0]; unique, counts = np.unique(y, return_counts=True)
    bars1 = ax1.bar(classes, counts, color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5)
    ax1.set_title("(a) Distribusi Label Sentimen", fontweight="bold")
    ax1.set_xlabel("Sentimen"); ax1.set_ylabel("Jumlah Tweet")
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                  f"{int(bar.get_height()):,}", ha="center", fontsize=12, fontweight="bold")

    ax2 = axes[0, 1]; cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
                ax=ax2, linewidths=0.5, annot_kws={"size": 13})
    ax2.set_title("(b) Confusion Matrix (5-Fold CV)", fontweight="bold")
    ax2.set_xlabel("Prediksi"); ax2.set_ylabel("Aktual")

    ax3 = axes[1, 0]; report = classification_report(y, y_pred, target_names=classes, output_dict=True, zero_division=0)
    xp = np.arange(len(classes)); w = 0.25
    for i, (metric, color) in enumerate(zip(["precision", "recall", "f1-score"], ["#3498db", "#e67e22", "#9b59b6"])):
        vals = [report[lbl][metric] for lbl in classes]
        ax3.bar(xp + i * w, vals, width=w, label=metric.capitalize(), color=color, edgecolor="white")
        for j, v in enumerate(vals):
            ax3.text(xp[j] + i * w, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax3.set_xticks(xp + w); ax3.set_xticklabels(classes, fontsize=11)
    ax3.set_ylim(0, 1.12); ax3.set_title("(c) Metrik Per Kelas", fontweight="bold")
    ax3.set_ylabel("Skor"); ax3.legend(loc="lower right")

    ax4 = axes[1, 1]; gm = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}
    bars4 = ax4.bar(gm.keys(), gm.values(), color=["#1abc9c", "#e74c3c", "#f39c12", "#8e44ad"],
                     edgecolor="white", width=0.5)
    ax4.set_ylim(0, 1.12); ax4.set_title("(d) Ringkasan Metrik Global", fontweight="bold"); ax4.set_ylabel("Skor")
    for bar in bars4:
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{bar.get_height():.4f}", ha="center", fontsize=12, fontweight="bold")

    fig.text(0.5, 0.01, f"Akurasi: {acc * 100:.2f}%  |  F1: {f1:.4f}  |  n={len(y):,} tweets  |  5-Fold CV",
              ha="center", fontsize=11, color="#2c3e50",
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1", alpha=0.8))
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig


def make_figure3(df_all, df_s, df_fold, tfidf_w, tfidf_c, n_per_class, y, y_pred, acc, prec, rec, f1):
    sample_pos = df_s[df_s["sentiment"] == "Positif"].iloc[0]
    sample_neg = df_s[df_s["sentiment"] == "Negatif"].iloc[0]

    total_raw = len(df_all)
    n_pos_raw = (df_all["sentiment"] == "Positif").sum()
    n_neg_raw = (df_all["sentiment"] == "Negatif").sum()
    n_net_raw = (df_all["sentiment"] == "Netral").sum()
    pct_pos = n_pos_raw / total_raw * 100
    pct_neg = n_neg_raw / total_raw * 100
    pct_net = n_net_raw / total_raw * 100

    try:
        word_feat_names = np.array(tfidf_w.get_feature_names_out())
        char_feat_names = np.array(tfidf_c.get_feature_names_out())
        top_word_feats = word_feat_names[:8].tolist()
        top_char_feats = char_feat_names[:8].tolist()
    except Exception:
        top_word_feats = ["good", "health", "safe", "care", "happy", "well", "great", "better"]
        top_char_feats = ["ood", "alt", "afe", "are", "app", "ell", "rea", "ett"]

    cm_g2 = confusion_matrix(y, y_pred)
    tn_g2, fp_g2, fn_g2, tp_g2 = cm_g2.ravel()

    fig3 = plt.figure(figsize=(20, 28))
    fig3.patch.set_facecolor("#ffffff")
    fig3.text(0.5, 0.985, "STEP-BY-STEP ANALISIS SENTIMEN HYBRID SVM",
              ha="center", fontsize=17, fontweight="bold", color="#1a252f")
    fig3.text(0.5, 0.978, "Penjelasan Lengkap Proses Klasifikasi: dari Tweet Mentah hingga Akurasi",
              ha="center", fontsize=12, color="#555", style="italic")
    fig3.add_artist(Line2D([0.03, 0.97], [0.974, 0.974], transform=fig3.transFigure, color="#bdc3c7", lw=1.5))

    # STEP 1
    ax_s1 = fig3.add_axes([0.03, 0.90, 0.94, 0.07])
    ax_s1.set_xlim(0, 1); ax_s1.set_ylim(0, 1); ax_s1.axis("off")
    ax_s1.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#2c3e50", alpha=0.07, zorder=0))
    ax_s1.text(0.0, 0.92, "STEP 1  —  INPUT DATA & PREPROCESSING", fontsize=12, fontweight="bold", color="#2c3e50", va="top")
    info_cols = [
        (f"{total_raw:,}\nTotal Tweet", "#3498db"),
        (f"{n_pos_raw:,}\nTweet Positif\n({pct_pos:.1f}%)", "#27ae60"),
        (f"{n_neg_raw:,}\nTweet Negatif\n({pct_neg:.1f}%)", "#e74c3c"),
        (f"{n_net_raw:,}\nTweet Netral\n(dibuang, {pct_net:.1f}%)", "#95a5a6"),
        ("3 Sumber\nStep_one, Step_two\nStep_three", "#8e44ad"),
    ]
    for k, (label, col) in enumerate(info_cols):
        xk = 0.02 + k * 0.19
        ax_s1.add_patch(FancyBboxPatch((xk, 0.03), 0.17, 0.75, boxstyle="round,pad=0.01", color=col, alpha=0.15))
        ax_s1.add_patch(FancyBboxPatch((xk, 0.03), 0.17, 0.75, boxstyle="round,pad=0.01", fill=False, edgecolor=col, lw=1.5))
        ax_s1.text(xk + 0.085, 0.42, label, ha="center", va="center", fontsize=9, fontweight="bold", color=col)

    # STEP 2
    ax_s2 = fig3.add_axes([0.03, 0.80, 0.94, 0.09])
    ax_s2.set_xlim(0, 1); ax_s2.set_ylim(0, 1); ax_s2.axis("off")
    ax_s2.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#16a085", alpha=0.05))
    ax_s2.text(0.0, 0.97, "STEP 2  —  PREPROCESSING TEKS (Contoh Transformasi)", fontsize=12, fontweight="bold", color="#16a085", va="top")
    preprocess_steps = [
        ("Tweet Asli", sample_pos["tweet"][:90] + "..." if len(sample_pos["tweet"]) > 90 else sample_pos["tweet"], "#e74c3c"),
        ("① Lowercase + hapus URL/mention/#", (sample_pos["tweet"].lower()[:90] + "..."), "#e67e22"),
        ("② Hapus karakter non-huruf", re.sub(r"[^a-z\s]", " ", sample_pos["tweet"].lower())[:90] + "...", "#f39c12"),
        ("③ Hapus stopword + filter len>2 → HASIL BERSIH",
         sample_pos["clean_tweet"][:90] + ("..." if len(sample_pos["clean_tweet"]) > 90 else ""), "#27ae60"),
    ]
    for k, (step_name, step_val, col) in enumerate(preprocess_steps):
        y_k = 0.78 - k * 0.20
        ax_s2.text(0.01, y_k, f"{step_name}:", fontsize=8.5, fontweight="bold", color=col, va="top")
        ax_s2.text(0.28, y_k, f'"{step_val}"', fontsize=8, color="#2c3e50", va="top", family="monospace",
                   bbox=dict(boxstyle="round,pad=0.2", fc=col, alpha=0.08, ec=col, lw=0.8))
        if k < 3:
            ax_s2.annotate("", xy=(0.14, y_k - 0.14), xytext=(0.14, y_k - 0.04),
                           arrowprops=dict(arrowstyle="->", color="#aaa", lw=1.2))

    # STEP 3
    ax_s3 = fig3.add_axes([0.03, 0.70, 0.94, 0.09])
    ax_s3.set_xlim(0, 1); ax_s3.set_ylim(0, 1); ax_s3.axis("off")
    ax_s3.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#8e44ad", alpha=0.05))
    ax_s3.text(0.0, 0.97, "STEP 3  —  AUTO-LABELING dengan TextBlob (Analisis Polaritas)", fontsize=12, fontweight="bold", color="#8e44ad", va="top")
    ax_s3.text(0.01, 0.72,
               "TextBlob menghitung polarity score [-1.0 s.d +1.0] untuk setiap tweet. "
               "Threshold ketat |polarity| > 0.1 memastikan hanya tweet yang jelas sentimen-nya yang masuk.",
               fontsize=9, color="#2c3e50", va="top", wrap=True)
    ax_s3.add_patch(FancyArrowPatch((0.01, 0.25), (0.99, 0.25), arrowstyle="-|>", color="#555", lw=1.5, mutation_scale=10))
    zones = [
        (0.01, 0.20, "#e74c3c", "Negatif kuat\n(-1.0 s.d -0.5)"),
        (0.21, 0.17, "#e67e22", "Negatif lemah\n(-0.5 s.d -0.1)"),
        (0.38, 0.20, "#95a5a6", "NETRAL\n(-0.1 s.d +0.1)\nDIBUANG"),
        (0.56, 0.17, "#2ecc71", "Positif lemah\n(+0.1 s.d +0.5)"),
        (0.76, 0.20, "#27ae60", "Positif kuat\n(+0.5 s.d +1.0)"),
    ]
    scale_labels = ["-1.0", "-0.5", "-0.1", "0.0", "+0.1", "+0.5", "+1.0"]
    scale_x = [0.01, 0.20, 0.37, 0.47, 0.55, 0.75, 0.97]
    for zx, zy, zcol, zlbl in zones:
        ax_s3.text(zx, zy, zlbl, fontsize=7.5, color=zcol, va="top", ha="left", fontweight="bold")
    for sx, slbl in zip(scale_x, scale_labels):
        ax_s3.text(sx, 0.30, slbl, fontsize=8, ha="center", color="#555")
        ax_s3.add_artist(Line2D([sx, sx], [0.22, 0.28], color="#bbb", lw=1))
    for tx in [0.37, 0.55]:
        ax_s3.add_artist(Line2D([tx, tx], [0.10, 0.38], color="#e74c3c", lw=2, ls="--"))
    ax_s3.text(0.38, 0.08, "threshold -0.1", fontsize=8, color="#e74c3c", ha="left")
    ax_s3.text(0.53, 0.08, "threshold +0.1", fontsize=8, color="#e74c3c", ha="right")

    # STEP 4
    ax_s4 = fig3.add_axes([0.03, 0.58, 0.94, 0.11])
    ax_s4.set_xlim(0, 1); ax_s4.set_ylim(0, 1); ax_s4.axis("off")
    ax_s4.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#2980b9", alpha=0.05))
    ax_s4.text(0.0, 0.97, "STEP 4  —  EKSTRAKSI FITUR HYBRID TF-IDF", fontsize=12, fontweight="bold", color="#2980b9", va="top")
    ax_s4.text(0.01, 0.79,
               "TF-IDF = Term Frequency × Inverse Document Frequency\n"
               "TF(t,d) = frekuensi kata t dalam dokumen d  |  "
               "IDF(t) = log(N / df(t))  →  kata langka lebih bernilai tinggi",
               fontsize=9, color="#2c3e50", va="top", family="monospace")
    ax_s4.add_patch(FancyBboxPatch((0.01, 0.05), 0.44, 0.46, boxstyle="round,pad=0.01", color="#3498db", alpha=0.12, ec="#3498db", lw=1.5))
    ax_s4.text(0.23, 0.50, "TF-IDF WORD N-GRAM (1,2,3)", ha="center", fontsize=9, fontweight="bold", color="#1a6fa8", va="top")
    ax_s4.text(0.23, 0.43, "max_features=20.000  |  sublinear_tf=True  |  min_df=2", ha="center", fontsize=8, color="#555", va="top")
    ax_s4.text(0.23, 0.37, "Contoh fitur: " + " | ".join(top_word_feats[:6]), ha="center", fontsize=8, color="#2980b9", va="top", family="monospace")
    ax_s4.text(0.23, 0.17, "Unigram: 'good', 'bad'\nBigram: 'very good', 'not safe'\nTrigram: 'feel very safe'",
               ha="center", fontsize=8, color="#444", va="top")
    ax_s4.add_patch(FancyBboxPatch((0.52, 0.05), 0.46, 0.46, boxstyle="round,pad=0.01", color="#9b59b6", alpha=0.12, ec="#9b59b6", lw=1.5))
    ax_s4.text(0.75, 0.50, "TF-IDF CHAR N-GRAM (3,4,5,6)", ha="center", fontsize=9, fontweight="bold", color="#7d3c98", va="top")
    ax_s4.text(0.75, 0.43, "max_features=12.000  |  analyzer=char_wb  |  min_df=3", ha="center", fontsize=8, color="#555", va="top")
    ax_s4.text(0.75, 0.37, "Contoh fitur: " + " | ".join([f"'{c}'" for c in top_char_feats[:5]]), ha="center", fontsize=8, color="#9b59b6", va="top", family="monospace")
    ax_s4.text(0.75, 0.17, "Menangkap pola morfologi, typo\n& variasi penulisan kata\n(mis: 'gooood', 'awful')",
               ha="center", fontsize=8, color="#444", va="top")
    ax_s4.annotate("", xy=(0.52, 0.27), xytext=(0.45, 0.27), arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
    ax_s4.text(0.485, 0.32, "hstack\n32K fitur", ha="center", fontsize=8, fontweight="bold", color="#e74c3c")

    # STEP 5
    ax_s5 = fig3.add_axes([0.03, 0.49, 0.94, 0.08])
    ax_s5.set_xlim(0, 1); ax_s5.set_ylim(0, 1); ax_s5.axis("off")
    ax_s5.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#e67e22", alpha=0.05))
    ax_s5.text(0.0, 0.97, "STEP 5  —  NORMALISASI (MaxAbsScaler) & BALANCED SAMPLING", fontsize=12, fontweight="bold", color="#e67e22", va="top")
    norm_points = [
        ("MaxAbsScaler",
         "Setiap fitur dibagi nilai absolut maksimumnya → rentang [-1, +1]\n"
         "Menjaga sparsity matrix (tidak mengubah nilai 0 menjadi bukan 0)", "#e67e22"),
        ("Balanced Sampling",
         f"Maks {n_per_class:,} tweet per kelas → total dataset seimbang\n"
         f"Positif: {(df_s['sentiment'] == 'Positif').sum():,}  |  "
         f"Negatif: {(df_s['sentiment'] == 'Negatif').sum():,}  |  "
         f"Mencegah model bias ke kelas mayoritas", "#c0392b"),
        ("class_weight='balanced'",
         "LinearSVC memberi bobot lebih ke kelas minoritas saat training\n"
         "Double protection: sampling balanced + weight balanced", "#8e44ad"),
    ]
    for k, (title, desc, col) in enumerate(norm_points):
        xk = 0.01 + k * 0.325
        ax_s5.add_patch(FancyBboxPatch((xk, 0.05), 0.31, 0.68, boxstyle="round,pad=0.01", color=col, alpha=0.10, ec=col, lw=1.2))
        ax_s5.text(xk + 0.155, 0.72, title, ha="center", fontsize=9, fontweight="bold", color=col, va="top")
        ax_s5.text(xk + 0.01, 0.58, desc, fontsize=8, color="#2c3e50", va="top")

    # STEP 6
    ax_s6 = fig3.add_axes([0.03, 0.37, 0.44, 0.11])
    ax_s6.set_xlim(0, 1); ax_s6.set_ylim(0, 1); ax_s6.axis("off")
    ax_s6.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#c0392b", alpha=0.05))
    ax_s6.text(0.0, 0.97, "STEP 6  —  CARA KERJA LinearSVC", fontsize=12, fontweight="bold", color="#c0392b", va="top")
    ax_s6.text(0.01, 0.78,
               "LinearSVC mencari HYPERPLANE optimal yang memisahkan\n"
               "kelas Positif dan Negatif dengan margin maksimum.\n\n"
               "Fungsi keputusan: f(x) = w · x + b\n"
               "  w = bobot fitur (dipelajari dari data)\n"
               "  x = vektor fitur TF-IDF tweet\n"
               "  b = bias\n\n"
               "Jika f(x) > 0 → prediksi Positif\n"
               "Jika f(x) < 0 → prediksi Negatif\n\n"
               "Parameter C=1.0: mengontrol trade-off antara\n"
               "margin lebar vs kesalahan klasifikasi",
               fontsize=8.8, color="#2c3e50", va="top", family="monospace")

    # STEP 6B
    ax_s6b = fig3.add_axes([0.52, 0.37, 0.46, 0.11])
    ax_s6b.set_xlim(0, 1); ax_s6b.set_ylim(0, 1); ax_s6b.axis("off")
    ax_s6b.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#27ae60", alpha=0.05))
    ax_s6b.text(0.0, 0.97, "STEP 6B  —  STRATIFIED 5-FOLD CROSS-VALIDATION", fontsize=12, fontweight="bold", color="#27ae60", va="top")
    n_total_cv = len(y); n_test_cv = n_total_cv // 5; n_train_cv = n_total_cv - n_test_cv
    fold_colors_cv = ["#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#e74c3c"]
    ax_s6b.text(0.01, 0.80,
                f"Total data: {n_total_cv:,} tweet  →  dibagi 5 subset sama besar\n"
                f"Tiap fold: ~{n_train_cv:,} train + ~{n_test_cv:,} test  |  Stratified = proporsi kelas sama",
                fontsize=8.5, color="#2c3e50", va="top")
    for fi in range(5):
        for fj in range(5):
            col = fold_colors_cv[fi] if fj == fi else "#ecf0f1"
            ec = fold_colors_cv[fi] if fj == fi else "#bdc3c7"
            ax_s6b.add_patch(FancyBboxPatch((0.01 + fj * 0.195, 0.17 - fi * 0.09), 0.18, 0.07,
                                            boxstyle="round,pad=0.005", color=col, alpha=0.9, ec=ec, lw=1))
            lbl = "TEST" if fj == fi else "train"
            ax_s6b.text(0.01 + fj * 0.195 + 0.09, 0.205 - fi * 0.09, lbl, ha="center", va="center", fontsize=7,
                        fontweight="bold" if fj == fi else "normal", color="#1a252f" if fj == fi else "#888")
        ax_s6b.text(0.985, 0.205 - fi * 0.09, f"Fold {fi + 1}", ha="right", va="center", fontsize=8,
                    color=fold_colors_cv[fi], fontweight="bold")

    # STEP 7
    ax_s7 = fig3.add_axes([0.03, 0.22, 0.94, 0.14])
    ax_s7.set_xlim(0, 1); ax_s7.set_ylim(0, 1); ax_s7.axis("off")
    ax_s7.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#1abc9c", alpha=0.05))
    ax_s7.text(0.0, 0.98, "STEP 7  —  PERHITUNGAN METRIK EVALUASI (Substitusi Angka Nyata)", fontsize=12, fontweight="bold", color="#1abc9c", va="top")
    ax_s7.add_patch(FancyBboxPatch((0.01, 0.04), 0.22, 0.85, boxstyle="round,pad=0.01", color="#2c3e50", alpha=0.08, ec="#2c3e50", lw=1))
    ax_s7.text(0.12, 0.91, "Confusion Matrix", ha="center", fontsize=9, fontweight="bold", color="#2c3e50", va="top")
    cm_cells = [
        (0.035, 0.70, f"TN={tn_g2:,}", "#27ae60", "Negatif benar\nprediksi Negatif"),
        (0.135, 0.70, f"FP={fp_g2:,}", "#e74c3c", "Negatif salah\nprediksi Positif"),
        (0.035, 0.40, f"FN={fn_g2:,}", "#e74c3c", "Positif salah\nprediksi Negatif"),
        (0.135, 0.40, f"TP={tp_g2:,}", "#27ae60", "Positif benar\nprediksi Positif"),
    ]
    for cx, cy, clbl, ccol, cdesc in cm_cells:
        ax_s7.add_patch(FancyBboxPatch((cx, cy - 0.10), 0.085, 0.23, boxstyle="round,pad=0.005", color=ccol, alpha=0.15, ec=ccol, lw=1.2))
        ax_s7.text(cx + 0.0425, cy + 0.08, clbl, ha="center", va="top", fontsize=9, fontweight="bold", color=ccol)
        ax_s7.text(cx + 0.0425, cy, cdesc, ha="center", va="top", fontsize=6.5, color="#555")
    ax_s7.text(0.01, 0.31, "  Aktual Negatif →", fontsize=7.5, color="#555", va="top")
    ax_s7.text(0.01, 0.21, "  Aktual Positif  →", fontsize=7.5, color="#555", va="top")
    ax_s7.text(0.045, 0.93, "Pred.Neg", fontsize=7, color="#555", ha="center")
    ax_s7.text(0.150, 0.93, "Pred.Pos", fontsize=7, color="#555", ha="center")

    metric_defs = [
        ("Accuracy", "(TP + TN)", "(TP+TN+FP+FN)", f"({tp_g2:,} + {tn_g2:,})", f"({tp_g2+tn_g2+fp_g2+fn_g2:,})",
         acc, f"{acc*100:.2f}%", "Proporsi prediksi benar dari semua tweet", "#1abc9c"),
        ("Precision", "TP", "(TP + FP)", f"{tp_g2:,}", f"({tp_g2:,} + {fp_g2:,})",
         prec, f"{prec:.4f}", "Dari semua prediksi Positif, berapa yang benar?", "#3498db"),
        ("Recall", "TP", "(TP + FN)", f"{tp_g2:,}", f"({tp_g2:,} + {fn_g2:,})",
         rec, f"{rec:.4f}", "Dari semua Positif aktual, berapa yang tertangkap?", "#e67e22"),
        ("F1-Score", "2 × Precision × Recall", "Precision + Recall", f"2 × {prec:.4f} × {rec:.4f}", f"{prec:.4f} + {rec:.4f}",
         f1, f"{f1:.4f}", "Harmonic mean Precision & Recall (balance)", "#9b59b6"),
    ]
    for k, (name, num, den, num_val, den_val, result, res_str, desc, col) in enumerate(metric_defs):
        xk = 0.25 + k * 0.185
        ax_s7.add_patch(FancyBboxPatch((xk, 0.04), 0.175, 0.85, boxstyle="round,pad=0.01", color=col, alpha=0.10, ec=col, lw=1.5))
        ax_s7.text(xk + 0.0875, 0.91, name, ha="center", fontsize=10, fontweight="bold", color=col, va="top")
        frac_y = 0.65
        ax_s7.add_artist(Line2D([xk + 0.01, xk + 0.165], [frac_y, frac_y], color=col, lw=1.2, alpha=0.6))
        ax_s7.text(xk + 0.0875, frac_y + 0.02, num, ha="center", fontsize=8, color="#2c3e50", va="bottom", family="monospace")
        ax_s7.text(xk + 0.0875, frac_y - 0.02, den, ha="center", fontsize=8, color="#2c3e50", va="top", family="monospace")
        ax_s7.text(xk + 0.0875, 0.46, "=", ha="center", fontsize=10, color=col)
        ax_s7.add_artist(Line2D([xk + 0.01, xk + 0.165], [0.43, 0.43], color=col, lw=1, ls="--", alpha=0.4))
        ax_s7.text(xk + 0.0875, 0.41, num_val, ha="center", fontsize=7.5, color="#555", va="top", family="monospace")
        ax_s7.text(xk + 0.0875, 0.31, den_val, ha="center", fontsize=7.5, color="#555", va="top", family="monospace")
        ax_s7.add_patch(FancyBboxPatch((xk + 0.01, 0.06), 0.155, 0.16, boxstyle="round,pad=0.005", color=col, alpha=0.25, ec=col, lw=1))
        ax_s7.text(xk + 0.0875, 0.20, f"= {res_str}", ha="center", fontsize=11, fontweight="bold", color=col, va="top")
        ax_s7.text(xk + 0.0875, 0.04, desc, ha="center", fontsize=6.5, color="#555", va="bottom", wrap=True)

    # STEP 8
    ax_s8 = fig3.add_axes([0.03, 0.10, 0.94, 0.11])
    ax_s8.set_xlim(0, 1); ax_s8.set_ylim(0, 1); ax_s8.axis("off")
    ax_s8.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#2c3e50", alpha=0.04))
    ax_s8.text(0.0, 0.97, "STEP 8  —  HASIL TIAP FOLD & RATA-RATA AKHIR", fontsize=12, fontweight="bold", color="#2c3e50", va="top")
    hdr_cols = ["Fold", "N-Train", "N-Test", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1-Score", "Keterangan"]
    col_x = [0.01, 0.07, 0.14, 0.20, 0.26, 0.32, 0.38, 0.44, 0.53, 0.62, 0.71, 0.80]
    col_w = [0.06, 0.07, 0.06, 0.06, 0.06, 0.06, 0.06, 0.09, 0.09, 0.09, 0.09, 0.19]
    for j, (hdr, cx) in enumerate(zip(hdr_cols, col_x)):
        ax_s8.add_patch(FancyBboxPatch((cx, 0.76), col_w[j] - 0.005, 0.14, boxstyle="square,pad=0", color="#2c3e50", alpha=0.85))
        ax_s8.text(cx + col_w[j] / 2 - 0.0025, 0.83, hdr, ha="center", va="center", fontsize=7.5, fontweight="bold", color="white")
    fold_colors_tbl = ["#eaf4fb", "#fdfefe", "#eaf4fb", "#fdfefe", "#eaf4fb"]
    for ri, (_, r) in enumerate(df_fold.iterrows()):
        ry = 0.62 - ri * 0.12
        best_acc = df_fold["accuracy"].max()
        row_vals = [
            f"Fold {int(r.fold)}", f"{int(r.n_train):,}", f"{int(r.n_test):,}",
            f"{r.tp:,}", f"{r.tn:,}", f"{r.fp:,}", f"{r.fn:,}",
            f"{r.accuracy:.4f}", f"{r.precision:.4f}", f"{r.recall:.4f}", f"{r.f1:.4f}",
            "★ Terbaik" if r.accuracy == best_acc else f"Δ={abs(r.accuracy - acc):.4f} dari mean"
        ]
        note_col = "#27ae60" if r.accuracy == best_acc else "#555"
        for j, (val, cx) in enumerate(zip(row_vals, col_x)):
            ax_s8.add_patch(FancyBboxPatch((cx, ry), col_w[j] - 0.005, 0.10, boxstyle="square,pad=0", color=fold_colors_tbl[ri], alpha=0.9, ec="#ddd", lw=0.5))
            col_v = note_col if j == 11 else ("#1a252f" if j < 7 else "#1a6fa8")
            ax_s8.text(cx + col_w[j] / 2 - 0.0025, ry + 0.05, val, ha="center", va="center", fontsize=7,
                       color=col_v, fontweight="bold" if j == 11 and r.accuracy == best_acc else "normal")
    mean_ry = 0.62 - 5 * 0.12
    mean_vals = ["MEAN", "", "",
                 f"{df_fold.tp.mean():.0f}", f"{df_fold.tn.mean():.0f}",
                 f"{df_fold.fp.mean():.0f}", f"{df_fold.fn.mean():.0f}",
                 f"{df_fold.accuracy.mean():.4f}", f"{df_fold.precision.mean():.4f}",
                 f"{df_fold.recall.mean():.4f}", f"{df_fold.f1.mean():.4f}",
                 f"Std: ±{df_fold.accuracy.std():.4f}"]
    for j, (val, cx) in enumerate(zip(mean_vals, col_x)):
        ax_s8.add_patch(FancyBboxPatch((cx, mean_ry), col_w[j] - 0.005, 0.10, boxstyle="square,pad=0", color="#d5f5e3", alpha=0.95, ec="#27ae60", lw=1))
        ax_s8.text(cx + col_w[j] / 2 - 0.0025, mean_ry + 0.05, val, ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1a5e39")

    # STEP 9
    ax_s9 = fig3.add_axes([0.03, 0.02, 0.94, 0.07])
    ax_s9.set_xlim(0, 1); ax_s9.set_ylim(0, 1); ax_s9.axis("off")
    ax_s9.add_patch(mpatches.Rectangle((0, 0), 1, 1, color="#1a252f", alpha=0.08))
    ax_s9.text(0.5, 0.93, "STEP 9  —  KESIMPULAN AKHIR", ha="center", fontsize=12, fontweight="bold", color="#1a252f", va="top")
    kesimpulan = [
        (f"Akurasi\n{acc*100:.2f}%", f"(TP+TN)/(Total)\n= ({tp_g2:,}+{tn_g2:,})/{tp_g2+tn_g2+fp_g2+fn_g2:,}", "#1abc9c"),
        (f"F1-Score\n{f1:.4f}", "Harmonic mean\nPrecision & Recall\nBalanced metric", "#9b59b6"),
        ("Validasi\n5-Fold CV", f"Mean acc: {df_fold.accuracy.mean():.4f}\nStd: ±{df_fold.accuracy.std():.4f}\nModel stabil", "#3498db"),
        ("Fitur\n32.000", "Word: 20K + Char: 12K\nhstack sparse matrix\nMaxAbsScaler", "#e67e22"),
    ]
    for k, (title, detail, col) in enumerate(kesimpulan):
        xk = 0.01 + k * 0.245
        ax_s9.add_patch(FancyBboxPatch((xk, 0.04), 0.235, 0.64, boxstyle="round,pad=0.01", color=col, alpha=0.15, ec=col, lw=2))
        ax_s9.text(xk + 0.1175, 0.68, title, ha="center", va="top", fontsize=11, fontweight="bold", color=col)
        ax_s9.text(xk + 0.1175, 0.42, detail, ha="center", va="top", fontsize=8, color="#2c3e50")

    return fig3


# ──────────────────────────────────────────────────────────────────
# UI STREAMLIT
# ──────────────────────────────────────────────────────────────────
st.title("Analisis Sentimen Hybrid SVM")
st.caption(
    "TF-IDF (Word N-gram + Char N-gram)  ·  LinearSVC  ·  Stratified 5-Fold Cross-Validation  "
    "·  Dataset Tweet COVID-19 London"
)
st.markdown('<div class="header-divider"></div>', unsafe_allow_html=True)

with st.expander("Tentang Aplikasi", expanded=False):
    st.markdown(
        "Aplikasi ini menjalankan pipeline analisis sentimen secara interaktif. "
        "File **Step_one.csv**, **Step_two.csv**, dan **Step_three.csv** harus "
        "berada di folder yang sama dengan `app.py`.\n\n"
        "**Parameter Model:**\n"
        "- TF-IDF Word N-gram (1–3), max_features = 20.000\n"
        "- TF-IDF Char N-gram (3–6, char\\_wb), max_features = 12.000\n"
        "- Threshold labeling TextBlob: |polarity| > 0.1\n"
        "- Balanced sampling, maksimum 7.000 tweet per kelas\n"
        "- LinearSVC (C = 1.0, class\\_weight = 'balanced')\n"
        "- Stratified 5-Fold Cross-Validation"
    )

run_btn = st.button("Jalankan Analisis", type="primary")

if run_btn:
    st.session_state["run_pipeline"] = True

if st.session_state.get("run_pipeline"):

    # ── 1. Load data ──
    with st.spinner("Memuat dan menggabungkan data..."):
        df_all_raw, missing = load_and_merge_data()

    if missing:
        st.error(
            "File berikut tidak ditemukan di folder yang sama dengan `app.py`:\n\n"
            + "\n".join(f"- `{m}`" for m in missing)
            + f"\n\nPastikan file CSV berada di: `{BASE_DIR}`"
        )
        st.stop()

    st.success(f"Total data dimuat: **{len(df_all_raw):,}** baris tweet")

    # ── 2. Preprocessing & labeling ──
    with st.spinner("Preprocessing teks & auto-labeling dengan TextBlob (mungkin perlu beberapa menit)..."):
        df_all = run_preprocessing(df_all_raw)

    n_pos = (df_all["sentiment"] == "Positif").sum()
    n_neg = (df_all["sentiment"] == "Negatif").sum()
    n_net = (df_all["sentiment"] == "Netral").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Tweet Bersih", f"{len(df_all):,}")
    col2.metric("Positif", f"{n_pos:,}")
    col3.metric("Negatif", f"{n_neg:,}")
    col4.metric("Netral (dibuang)", f"{n_net:,}")

    # ── 3. Balanced sampling ──
    with st.spinner("Membuat sampel balanced..."):
        df_s, n_per_class = build_balanced_sample(df_all)

    st.info(f"Sampel terlatih (balanced, maks {n_per_class:,}/kelas): **{len(df_s):,}** tweet — "
            f"{df_s['sentiment'].value_counts().to_dict()}")

    texts = df_s["clean_tweet"].tolist()
    le = LabelEncoder()
    y = le.fit_transform(df_s["sentiment"])
    classes = le.classes_

    # ── 4. Ekstraksi fitur ──
    with st.spinner("Ekstraksi fitur TF-IDF (Word + Char N-gram)..."):
        X_sc, tfidf_w, tfidf_c = extract_features(tuple(texts))

    st.info(f"Shape fitur hybrid: **{X_sc.shape[0]:,} × {X_sc.shape[1]:,}** (Word 20K + Char 12K)")

    # ── 5. Training & evaluasi ──
    cache_key = f"{len(texts)}_{X_sc.shape[1]}"
    with st.spinner("Melatih LinearSVC dengan Stratified 5-Fold Cross-Validation..."):
        y_pred, df_fold = train_and_evaluate(cache_key, X_sc, y)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted")
    f1 = f1_score(y, y_pred, average="weighted")

    st.markdown("## Hasil Evaluasi Model")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc*100:.2f}%")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1-Score", f"{f1:.4f}")

    with st.expander("Classification Report"):
        report_txt = classification_report(y, y_pred, target_names=classes, zero_division=0)
        st.code(report_txt)

    st.markdown("### Detail Per Fold")
    df_fold_display = df_fold.copy()
    df_fold_display = df_fold_display[
        ["fold", "n_train", "n_test", "tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "f1"]
    ]
    st.dataframe(df_fold_display, use_container_width=True, hide_index=True)

    mean_row = {
        "fold": "Mean", "n_train": "", "n_test": "",
        "tp": round(df_fold.tp.mean()), "tn": round(df_fold.tn.mean()),
        "fp": round(df_fold.fp.mean()), "fn": round(df_fold.fn.mean()),
        "accuracy": round(df_fold.accuracy.mean(), 4),
        "precision": round(df_fold.precision.mean(), 4),
        "recall": round(df_fold.recall.mean(), 4),
        "f1": round(df_fold.f1.mean(), 4),
    }
    std_row = {
        "fold": "Std Dev", "n_train": "", "n_test": "", "tp": "", "tn": "", "fp": "", "fn": "",
        "accuracy": round(df_fold.accuracy.std(), 4),
        "precision": round(df_fold.precision.std(), 4),
        "recall": round(df_fold.recall.std(), 4),
        "f1": round(df_fold.f1.std(), 4),
    }
    st.dataframe(pd.DataFrame([mean_row, std_row]), use_container_width=True, hide_index=True)

    # ── 6. Visualisasi (3 Figure) ──
    st.markdown("## Visualisasi")

    tab1, tab2, tab3 = st.tabs([
        "Figure 1 — Cara Klasifikasi & Perhitungan",
        "Figure 2 — Hasil Analisis",
        "Figure 3 — Step-by-Step Lengkap",
    ])

    with tab1:
        with st.spinner("Membuat Figure 1..."):
            fig1 = make_figure1(df_all, df_s, df_fold, X_sc, y, y_pred, classes, acc, prec, rec, f1)
        st.pyplot(fig1, use_container_width=True)
        import io as _io
        _buf1 = _io.BytesIO()
        fig1.savefig(_buf1, format="png", dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
        st.download_button("Unduh Figure 1 (PNG)", _buf1.getvalue(), file_name="cara_klasifikasi_perhitungan.png", mime="image/png")
        plt.close(fig1)

    with tab2:
        with st.spinner("Membuat Figure 2..."):
            fig2 = make_figure2(df_all, y, classes, y_pred, acc, prec, rec, f1)
        st.pyplot(fig2, use_container_width=True)
        _buf2 = _io.BytesIO()
        fig2.savefig(_buf2, format="png", dpi=150, bbox_inches="tight")
        st.download_button("Unduh Figure 2 (PNG)", _buf2.getvalue(), file_name="hasil_analisis_sentimen_hybrid_svm.png", mime="image/png")
        plt.close(fig2)

    with tab3:
        with st.spinner("Membuat Figure 3 (step-by-step, ukuran besar)..."):
            fig3 = make_figure3(df_all, df_s, df_fold, tfidf_w, tfidf_c, n_per_class, y, y_pred, acc, prec, rec, f1)
        st.pyplot(fig3, use_container_width=True)
        _buf3 = _io.BytesIO()
        fig3.savefig(_buf3, format="png", dpi=150, bbox_inches="tight", facecolor="white")
        st.download_button("Unduh Figure 3 (PNG)", _buf3.getvalue(), file_name="stepbystep_analisis_klasifikasi.png", mime="image/png")
        plt.close(fig3)

    # ── 7. Data hasil prediksi ──
    st.markdown("## Data Hasil Prediksi")
    df_s_out = df_s.copy()
    df_s_out["sentimen_prediksi"] = classes[y_pred]
    df_result = df_s_out[["tweet", "clean_tweet", "sentiment", "sentimen_prediksi", "source"]]
    st.dataframe(df_result.head(200), use_container_width=True, hide_index=True)
    st.caption(f"Menampilkan 200 dari {len(df_result):,} baris. Unduh CSV lengkap di bawah.")

    csv_bytes = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Unduh Hasil Prediksi Lengkap (CSV)",
        data=csv_bytes,
        file_name="hasil_prediksi_sentimen.csv",
        mime="text/csv",
    )

    st.success(f"Proses selesai — Akurasi: {acc*100:.2f}%")

else:
    st.info(
        "Klik tombol **Jalankan Analisis** untuk memulai pipeline. "
        "Pastikan file **Step_one.csv**, **Step_two.csv**, dan **Step_three.csv** "
        "berada di folder yang sama dengan `app.py`."
    )
