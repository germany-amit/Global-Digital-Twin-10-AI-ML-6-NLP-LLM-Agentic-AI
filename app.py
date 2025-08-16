# app.py
# Unified Web App: Planet Digital Twin + ML/DL Benchmark + 6 LLM Summaries + 6 Agentic AI Analyses
# Designed for Streamlit Cloud (free tier): lightweight defaults, lazy-loading models, and no-warnings hygiene.

import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    mean_squared_error, r2_score
)

# ---------------------------
# Optional XGBoost (fallback if not installed)
# ---------------------------
XGB_AVAILABLE = True
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGB_AVAILABLE = False

# ---------------------------
# Sumy summarizers (for Agentic section)
# ---------------------------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# ---------------------------
# NLTK runtime resources (punkt split is now punkt + punkt_tab)
# ---------------------------
import nltk
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif pkg == "punkt_tab":
            nltk.data.find("tokenizers/punkt_tab")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

# ---------------------------
# STREAMLIT BASE CONFIG
# ---------------------------
st.set_page_config(page_title="🌍 Planet Digital Twin + ML/LLM Lab", layout="wide")
st.title("🌍 Planet Digital Twin + 🧪 ML/LLM Lab + 🕵️ Agentic Analyses")
st.caption("Real-time data • classic ML benchmarks • 6 Hugging Face LLM summaries • 6 agent roles")

# ---------------------------
# CACHING HELPERS
# ---------------------------
@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def fetch_csv(url: str):
    return pd.read_csv(url)

@st.cache_data(ttl=15 * 60, show_spinner=False)
def get_json(url, params=None):
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def geocode_city(city_name: str):
    if not city_name:
        return None
    url = "https://geocoding-api.open-meteo.com/v1/search"
    js = get_json(url, params={"name": city_name, "count": 1, "language": "en", "format": "json"})
    if js.get("results"):
        r = js["results"][0]
        return (r["latitude"], r["longitude"], f'{r["name"]}, {r.get("country_code","")}')
    return None

@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_covid_global():
    url = "https://disease.sh/v3/covid-19/all"
    js = get_json(url)
    return {
        "cases": int(js.get("cases", 0)),
        "deaths": int(js.get("deaths", 0)),
        "recovered": int(js.get("recovered", 0)),
    }

@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_earthquakes(min_mag: float = 1.0):
    # USGS day feed for given magnitude threshold (1.0 by default)
    thresh = "1.0" if min_mag < 2 else ("2.5" if min_mag < 4.5 else "4.5")
    url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{thresh}_day.geojson"
    js = get_json(url)
    feats = js.get("features", [])
    rows = []
    for f in feats:
        p = f.get("properties", {})
        g = f.get("geometry", {}) or {}
        coords = g.get("coordinates") or [None, None, None]
        rows.append({
            "time": p.get("time"),
            "mag": p.get("mag"),
            "place": p.get("place"),
            "lon": coords[0],
            "lat": coords[1],
            "depth_km": coords[2],
            "id": f.get("id")
        })
    df = pd.DataFrame(rows).dropna(subset=["lat","lon"])
    return df

@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_weather(lat: float, lon: float):
    url = "https://api.open-meteo.com/v1/forecast"
    js = get_json(url, params={
        "latitude": lat, "longitude": lon,
        "current_weather": True,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "timezone": "auto"
    })
    cw = js.get("current_weather")
    if not cw:
        return {"ok": False}
    return {
        "temperature": cw.get("temperature"),
        "windspeed": cw.get("windspeed"),
        "winddirection": cw.get("winddirection"),
        "time": cw.get("time"),
        "ok": True
    }

# ---------------------------
# DATASETS FOR BENCHMARKS
# ---------------------------
DATASETS = {
    "Finance/Insurance (Titanic)": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    "Healthcare (Diabetes)": "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    "Marketing (Iris)": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
}

# ---------------------------
# MODEL FACTORIES (≈10 classic models per task)
# ---------------------------
def is_classification(y: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(y):
        return y.nunique() <= 10
    return True


def get_classification_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "KNN": KNeighborsClassifier(),
        "GaussianNB": GaussianNB(),
        "Linear Discriminant": LinearDiscriminantAnalysis(),
        "SVM (RBF)": SVC(probability=True),
        "MLP (Neural Net)": MLPClassifier(max_iter=1000),
    }
    if XGB_AVAILABLE:
        models["XGBoost Classifier"] = XGBClassifier(eval_metric="logloss")
    return models


def get_regression_models():
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "KNN Regressor": KNeighborsRegressor(),
        "SVR (RBF)": SVR(),
        "MLP Regressor": MLPRegressor(max_iter=1000),
    }
    if XGB_AVAILABLE:
        models["XGBoost Regressor"] = XGBRegressor()
    return models

# ---------------------------
# HUGGING FACE LLM PIPELINES (lazy load per model)
# ---------------------------
LIGHT_MODELS = {
    # name: (task, model_id)
    "distilgpt2": ("text-generation", "distilgpt2"),
    "t5-small": ("summarization", "t5-small"),
    "google/flan-t5-small": ("summarization", "google/flan-t5-small"),
    "sshleifer/distilbart-cnn-12-6": ("summarization", "sshleifer/distilbart-cnn-12-6"),
    "philschmid/bart-large-cnn-samsum": ("summarization", "philschmid/bart-large-cnn-samsum"),
    "google/pegasus-xsum": ("summarization", "google/pegasus-xsum"),
}

@st.cache_resource(show_spinner=True)
def load_pipeline(model_name: str, task: str):
    from transformers import pipeline
    return pipeline(task, model=model_name)


def summarize_with_pipeline(p, text: str):
    task = getattr(p, "task", "")
    if "generation" in task:
        out = p(text, max_length=120, num_return_sequences=1)[0]["generated_text"]
        if "." in out:
            out = out[: out.rfind(".") + 1]
        return out.strip()
    else:
        out = p(text, max_length=140, min_length=45, do_sample=False)[0]["summary_text"]
        return out.strip()

# ---------------------------
# AGENTIC AI ROLES (6 agents over same context)
# ---------------------------
LANGUAGE = "english"
STEMMER = Stemmer(LANGUAGE)

SUMY_ENGINES = {
    "LSA": LsaSummarizer(STEMMER),
    "LexRank": LexRankSummarizer(STEMMER),
    "Luhn": LuhnSummarizer(STEMMER),
    "TextRank": TextRankSummarizer(STEMMER),
}
for s in SUMY_ENGINES.values():
    s.stop_words = get_stop_words(LANGUAGE)


def sumy_summarize(text: str, engine_name: str, sentences: int = 3) -> str:
    parser = PlaintextParser.from_string(text, SumyTokenizer(LANGUAGE))
    engine = SUMY_ENGINES[engine_name]
    summary = engine(parser.document, sentences)
    return " ".join(str(s) for s in summary)


def key_points(text: str, top_k: int = 6) -> list:
    # Simple keyword extraction via TF-IDF-like scoring
    from collections import Counter
    import re
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z\-]+", text)]
    counts = Counter(tokens)
    return [w for w, _ in counts.most_common(top_k)]


def run_agents(context: str, city_label: str) -> dict:
    results = {}
    # 1) Researcher (evidence-first) — LexRank
    results["Researcher (LexRank)"] = sumy_summarize(context, "LexRank", 3)

    # 2) Risk Analyst — highlight hazards
    hazards = [w for w in key_points(context, 10) if w in {"quake", "earthquake", "heatwaves", "monsoon", "flood", "storm", "covid", "deaths"}]
    results["Risk Analyst"] = (
        f"Primary hazards for {city_label}: " + (", ".join(sorted(set(hazards))) or "none flagged").strip()
        + ". Suggested actions: monitor alerts, prepare emergency kit, verify local advisories."
    )

    # 3) Economist — macro snapshot
    results["Economist"] = (
        "Markets mixed; consider volatility from weather extremes and health updates. "
        "Focus on energy demand, insurance exposure, and supply chain sensitivity."
    )

    # 4) Meteorologist — extract weather line
    for line in context.split("."):
        if "Local weather" in line or "weather" in line:
            results["Meteorologist"] = line.strip() + "."
            break
    results.setdefault("Meteorologist", "Weather details unavailable in context.")

    # 5) Humanitarian Ops — actionable checklist
    results["Humanitarian Ops"] = (
        "Checklist: (1) Validate incident locations, (2) Contact local partners, "
        "(3) Stage water and medical kits, (4) Share verified info with communities."
    )

    # 6) Editor — LSA summary with bullet polish
    ed_sum = sumy_summarize(context, "LSA", 3)
    results["Editor (LSA)"] = ed_sum

    return results

# ---------------------------
# UI TABS
# ---------------------------
T1, T2, T3, T4 = st.tabs([
    "🗺️ Planet Digital Twin",
    "📊 ML/DL Benchmark",
    "🧾 6 LLM Summaries",
    "🕵️ 6 Agentic AI"
])

# =========================================================
# TAB 1: PLANET DIGITAL TWIN
# =========================================================
with T1:
    colA, colB = st.columns([2, 1])
    with colB:
        city_in = st.text_input("City for local weather", value="New Delhi")
        min_mag = st.slider("Min earthquake magnitude (USGS day feed)", 0.0, 5.0, 1.0, 0.5)

    geo = geocode_city(city_in)
    if geo is None:
        st.warning("City not found. Falling back to New Delhi.")
        lat, lon, city_label = 28.6139, 77.2090, "New Delhi (fallback)"
    else:
        lat, lon, city_label = geo

    with st.spinner("Fetching global feeds..."):
        covid = fetch_covid_global()
        quakes = fetch_earthquakes(min_mag=min_mag)
        weather = fetch_weather(lat, lon)

    with colA:
        st.subheader("🌋 Earthquakes (last 24h)")
        if not quakes.empty:
            # World map via st.map (fast & lightweight)
            map_df = quakes[["lat", "lon"]].copy()
            st.map(map_df, zoom=1, use_container_width=True)
            st.dataframe(quakes[["place", "mag", "depth_km"]].sort_values("mag", ascending=False).head(15), use_container_width=True)
        else:
            st.info("No earthquakes returned for the selected threshold.")

    st.markdown("---")
    st.subheader("🦠 COVID-19 (global snapshot)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cases", f"{covid['cases']:,}")
    c2.metric("Deaths", f"{covid['deaths']:,}")
    c3.metric("Recovered", f"{covid['recovered']:,}")

    fig, ax = plt.subplots()
    ax.bar(["Cases", "Deaths", "Recovered"], [covid["cases"], covid["deaths"], covid["recovered"]])
    ax.set_title("Global COVID-19 Totals")
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("🌦️ Local Weather")
    if weather.get("ok"):
        st.success(f"{city_label}: {weather['temperature']}°C, wind {weather['windspeed']} km/h (dir {weather['winddirection']}°) at {weather['time']}")
    else:
        st.warning(f"Local weather for {city_label} is unavailable.")

# =========================================================
# TAB 2: ML/DL BENCHMARK (≈10 models)
# =========================================================
with T2:
    st.subheader("Benchmark classic AI/ML/DL models on a CSV")

    left, right = st.columns([1, 1])
    with left:
        dataset_name = st.radio("Choose a dataset:", list(DATASETS.keys()))
        df = fetch_csv(DATASETS[dataset_name])
        st.write("### Preview")
        st.dataframe(df.head(), use_container_width=True)

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.shape[1] < 2:
        st.error("This dataset doesn't have enough numeric columns after dropping NAs.")
        st.stop()

    with right:
        target_col = st.radio("Choose target column:", numeric_df.columns, index=len(numeric_df.columns)-1)
        X = numeric_df.drop(columns=[target_col])
        y = numeric_df[target_col]

        problem_is_classification = is_classification(y)
        st.info(f"Detected task: **{'Classification' if problem_is_classification else 'Regression'}**")

        mode = st.radio("Mode:", ["Run All Models (Leaderboard)", "Run Single Model"]) 
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y if problem_is_classification else None
    )

    do_scale = st.checkbox("Standardize features (recommended for SVM/MLP/KNN)", value=True)
    if do_scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if mode.startswith("Run All"):
        models = get_classification_models() if problem_is_classification else get_regression_models()
        rows = []
        for name, model in models.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if problem_is_classification:
                    acc = accuracy_score(y_test, preds)
                    auc = None
                    try:
                        if hasattr(model, "predict_proba"):
                            probs = model.predict_proba(X_test)
                            if probs.shape[1] == 2:
                                auc = roc_auc_score(y_test, probs[:, 1])
                    except Exception:
                        pass
                    rows.append({"Model": name, "Primary": acc, "Aux": auc})
                else:
                    mse = mean_squared_error(y_test, preds)
                    r2 = r2_score(y_test, preds)
                    rows.append({"Model": name, "Primary": -mse, "Aux": r2})  # higher better
            except Exception:
                rows.append({"Model": name, "Primary": np.nan, "Aux": np.nan})

        res_df = pd.DataFrame(rows)
        st.write("## 🏆 Leaderboard")
        st.dataframe(res_df.sort_values("Primary", ascending=False), use_container_width=True)

        plot_df = res_df.dropna().sort_values("Primary", ascending=True)
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.barh(plot_df["Model"], plot_df["Primary"]) 
        ax2.set_xlabel("Score (Accuracy or -MSE)")
        ax2.set_title("Model Comparison")
        st.pyplot(fig2)
        plt.close(fig2)

        if st.checkbox("Run PCA (2D) + KMeans (k=2)"):
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            km = KMeans(n_clusters=2, n_init=10, random_state=42)
            clusters = km.fit_predict(X)
            st.write("PCA sample (first 10 rows):")
            st.write(pd.DataFrame(X_pca, columns=["PC1", "PC2"]).head(10))
            st.write("KMeans cluster labels (first 20):", clusters[:20].tolist())

    else:
        models = get_classification_models() if problem_is_classification else get_regression_models()
        model_name = st.radio("Choose a single model:", list(models.keys()))
        model = models[model_name]
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if problem_is_classification:
                acc = accuracy_score(y_test, preds)
                st.success(f"{model_name} — Accuracy: {acc:.4f}")
            else:
                mse = mean_squared_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.success(f"{model_name} — MSE: {mse:.4f}, R²: {r2:.4f}")
        except Exception as e:
            st.error(f"Error training {model_name}: {e}")

# =========================================================
# TAB 3: 6 HUGGING FACE LLM SUMMARIES
# =========================================================
with T3:
    st.subheader("Six Hugging Face models summarize the same global context")

    with st.sidebar:
        st.header("Global Context Settings")
        city_in = st.text_input("City for local weather (LLM tab)", value="New Delhi", key="city_llm")
        view = st.radio("Mode", ["Choose One (recommended)", "Run All 6 (heavy)"])

    # Build global context (reuse helpers)
    geo = geocode_city(city_in)
    if geo is None:
        st.warning("City not found. Falling back to New Delhi.")
        lat, lon, city_label = 28.6139, 77.2090, "New Delhi (fallback)"
    else:
        lat, lon, city_label = geo

    with st.spinner("Fetching global data..."):
        covid = fetch_covid_global()
        quakes = fetch_earthquakes()
        weather = fetch_weather(lat, lon)

    quakes_n = len(quakes)
    top_mag = float(np.nanmax(quakes["mag"])) if quakes_n else 0.0
    regions = (quakes["place"].dropna().head(8).tolist()) if quakes_n else []

    covid_str = f"COVID-19 global snapshot — cases: {covid['cases']:,}, deaths: {covid['deaths']:,}, recovered: {covid['recovered']:,}."
    quake_str = (
        f"USGS lists {quakes_n} earthquakes (M≥1) in the past 24 hours; strongest ≈ {top_mag:.1f}. Sample regions: {', '.join(regions[:5])}."
        if quakes_n else "USGS shows no M≥1 earthquakes in the past 24 hours."
    )
    weather_str = (
        f"Local weather in {city_label}: {weather['temperature']}°C, wind {weather['windspeed']} km/h (dir {weather['winddirection']}°)."
        if weather.get("ok") else f"Local weather for {city_label} is unavailable."
    )
    news_stub = "Global news: heatwaves in Europe, heavy monsoon rains across South Asia, tech markets mixed."

    global_context = " ".join([covid_str, quake_str, weather_str, news_stub, "Provide a concise, data-driven update."])

    st.markdown("##### Combined Context (sent to models)")
    with st.expander("Show context"):
        st.write(global_context)

    if view.startswith("Choose One"):
        model_choice = st.selectbox("Pick a model", list(LIGHT_MODELS.keys()))
        task, model_id = LIGHT_MODELS[model_choice]
        if st.button(f"Generate with {model_choice}", type="primary"):
            try:
                pipe = load_pipeline(model_id, task)
                out = summarize_with_pipeline(pipe, global_context)
                st.markdown(f"### 🤖 Summary from **{model_choice}**")
                st.info(out)
            except Exception as e:
                st.error(f"{model_choice} failed: {e}")
    else:
        if st.button("Generate with all 6 models", type="primary"):
            for model_choice, (task, model_id) in LIGHT_MODELS.items():
                st.markdown(f"### 🤖 Summary from **{model_choice}**")
                try:
                    pipe = load_pipeline(model_id, task)
                    out = summarize_with_pipeline(pipe, global_context)
                    st.info(out)
                except Exception as e:
                    st.error(f"{model_choice} failed: {e}")
        else:
            st.info("This may be slow on free tier. Use 'Choose One' for faster results.")

# =========================================================
# TAB 4: 6 AGENTIC AI ROLES (Sumy + heuristics)
# =========================================================
with T4:
    st.subheader("Six different Agentic AI roles analyze the same context")

    city_in = st.text_input("City for local weather (Agents tab)", value="New Delhi", key="city_agents")
    geo = geocode_city(city_in)
    if geo is None:
        st.warning("City not found. Falling back to New Delhi.")
        lat, lon, city_label = 28.6139, 77.2090, "New Delhi (fallback)"
    else:
        lat, lon, city_label = geo

    with st.spinner("Fetching context feeds..."):
        covid = fetch_covid_global()
        quakes = fetch_earthquakes()
        weather = fetch_weather(lat, lon)

    quakes_n = len(quakes)
    top_mag = float(np.nanmax(quakes["mag"])) if quakes_n else 0.0
    regions = (quakes["place"].dropna().head(8).tolist()) if quakes_n else []

    covid_str = f"COVID-19 global snapshot — cases: {covid['cases']:,}, deaths: {covid['deaths']:,}, recovered: {covid['recovered']:,}."
    quake_str = (
        f"USGS lists {quakes_n} earthquakes (M≥1) in the past 24 hours; strongest ≈ {top_mag:.1f}. Sample regions: {', '.join(regions[:5])}."
        if quakes_n else "USGS shows no M≥1 earthquakes in the past 24 hours."
    )
    weather_str = (
        f"Local weather in {city_label}: {weather['temperature']}°C, wind {weather['windspeed']} km/h (dir {weather['winddirection']}°)."
        if weather.get("ok") else f"Local weather for {city_label} is unavailable."
    )
    news_stub = "Global news: heatwaves in Europe, heavy monsoon rains across South Asia, tech markets mixed."

    context_agents = " ".join([covid_str, quake_str, weather_str, news_stub])

    st.markdown("##### Combined Context (agents use this)")
    with st.expander("Show context"):
        st.write(context_agents)

    if st.button("Run 6 Agents", type="primary"):
        results = run_agents(context_agents, city_label)
        for role, output in results.items():
            with st.container():
                st.markdown(f"### {role}")
                st.info(output)
    else:
        st.info("Click to generate analyses from six roles (LexRank, LSA, risk, met, ops, econ).")
