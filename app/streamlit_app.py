"""Streamlit deployment app for Burundi harvest prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import build_inference_input  # noqa: E402


MODEL_CONFIG = {
    "Decision Tree": {
        "key": "decision_tree",
        "path": PROJECT_ROOT / "models" / "decision_tree.pkl",
    },
    "Random Forest": {
        "key": "random_forest",
        "path": PROJECT_ROOT / "models" / "random_forest.pkl",
    },
    "Logistic Regression": {
        "key": "logistic_regression",
        "path": PROJECT_ROOT / "models" / "logistic_regression.pkl",
    },
}

PROVINCES = [
    "Bubanza",
    "Bujumbura Rural",
    "Bururi",
    "Cankuzo",
    "Cibitoke",
    "Gitega",
    "Kayanza",
    "Kirundo",
    "Makamba",
    "Muramvya",
    "Muyinga",
    "Mwaro",
    "Ngozi",
    "Rutana",
    "Ruyigi",
]

CULTURES = ["Maïs", "Haricot", "Manioc", "Patate douce", "Sorgho", "Bananier"]


st.set_page_config(
    page_title="Prédiction des Récoltes — Burundi",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f6f8f4;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    [data-testid="stSidebar"] {
        background: #10251b;
    }
    [data-testid="stSidebar"] * {
        color: #f5f7f2;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #ffffff;
        border-color: #d9e3d7;
        color: #111827 !important;
        -webkit-text-fill-color: #111827;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #d9e3d7;
        color: #111827;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #111827 !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #10251b;
        fill: #10251b;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button {
        background: #edf4ed !important;
        border-color: #d9e3d7 !important;
        color: #173b2a !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:hover,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button:focus {
        background: #dfeade !important;
        border-color: #b9cbb7 !important;
        color: #0f241a !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button svg,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] button span {
        color: #173b2a !important;
        fill: #173b2a !important;
        opacity: 1 !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
        background: #d6a640 !important;
        border-color: #f4ead0 !important;
        box-shadow: 0 0 0 2px rgba(214, 166, 64, 0.22) !important;
    }
    [data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"]:hover {
        background: #c8952c !important;
        border-color: #f7edcf !important;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #2f7d46;
        border: 1px solid #2f7d46;
        color: #ffffff;
        font-weight: 700;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #27693b;
        border-color: #27693b;
        color: #ffffff;
    }
    .app-header {
        border-left: 6px solid #2f7d46;
        padding: 0.25rem 0 0.25rem 1rem;
        margin-bottom: 1.5rem;
    }
    .app-subtitle {
        color: #4b5563;
        font-size: 1.05rem;
        margin-top: -0.5rem;
    }
    .result-banner {
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .result-good {
        background: #e6f4ea;
        border: 1px solid #2f7d46;
        color: #174a2a;
    }
    .result-bad {
        background: #fdecec;
        border: 1px solid #b42318;
        color: #7a271a;
    }
    div[data-testid="stProgress"] > div > div > div {
        background-color: #2f7d46;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1e8dc;
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load fitted models and preprocessing artifacts once per Streamlit session."""
    models = {
        model_name: joblib.load(config["path"])
        for model_name, config in MODEL_CONFIG.items()
    }
    scaler = joblib.load(PROJECT_ROOT / "models" / "scaler.pkl")
    feature_columns = joblib.load(PROJECT_ROOT / "models" / "feature_columns.pkl")
    numerical_cols = list(scaler.feature_names_in_)
    return models, scaler, feature_columns, numerical_cols


@st.cache_data
def load_metrics():
    """Load saved evaluation metrics from metrics.json."""
    with (PROJECT_ROOT / "metrics.json").open("r", encoding="utf-8") as file:
        return json.load(file)


def canonicalize_culture(culture: str) -> str:
    """Map the UI spelling of maize to the exact spelling stored in model columns."""
    if culture == "Maïs":
        maize_columns = [
            column for column in load_artifacts()[2] if column.startswith("culture_Ma")
        ]
        if maize_columns:
            return maize_columns[0].replace("culture_", "", 1)
    return culture


def build_raw_input(
    province: str,
    culture: str,
    altitude: int,
    pluviometrie: int,
    temperature: float,
    superficie: float,
    nb_menages: int,
    annee: int,
    saison: str,
    utilisation_engrais: bool,
    acces_irrigation: bool,
) -> dict:
    """Build the raw input dictionary expected by build_inference_input()."""
    return {
        "annee": annee,
        "saison": saison,
        "province": province,
        "culture": canonicalize_culture(culture),
        "altitude_m": altitude,
        "pluviometrie_mm": pluviometrie,
        "temperature_moy_C": temperature,
        "superficie_ha": superficie,
        "utilisation_engrais": int(utilisation_engrais),
        "acces_irrigation": int(acces_irrigation),
        "nb_menages": nb_menages,
    }


def predict(model, raw_input: dict, feature_columns: list[str], scaler, numerical_cols):
    """Transform one user input and return prediction label plus confidence."""
    model_input = build_inference_input(
        raw_input, feature_columns, scaler, numerical_cols
    )
    predicted_class = int(model.predict(model_input)[0])
    probabilities = model.predict_proba(model_input)[0]
    confidence = float(probabilities[predicted_class])
    label = "Bonne Récolte" if predicted_class == 1 else "Mauvaise Récolte"
    return label, confidence, model_input


def get_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    """Return feature importance values for tree models or logistic regression."""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_[0])
    else:
        raise ValueError("Le modèle sélectionné ne fournit pas d'importance.")

    return (
        pd.DataFrame({"Variable": feature_columns, "Importance": importance})
        .sort_values("Importance", ascending=False)
        .head(15)
        .sort_values("Importance", ascending=True)
    )


def plot_feature_importance(model, feature_columns: list[str], model_name: str):
    """Create a horizontal feature-importance chart for the selected model."""
    importance_df = get_feature_importance(model, feature_columns)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    fig.patch.set_facecolor("#f6f8f4")
    ax.set_facecolor("#f6f8f4")
    ax.barh(importance_df["Variable"], importance_df["Importance"], color="#2f7d46")
    ax.set_title(f"Importance des variables — {model_name}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.grid(axis="x", color="#d8e2d4", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9d6c5")
    ax.spines["bottom"].set_color("#c9d6c5")
    ax.tick_params(axis="both", labelsize=10, colors="#1f2937")
    fig.tight_layout()
    return fig


def render_input_summary(display_input: dict) -> None:
    """Render a compact key-value summary of the current prediction input."""
    rows = [
        ("Année", display_input["annee"]),
        ("Saison", display_input["saison"]),
        ("Province", display_input["province"]),
        ("Culture", display_input["culture"]),
        ("Altitude", f"{display_input['altitude_m']} m"),
        ("Pluviométrie", f"{display_input['pluviometrie_mm']} mm"),
        ("Température", f"{display_input['temperature_moy_C']:.1f} °C"),
        ("Superficie", f"{display_input['superficie_ha']:.2f} ha"),
        ("Ménages", display_input["nb_menages"]),
        ("Engrais", display_input["utilisation_engrais"]),
        ("Irrigation", display_input["acces_irrigation"]),
    ]
    summary_df = pd.DataFrame(rows, columns=["Champ", "Valeur"])
    summary_df["Valeur"] = summary_df["Valeur"].astype(str)
    st.dataframe(summary_df, hide_index=True, width="stretch")


models, scaler, feature_columns, numerical_cols = load_artifacts()
metrics = load_metrics()


with st.sidebar:
    st.header("Paramètres de la parcelle")

    province = st.selectbox("Province", PROVINCES, index=PROVINCES.index("Gitega"))
    culture = st.selectbox("Culture", CULTURES, index=CULTURES.index("Haricot"))
    altitude = st.slider("Altitude (m)", 500, 2800, 1720)
    pluviometrie = st.slider("Pluviométrie (mm)", 200, 1400, 430)
    temperature = st.slider("Température moyenne (°C)", 14.0, 28.0, 18.2, step=0.1)
    superficie = st.number_input("Superficie (ha)", 0.5, 500.0, 1.5, step=0.5)
    nb_menages = st.number_input("Nombre de ménages", 10, 5000, 120, step=10)
    annee = st.selectbox("Année", list(range(2015, 2031)), index=8)
    saison = st.selectbox("Saison", ["A", "B"])
    utilisation_engrais = st.checkbox("Utilisation d'engrais")
    acces_irrigation = st.checkbox("Accès à l'irrigation")

    st.divider()
    selected_model_name = st.selectbox("Modèle", list(MODEL_CONFIG.keys()))
    predict_clicked = st.button("Prédire", type="primary", width="stretch")


st.markdown(
    """
    <div class="app-header">
        <h1>Prédiction des Récoltes — Burundi</h1>
        <p class="app-subtitle">
            Estimation de la qualité d'une récolte à partir de données agricoles,
            climatiques et géographiques.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


selected_model = models[selected_model_name]
selected_metrics = metrics[MODEL_CONFIG[selected_model_name]["key"]]

raw_input = build_raw_input(
    province,
    culture,
    altitude,
    pluviometrie,
    temperature,
    superficie,
    nb_menages,
    annee,
    saison,
    utilisation_engrais,
    acces_irrigation,
)


if predict_clicked:
    prediction_label, confidence, _ = predict(
        selected_model, raw_input, feature_columns, scaler, numerical_cols
    )
else:
    prediction_label, confidence, _ = predict(
        selected_model, raw_input, feature_columns, scaler, numerical_cols
    )


result_class = "result-good" if prediction_label == "Bonne Récolte" else "result-bad"

st.markdown(
    f"""
    <div class="result-banner {result_class}">
        {prediction_label}
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Probabilité associée à la classe prédite")
st.progress(confidence)
st.write(f"**{confidence * 100:.1f} %**")

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Accuracy", f"{selected_metrics['accuracy']:.3f}")
metric_col_2.metric("F1-score", f"{selected_metrics['f1']:.3f}")
metric_col_3.metric("AUC", f"{selected_metrics['auc']:.3f}")

chart_col, input_col = st.columns([1.45, 1])
with chart_col:
    st.subheader("Variables les plus influentes")
    st.pyplot(plot_feature_importance(selected_model, feature_columns, selected_model_name))

with input_col:
    st.subheader("Entrée utilisée")
    display_input = raw_input.copy()
    display_input["culture"] = culture
    display_input["utilisation_engrais"] = "Oui" if utilisation_engrais else "Non"
    display_input["acces_irrigation"] = "Oui" if acces_irrigation else "Non"
    render_input_summary(display_input)
