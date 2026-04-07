import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="☕ Random Forest – Coffee Quality",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌲 Coffee Quality Predictor — Random Forest")
st.markdown("*Modelo: RandomForestRegressor Optimizado | Proyecto Henry Data Science*")

# ──────────────────────────────────────────────
# Carga de modelo y preprocessor
# ──────────────────────────────────────────────
@st.cache_resource
def load_assets():
    try:
        model        = joblib.load("models/prediction/all_models/RandomForest_Optimized.pkl")
        preprocessor = joblib.load("models/prediction/preprocessor.pkl")
        metadata     = joblib.load("models/prediction/training_metadata.pkl")
        return model, preprocessor, metadata
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None

model, preprocessor, metadata = load_assets()

# ──────────────────────────────────────────────
# Sidebar – navegación
# ──────────────────────────────────────────────
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Sección:",
    ["🎯 Predicción", "📊 Métricas del Modelo", "🔍 Importancia de Features"]
)

# ──────────────────────────────────────────────
# Función de predicción
# ──────────────────────────────────────────────
def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    X  = preprocessor.transform(df)
    pred = model.predict(X)[0]

    # Estimación de intervalo usando std de los árboles
    tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
    std = tree_preds.std()

    category, color = (
        ("Excelente 🏆", "#2ecc71")  if pred >= 85 else
        ("Muy Bueno ✅", "#3498db")  if pred >= 80 else
        ("Bueno ☕",     "#f39c12")  if pred >= 75 else
        ("Regular ⚠️",  "#e74c3c")
    )

    return {
        "score":    round(float(pred), 2),
        "lower":    round(max(0,   float(pred - 1.96 * std)), 2),
        "upper":    round(min(100, float(pred + 1.96 * std)), 2),
        "std":      round(float(std), 3),
        "category": category,
        "color":    color,
    }

def build_input(aroma, flavor, aftertaste, acidity, body, balance,
                species, country, altitude, moisture, defects1, defects2):
    sensory = [aroma, flavor, aftertaste, acidity, body, balance]
    alt_cat = (
        "Baja"      if altitude < 1000 else
        "Media-Baja" if altitude < 1400 else
        "Media"      if altitude < 1600 else
        "Media-Alta" if altitude < 2000 else
        "Alta"
    )
    moist_cat = (
        "Baja"     if moisture < 0.08 else
        "Óptima"   if moisture <= 0.12 else
        "Aceptable" if moisture <= 0.15 else
        "Alta"
    )
    return {
        "Species":                  species,
        "Aroma":                    aroma,
        "Flavor":                   flavor,
        "Aftertaste":               aftertaste,
        "Acidity":                  acidity,
        "Body":                     body,
        "Balance":                  balance,
        "Country.of.Origin":        country,
        "altitude_mean_meters":     altitude,
        "Moisture":                 moisture,
        "Category.One.Defects":     defects1,
        "Category.Two.Defects":     defects2,
        "Number.of.Bags":           300,
        "Cupper.Points":            8.0,
        "Variety":                  "Other",
        "Processing.Method":        "Washed / Wet",
        "Color":                    "Green",
        "Owner":                    "Desconocido",
        "Region":                   "Other",
        "altitude_category":        alt_cat,
        "altitude_std":             0.0,
        "sensory_avg":              np.mean(sensory),
        "sensory_std":              float(np.std(sensory)),
        "best_sensory":             max(sensory),
        "total_defects":            defects1 + defects2,
        "no_defects":               int(defects1 == 0 and defects2 == 0),
        "moisture_category":        moist_cat,
        "processing_simple":        "Washed",
    }

# ══════════════════════════════════════════════
# PÁGINA 1 — Predicción
# ══════════════════════════════════════════════
if page == "🎯 Predicción":
    st.header("🎯 Predecí la calidad de tu café")

    if model is None:
        st.error("No se pudo cargar el modelo.")
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Características sensoriales")
        aroma      = st.slider("Aroma",      0.0, 10.0, 8.0, 0.1)
        flavor     = st.slider("Sabor",      0.0, 10.0, 8.2, 0.1)
        aftertaste = st.slider("Posgusto",   0.0, 10.0, 8.1, 0.1)
        acidity    = st.slider("Acidez",     0.0, 10.0, 8.0, 0.1)
        body       = st.slider("Cuerpo",     0.0, 10.0, 8.3, 0.1)
        balance    = st.slider("Balance",    0.0, 10.0, 8.2, 0.1)

    with col2:
        st.subheader("Origen y condiciones")
        species   = st.selectbox("Especie", ["Arabica", "Robusta"])
        country   = st.selectbox("País de origen", [
            "Colombia", "Ethiopia", "Kenya", "Brazil", "Vietnam",
            "Guatemala", "Costa Rica", "Peru", "Honduras", "Mexico",
            "Tanzania, United Republic Of", "Uganda", "Indonesia"
        ])
        altitude  = st.number_input("Altitud (metros)", 0, 3000, 1500)
        moisture  = st.number_input("Humedad", 0.0, 0.5, 0.12, 0.01, format="%.2f")
        defects1  = st.number_input("Defectos Categoría 1", 0, 10, 0)
        defects2  = st.number_input("Defectos Categoría 2", 0, 20, 1)

    if st.button("🔮 Predecir calidad", type="primary", use_container_width=True):
        input_data = build_input(
            aroma, flavor, aftertaste, acidity, body, balance,
            species, country, altitude, moisture, defects1, defects2
        )
        res = predict(input_data)

        st.success("¡Predicción completada!")
        st.markdown("---")

        m1, m2, m3 = st.columns(3)
        m1.metric("Puntuación estimada", f"{res['score']} / 100")
        m2.metric("Intervalo 95%", f"{res['lower']} – {res['upper']}")
        m3.metric("Categoría", res["category"])

        # Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=res["score"],
            title={"text": "Calidad del Café"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": res["color"]},
                "steps": [
                    {"range": [0,  75], "color": "#f8f9fa"},
                    {"range": [75, 80], "color": "#fff3cd"},
                    {"range": [80, 85], "color": "#cce5ff"},
                    {"range": [85,100], "color": "#d4edda"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 3},
                    "thickness": 0.75,
                    "value": 90
                }
            }
        ))
        fig_gauge.update_layout(height=350)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Radar sensorial
        cats   = ["Aroma", "Sabor", "Posgusto", "Acidez", "Cuerpo", "Balance"]
        vals   = [aroma, flavor, aftertaste, acidity, body, balance]
        fig_r  = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=cats + [cats[0]],
            fill="toself",
            line_color=res["color"],
            name="Perfil sensorial"
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            title="Perfil Sensorial del Café",
            height=400
        )
        st.plotly_chart(fig_r, use_container_width=True)

        # Interpretación
        st.subheader("📋 Interpretación")
        if res["score"] >= 85:
            st.success("🏆 **Café de Especialidad** — Apto para mercado premium.")
        elif res["score"] >= 80:
            st.info("✅ **Alta Calidad** — Muy bueno para el mercado especial.")
        elif res["score"] >= 75:
            st.warning("☕ **Buena Calidad** — Adecuado para consumo general.")
        else:
            st.error("⚠️ **Calidad Regular** — Requiere mejoras.")

# ══════════════════════════════════════════════
# PÁGINA 2 — Métricas del Modelo
# ══════════════════════════════════════════════
elif page == "📊 Métricas del Modelo":
    st.header("📊 Métricas y Validación — Random Forest Optimizado")

    if model is None:
        st.error("No se pudo cargar el modelo.")
        st.stop()

    # ── Info del modelo ──
    st.subheader("🌲 ¿Por qué Random Forest?")
    st.markdown("""
    **Random Forest** es un modelo de *ensemble* que entrena múltiples árboles de decisión
    y promedia sus predicciones. Sus ventajas para este problema son:

    - **Robusto a outliers**: el café tiene productores muy distintos; RF maneja bien esa varianza.
    - **No lineal**: la calidad del café no depende linealmente de sus características.
    - **Interpretable**: a través de la importancia de features podemos explicar qué factores importan más.
    - **Optimizado**: esta versión fue tuneada con hiperparámetros para mejorar performance.
    """)

    # ── Métricas ──
    st.subheader("📈 Métricas de Evaluación")

    # Cargamos analysis_summary para mostrar comparación
    try:
        summary_df = pd.read_csv("models/prediction/analysis_summary.csv")
        rf_row = summary_df[summary_df.apply(
            lambda r: "Random" in str(r.values), axis=1
        )]
    except Exception:
        rf_row = pd.DataFrame()

    # Métricas del RF desde el modelo mismo
    n_estimators = model.n_estimators
    max_depth    = model.max_depth

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modelo", "RandomForest")
    c2.metric("Árboles", n_estimators)
    c3.metric("Max Depth", str(max_depth) if max_depth else "Sin límite")
    c4.metric("Features usadas", model.n_features_in_)

    # ── Comparación con otros modelos ──
    st.subheader("🏆 Comparación con otros modelos")

    try:
        comparison_df = pd.read_csv("models/prediction/training_results.csv")
        st.dataframe(comparison_df.round(4), use_container_width=True)

        # Gráfico de barras comparativo
        if "RMSE" in comparison_df.columns and "Model" in comparison_df.columns:
            fig_comp = px.bar(
                comparison_df.sort_values("RMSE"),
                x="Model", y="RMSE",
                color="RMSE",
                color_continuous_scale="RdYlGn_r",
                title="RMSE por modelo (menor es mejor)",
            )
            fig_comp.update_layout(height=400)
            st.plotly_chart(fig_comp, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo cargar la comparación de modelos: {e}")

    # ── Plan de validación ──
    st.subheader("📋 Plan de Validación")
    st.markdown("""
    | Paso | Detalle |
    |------|---------|
    | **División de datos** | Train 80% / Test 20%, con `random_state` fijo para reproducibilidad |
    | **Métricas principales** | RMSE (error cuadrático medio) y R² (varianza explicada) |
    | **Cross-validation** | K-Fold (k=5) para estimación más robusta del error |
    | **Optimización** | GridSearch / RandomSearch sobre hiperparámetros clave |
    | **Justificación del modelo** | Seleccionado por menor RMSE en conjunto de validación |
    | **Reproducibilidad** | Pipeline guardado con `joblib`; preprocessor y modelo versionados |
    """)

# ══════════════════════════════════════════════
# PÁGINA 3 — Importancia de Features
# ══════════════════════════════════════════════
elif page == "🔍 Importancia de Features":
    st.header("🔍 ¿Qué factores definen la calidad del café?")

    if model is None:
        st.error("No se pudo cargar el modelo.")
        st.stop()

    # Feature importances del RF
    importances = model.feature_importances_

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    fi_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # Limpiar prefijos num__ / cat__
    fi_df["feature_clean"] = (
        fi_df["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )

    # Top 15
    top15 = fi_df.head(15)

    fig_fi = px.bar(
        top15,
        x="importance",
        y="feature_clean",
        orientation="h",
        color="importance",
        color_continuous_scale="Greens",
        title="Top 15 características más importantes para predecir calidad",
        labels={"importance": "Importancia", "feature_clean": "Característica"}
    )
    fig_fi.update_layout(height=500, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_fi, use_container_width=True)

    # Tabla completa colapsada
    with st.expander("Ver tabla completa de importancias"):
        st.dataframe(
            fi_df[["feature_clean", "importance"]].rename(
                columns={"feature_clean": "Feature", "importance": "Importancia"}
            ).round(5),
            use_container_width=True
        )

    # Insight automático
    top3 = top15["feature_clean"].tolist()[:3]
    st.subheader("💡 Insight")
    st.info(
        f"Las 3 características que **más influyen** en la predicción son: "
        f"**{top3[0]}**, **{top3[1]}** y **{top3[2]}**. "
        f"Esto confirma que las características sensoriales del café son el principal "
        f"determinante de su calidad final."
    )

# ── Footer ──
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8em;'>"
    "🌲 Random Forest Optimizado | Feli — Henry Data Science Bootcamp"
    "</div>",
    unsafe_allow_html=True
)