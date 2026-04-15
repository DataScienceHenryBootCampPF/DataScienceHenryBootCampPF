import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import sys

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predictor.recommendation_system import HybridCoffeeRecommendationSystem

# Configuración de la página
st.set_page_config(
    page_title="☕ Coffee Quality Predictor - Best Model",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("☕ Coffee Quality Predictor & Recommendation System - Best Model")
st.markdown("""
*Aplicación interactiva para predecir la calidad del café y encontrar recomendaciones personalizadas*
""")

# Importar demos como módulos
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from demo_prediction import configurar_rutas, mostrar_opciones_lista
from demo_recommender import ejecutar_demo as demo_recommender

# Función modificada para demo de predicción compatible con Streamlit
def demo_prediction_streamlit():
    """Versión modificada del demo de predicción para Streamlit"""
    prefix = configurar_rutas()
    path_data = os.path.join(prefix, "data", "processed", "coffee_data_for_training.csv")
    path_metrics = os.path.join(prefix, "metrics", "model_comparison_ranking.csv")
    path_models = os.path.join(prefix, "models", "prediction")

    if not os.path.exists(path_data) or not os.path.exists(path_metrics):
        return "❌ ERROR: Archivos no encontrados."

    try:
        df_train = pd.read_csv(path_data)
        df_results = pd.read_csv(path_metrics)
        
        best_model_name = df_results.iloc[0]['Model']
        mae_modelo = df_results.iloc[0]['MAE'] 
        best_model = joblib.load(os.path.join(path_models, f"coffee_model_{best_model_name}.pkl"))

        output = []
        output.append("\n" + "="*60)
        output.append(f"☕ SIMULADOR DE CALIDAD - COFFEE PREDICTOR PRO")
        output.append(f"🤖 Motor: {best_model_name} | Error Medio: ±{mae_modelo:.2f}")
        output.append("="*60)

        # Usar datos de ejemplo en lugar de input del usuario
        ejemplo_pais = "Colombia"
        ejemplo_region = "Huila"
        ejemplo_variedad = "Caturra"
        ejemplo_proceso = "Washed"
        ejemplo_color = "Green"
        ejemplo_altitud = 1500

        output.append(f"\n--- Ejemplo con País de Origen: {ejemplo_pais} ---")
        output.append(f"Región: {ejemplo_region}")
        output.append(f"Variedad: {ejemplo_variedad}")
        output.append(f"Proceso: {ejemplo_proceso}")
        output.append(f"Color: {ejemplo_color}")
        output.append(f"Altitud: {ejemplo_altitud} msnm")

        data_final = pd.DataFrame({
            'Country.of.Origin': [ejemplo_pais],
            'Region': [ejemplo_region],
            'Variety': [ejemplo_variedad],
            'Processing.Method': [ejemplo_proceso],
            'Color': [ejemplo_color],
            'categoria_altitud': ['Alta' if ejemplo_altitud > 1200 else 'Media' if ejemplo_altitud > 800 else 'Baja'],
            'Moisture': [0.11],
            'Category.One.Defects': [0],
            'Category.Two.Defects': [0],
            'altitude_mean_meters': [ejemplo_altitud]
        })

        pred = best_model.predict(data_final)[0]
        
        umbral_mediana = 82.5
        limite_inf = pred - mae_modelo
        limite_sup = pred + mae_modelo

        output.append("\n" + "*"*40)
        output.append(f"📊 RESULTADO DE LA PREDICCIÓN")
        output.append(f"Puntaje Estimado: {pred:.2f}")
        output.append(f"Rango de Confianza (95%): {limite_inf:.2f} a {limite_sup:.2f}")
        output.append("-" * 40)
        
        if pred >= umbral_mediana:
            output.append(f"🏆 CATEGORÍA: PREMIUM")
            output.append(f"   (Puntaje por encima de la mediana del mercado)")
        else:
            output.append(f"📦 CATEGORÍA: ESTÁNDAR")
            output.append(f"   (Puntaje dentro del rango base de comercialización)")
        
        output.append("*"*40 + "\n")
        
        return "\n".join(output)

    except Exception as e:
        return f"❌ Error en la simulación: {e}"

# Función modificada para demo de recomendación compatible con Streamlit
def demo_recommender_streamlit():
    """Versión modificada del demo de recomendación para Streamlit"""
    try:
        csv_path = './models/recommender/coffee_flavor_segments.csv'
        if not os.path.exists(csv_path):
            return "❌ Error: No existe el archivo de segmentos. Ejecuta el Profiler primero."

        df = pd.read_csv(csv_path)
        perfiles = sorted(list(df['Perfil_Nombre'].unique()))
        
        output = []
        output.append("\n☕ BIENVENIDO AL BUSCADOR DE CAFÉS PREMIUM ☕")
        output.append("-" * 55)
        
        # Usar ejemplo en lugar de input del usuario
        perfil_ejemplo = "Frutal y Floral" if "Frutal y Floral" in perfiles else perfiles[0]
        output.append(f"Paso 1: Perfil seleccionado automáticamente: {perfil_ejemplo}")
        
        # Filtramos países para ese perfil
        df_perfil = df[df['Perfil_Nombre'] == perfil_ejemplo]
        paises = sorted(df_perfil['Country.of.Origin'].unique())
        
        pais_ejemplo = paises[0] if paises else "Colombia"
        output.append(f"Paso 2: País seleccionado: {pais_ejemplo}")
        
        # Lógica de Filtrado
        resultados = df_perfil.copy()
        resultados = resultados[resultados['Country.of.Origin'] == pais_ejemplo]
        output.append(f"\n🔍 Buscando los mejores de {pais_ejemplo}...")
        
        if resultados.empty:
            output.append("❌ No se encontraron resultados.")
        else:
            final_top = (resultados
                         .sort_values(by='Total.Cup.Points', ascending=False)
                         .drop_duplicates(subset=['Nombre_Comercial'])
                         [['Nombre_Comercial', 'Total.Cup.Points', 'Categoria_Calidad']]
                         .head(5))

            output.append(f"\n🏆 TOP 5 CAFÉS RECOMENDADOS:")
            
            for i, (idx, row) in enumerate(final_top.iterrows(), 1):
                output.append(f"  {i}. {row['Nombre_Comercial']} - {row['Total.Cup.Points']:.2f} - {row['Categoria_Calidad']}")
            
            output.append(f"\n💡 Perfil sensorial seleccionado: {perfil_ejemplo}")
        
        return "\n".join(output)

    except Exception as e:
        return f"❌ Ocurrió un error: {e}"

# Cargar modelos y datos
@st.cache_resource
def load_models():
    """Carga el modelo predictivo y componentes relacionados"""
    try:
        # Usar las mismas rutas que demo_prediction.py
        prefix = "."
        path_models = os.path.join(prefix, "models", "prediction")
        path_metrics = os.path.join(prefix, "metrics", "model_comparison_ranking.csv")
        
        # Cargar métricas para obtener el mejor modelo
        if os.path.exists(path_metrics):
            df_results = pd.read_csv(path_metrics)
            best_model_name = df_results.iloc[0]['Model']
            model_path = os.path.join(path_models, f"coffee_model_{best_model_name}.pkl")
        else:
            # Fallback a best_model.pkl
            model_path = "models/prediction/best_model.pkl"
            
        preprocessor_path = "models/prediction/preprocessor.pkl"
        metadata_path = "models/prediction/training_metadata.pkl"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None
        
        return model, preprocessor, metadata
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None, None, None

@st.cache_resource
def load_recommender():
    """Carga el sistema de recomendación"""
    try:
        return HybridCoffeeRecommendationSystem()
    except Exception as e:
        st.error(f"Error al cargar el sistema de recomendación: {e}")
        return None

def get_enhanced_recommendations(recommender, flavor, aftertaste, aroma, acidity, body, balance, 
                                 species=None, country_filter=None, top_n=10):
    """
    Sistema mejorado de recomendación que prioriza países seleccionados y muestra similares.
    
    Args:
        recommender: Sistema de recomendación
        flavor, aftertaste, aroma, acidity, body, balance: Características sensoriales
        species: Filtro de especie opcional
        country_filter: País seleccionado para priorizar
        top_n: Número de recomendaciones
        
    Returns:
        pd.DataFrame: Recomendaciones priorizadas
    """
    try:
        # 1. Obtener recomendaciones base del sistema original
        base_recommendations = recommender.recomendar(
            Flavor=flavor,
            Aftertaste=aftertaste,
            Aroma=aroma,
            Acidity=acidity,
            Body=body,
            Balance=balance,
            species=species,
            top_n=50  # Obtener más resultados para poder filtrar y priorizar
        )
        
        if base_recommendations is None or base_recommendations.empty:
            return pd.DataFrame()
        
        # 2. Si hay un país seleccionado, priorizarlo
        if country_filter and country_filter != "Todos":
            # Separar recomendaciones del país seleccionado vs otros países
            selected_country_recs = base_recommendations[
                base_recommendations['Country.of.Origin'] == country_filter
            ]
            other_countries_recs = base_recommendations[
                base_recommendations['Country.of.Origin'] != country_filter
            ]
            
            # 3. Encontrar países similares estadísticamente
            similar_countries = find_similar_countries(base_recommendations, country_filter)
            similar_countries_recs = other_countries_recs[
                other_countries_recs['Country.of.Origin'].isin(similar_countries)
            ]
            
            # 4. Combinar resultados con priorización
            # Primero: país seleccionado
            # Segundo: países similares
            # Tercero: resto de países
            
            final_recs = pd.concat([
                selected_country_recs.head(top_n // 2),  # 50% del país seleccionado
                similar_countries_recs.head(top_n // 3),  # ~33% de países similares
                other_countries_recs.head(top_n - len(selected_country_recs.head(top_n // 2)) - len(similar_countries_recs.head(top_n // 3)))
            ], ignore_index=True)
            
            # 5. Añadir información de priorización
            final_recs['priority'] = final_recs['Country.of.Origin'].apply(
                lambda x: 'Alta' if x == country_filter 
                else 'Media' if x in similar_countries 
                else 'Baja'
            )
            
        else:
            # Sin filtro de país, usar recomendaciones normales
            final_recs = base_recommendations.head(top_n)
            final_recs['priority'] = 'Media'
        
        # 6. Ordenar por similitud y prioridad
        priority_order = {'Alta': 1, 'Media': 2, 'Baja': 3}
        final_recs['priority_order'] = final_recs['priority'].map(priority_order)
        final_recs = final_recs.sort_values(['priority_order', 'similarity_score'], 
                                          ascending=[True, False]).head(top_n)
        
        return final_recs.drop('priority_order', axis=1)
        
    except Exception as e:
        st.error(f"Error en el sistema mejorado de recomendación: {e}")
        # Fallback al sistema original
        return recommender.recomendar(
            Flavor=flavor, Aftertaste=aftertaste, Aroma=aroma,
            Acidity=acidity, Body=body, Balance=balance,
            species=species, top_n=top_n
        ) or pd.DataFrame()

def find_similar_countries(recommendations_df, target_country, top_similar=5):
    """
    Encuentra países con características estadísticas similares.
    
    Args:
        recommendations_df: DataFrame con recomendaciones
        target_country: País de referencia
        top_similar: Número de países similares a encontrar
        
    Returns:
        list: Lista de países similares
    """
    try:
        # Calcular estadísticas por país
        country_stats = recommendations_df.groupby('Country.of.Origin').agg({
            'Total.Cup.Points': ['mean', 'std'],
            'Flavor': 'mean',
            'Aroma': 'mean',
            'Acidity': 'mean',
            'Body': 'mean',
            'Balance': 'mean'
        }).round(2)
        
        # Aplanar nombres de columnas
        country_stats.columns = ['_'.join(col).strip() for col in country_stats.columns]
        
        # Obtener estadísticas del país objetivo
        if target_country not in country_stats.index:
            return []
        
        target_stats = country_stats.loc[target_country]
        
        # Calcular similitud con otros países
        similarities = []
        for country in country_stats.index:
            if country != target_country:
                country_data = country_stats.loc[country]
                # Similitud euclidiana simple
                distance = np.sqrt(sum((target_stats - country_data) ** 2))
                similarities.append((country, distance))
        
        # Ordenar por similitud (menor distancia = más similar)
        similarities.sort(key=lambda x: x[1])
        
        # Retornar los países más similares
        return [country for country, _ in similarities[:top_similar]]
        
    except Exception as e:
        st.warning(f"No se pudieron encontrar países similares: {e}")
        return []

# Cargar modelos
model, preprocessor, metadata = load_models()
recommender = load_recommender()

# Sidebar para navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Selecciona una funcionalidad:",
    ["🎯 Predicción de Calidad", "🔍 Sistema de Recomendación", "📊 Análisis de Modelos", "🎮 Demo Interactivo", "ℹ️ Información"]
)

# Función para predecir calidad
def predict_quality(input_data):
    """Realiza la predicción de calidad del café"""
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocesar datos si el preprocesador está disponible
        if preprocessor is not None:
            processed_data = preprocessor.transform(df)
        else:
            # Si no hay preprocesador, usar los datos directamente (como en los demos)
            processed_data = df
        
        # Hacer predicción
        prediction = model.predict(processed_data)[0]
        
        # Calcular intervalo de confianza
        if metadata and 'best_rmse' in metadata:
            rmse = metadata['best_rmse']
        else:
            rmse = 2.0  # Valor por defecto si no hay metadata
        
        lower_bound = max(0, prediction - 1.96 * rmse)
        upper_bound = min(100, prediction + 1.96 * rmse)
        
        # Determinar categoría de calidad (usando lógica de demo_prediction.py)
        umbral_mediana = 82.5
        if prediction >= umbral_mediana:
            quality_category = "Café Premium"
        else:
            quality_category = "Café Estándar"
        
        results = {
            'predicted_score': round(prediction, 2),
            'quality_category': quality_category,
            'confidence_interval': {
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2)
            },
            'model_rmse': rmse,
            'accuracy_estimate': f"{(1 - rmse/100)*100:.1f}%"
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
        return None


# Página 1: Predicción de Calidad
if page == "🎯 Predicción de Calidad":
    st.header("🎯 Predicción de Calidad de Café - Best Model")
    
    if model is None:
        st.error("❌ No se pudo cargar el modelo predictivo. Por favor, verifica que los archivos del modelo existan.")
    else:
        # Información del modelo
        st.info(f"""
        **Modelo Actual:** {metadata['best_model_name']}  
        **RMSE:** {metadata['best_rmse']:.3f}  
        **R²:** {metadata['best_r2']:.3f}  
        **Precisión Estimada:** {(1 - metadata['best_rmse']/100)*100:.1f}%
        """)
        
        # Tabs para diferentes modos de entrada
        tab1, tab2, tab3 = st.tabs(["Entrada Manual", "Ejemplos Predefinidos", "Entrada Avanzada"])
        
        with tab1:
            st.subheader("📝 Entrada Manual de Datos")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Características Sensoriales**")
                aroma = st.slider("Aroma", 0.0, 10.0, 8.0, 0.1)
                flavor = st.slider("Sabor", 0.0, 10.0, 8.2, 0.1)
                aftertaste = st.slider("Posgusto", 0.0, 10.0, 8.1, 0.1)
                acidity = st.slider("Acidez", 0.0, 10.0, 8.0, 0.1)
                body = st.slider("Cuerpo", 0.0, 10.0, 8.3, 0.1)
                balance = st.slider("Balance", 0.0, 10.0, 8.2, 0.1)
                
            with col2:
                st.write("**Características de Origen**")
                species = st.selectbox("Especie", ["Arabica", "Robusta"])
                country = st.selectbox("País de Origen", [
    "Brazil", "China", "Colombia", "Costa Rica", "El Salvador", "Ethiopia", 
    "Guatemala", "Honduras", "Indonesia", "Kenya", "Malawi", "Mexico", 
    "Nicaragua", "Other", "Peru", "Taiwan", "Tanzania", "Thailand", 
    "Uganda", "United States (Hawaii)"
])
                altitude = st.number_input("Altitud (metros)", 0, 3000, 1500)
                moisture = st.number_input("Humedad (%)", 0.0, 0.5, 0.12, 0.01)
                
                st.write("**Defectos**")
                defects1 = st.number_input("Defectos Categoría 1", 0, 10, 0)
                defects2 = st.number_input("Defectos Categoría 2", 0, 20, 1)
            
            # Botón de predicción
            if st.button("🔮 Predecir Calidad", type="primary"):
                # Construir datos de entrada
                input_data = {
                    'Species': species,
                    'Aroma': aroma,
                    'Flavor': flavor,
                    'Aftertaste': aftertaste,
                    'Acidity': acidity,
                    'Body': body,
                    'Balance': balance,
                    'Country.of.Origin': country,
                    'altitude_mean_meters': altitude,
                    'Moisture': moisture,
                    'Category.One.Defects': defects1,
                    'Category.Two.Defects': defects2,
                    # Valores por defecto para campos requeridos
                    'Number.of.Bags': 300,
                    'Cupper.Points': 8.0,
                    'Variety': 'Unknown',
                    'Processing.Method': 'Washed / Wet',
                    'Color': 'Green',
                    'Owner': 'Unknown Farm',
                    'Region': 'Unknown',
                    'altitude_category': 'Media' if altitude < 1600 else 'Alta' if altitude < 2000 else 'Muy-Alta',
                    'altitude_std': 0.0,
                    'sensory_avg': (aroma + flavor + aftertaste + acidity + body + balance) / 6,
                    'sensory_std': 0.1,
                    'best_sensory': max(aroma, flavor, aftertaste, acidity, body, balance),
                    'total_defects': defects1 + defects2,
                    'no_defects': 1 if defects1 == 0 and defects2 == 0 else 0,
                    'moisture_category': 'Óptima' if 0.1 <= moisture <= 0.12 else 'Aceptable',
                    'processing_simple': 'Washed'
                }
                
                # Realizar predicción
                results = predict_quality(input_data)
                
                if results:
                    # Mostrar resultados usando formato de demo_prediction.py
                    pred = results['predicted_score']
                    mae_modelo = results['model_rmse']
                    umbral_mediana = 82.5
                    limite_inf = pred - mae_modelo
                    limite_sup = pred + mae_modelo
                    
                    st.success("Predicción realizada con éxito!")
                    
                    # Formato similar a demo_prediction.py
                    st.markdown("#### " + "*"*40)
                    st.markdown(f"**RESULTADO DE LA PREDICCIÓN**")
                    st.markdown(f"**Puntaje Estimado:** {pred:.2f}")
                    st.markdown(f"**Rango de Confianza (95%):** {limite_inf:.2f} a {limite_sup:.2f}")
                    st.markdown("---")
                    
                    if pred >= umbral_mediana:
                        st.success(f"**CATEGORÍA: PREMIUM**")
                        st.info("(Puntaje por encima de la mediana del mercado)")
                    else:
                        st.warning(f"**CATEGORÍA: ESTÁNDAR**")
                        st.info("(Puntaje dentro del rango base de comercialización)")
                    
                    st.markdown("*"*40)
                    
                    # Métricas adicionales
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Puntuación Predicha",
                            value=f"{pred}/100",
                            delta=results['quality_category']
                        )
                    
                    with col2:
                        st.metric(
                            label="Intervalo de Confianza (95%)",
                            value=f"{limite_inf:.2f} - {limite_sup:.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Precisión Estimada",
                            value=results['accuracy_estimate']
                        )
                    
                    # Visualización de resultados
                    fig = go.Figure()
                    
                    # Barra de progreso para la puntuación
                    fig.add_trace(go.Indicator(
                        mode = "gauge+number+delta",
                        value = results['predicted_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Calidad del Café"},
                        delta = {'reference': 80},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 75], 'color': "lightgray"},
                                {'range': [75, 80], 'color': "yellow"},
                                {'range': [80, 85], 'color': "lightblue"},
                                {'range': [85, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Interpretación
                    st.subheader("📋 Interpretación de Resultados")
                    if results['predicted_score'] >= 85:
                        st.success("🏆 **Café de Especialidad** - Excelente para el mercado premium")
                    elif results['predicted_score'] >= 80:
                        st.info("✅ **Café de Alta Calidad** - Muy bueno para el mercado especial")
                    elif results['predicted_score'] >= 75:
                        st.warning("☕ **Café de Buena Calidad** - Adecuado para consumo general")
                    else:
                        st.error("⚠️ **Café de Calidad Regular** - Podría necesitar mejoras")
        
        with tab2:
            st.subheader("🎲 Ejemplos Predefinidos")
            
            # Ejemplos de café
            sample_coffees = [
                {
                    "name": "Café Especial Etiopía",
                    "description": "Café de altura, con notas florales y afrutadas",
                    "data": {
                        'Species': 'Arabica',
                        'Aroma': 8.5,
                        'Flavor': 8.7,
                        'Aftertaste': 8.6,
                        'Acidity': 8.8,
                        'Body': 8.4,
                        'Balance': 8.5,
                        'Country.of.Origin': 'Ethiopia',
                        'altitude_mean_meters': 1800,
                        'Moisture': 0.11,
                        'Category.One.Defects': 0,
                        'Category.Two.Defects': 2,
                    }
                },
                {
                    "name": "Café Colombiano Premium",
                    "description": "Café balanceado con notas de chocolate y nuez",
                    "data": {
                        'Species': 'Arabica',
                        'Aroma': 8.2,
                        'Flavor': 8.4,
                        'Aftertaste': 8.3,
                        'Acidity': 8.1,
                        'Body': 8.5,
                        'Balance': 8.3,
                        'Country.of.Origin': 'Colombia',
                        'altitude_mean_meters': 1600,
                        'Moisture': 0.12,
                        'Category.One.Defects': 0,
                        'Category.Two.Defects': 1,
                    }
                },
                {
                    "name": "Café Kenia AA",
                    "description": "Café de altura con acidez brillante y notas cítricas",
                    "data": {
                        'Species': 'Arabica',
                        'Aroma': 8.8,
                        'Flavor': 8.9,
                        'Aftertaste': 8.7,
                        'Acidity': 9.0,
                        'Body': 8.3,
                        'Balance': 8.6,
                        'Country.of.Origin': 'Kenya',
                        'altitude_mean_meters': 2000,
                        'Moisture': 0.10,
                        'Category.One.Defects': 0,
                        'Category.Two.Defects': 0,
                    }
                },
                {
                    "name": "Café Robusta Comercial",
                    "description": "Café robusto con cuerpo fuerte y amargor característico",
                    "data": {
                        'Species': 'Robusta',
                        'Aroma': 6.5,
                        'Flavor': 6.8,
                        'Aftertaste': 6.7,
                        'Acidity': 6.2,
                        'Body': 7.5,
                        'Balance': 6.6,
                        'Country.of.Origin': 'Vietnam',
                        'altitude_mean_meters': 800,
                        'Moisture': 0.13,
                        'Category.One.Defects': 1,
                        'Category.Two.Defects': 5,
                    }
                }
            ]
            
            selected_coffee = st.selectbox(
                "Selecciona un café de ejemplo:",
                options=range(len(sample_coffees)),
                format_func=lambda x: f"{sample_coffees[x]['name']} - {sample_coffees[x]['description']}"
            )
            
            if st.button("🎯 Analizar Café Seleccionado"):
                coffee = sample_coffees[selected_coffee]
                
                # Completar datos faltantes
                full_data = coffee['data'].copy()
                full_data.update({
                    'Number.of.Bags': 300,
                    'Cupper.Points': 8.0,
                    'Variety': 'Unknown',
                    'Processing.Method': 'Washed / Wet',
                    'Color': 'Green',
                    'Owner': 'Unknown Farm',
                    'Region': 'Unknown',
                    'altitude_category': 'Media' if full_data['altitude_mean_meters'] < 1600 else 'Alta' if full_data['altitude_mean_meters'] < 2000 else 'Muy-Alta',
                    'altitude_std': 0.0,
                    'sensory_avg': (full_data['Aroma'] + full_data['Flavor'] + full_data['Aftertaste'] + full_data['Acidity'] + full_data['Body'] + full_data['Balance']) / 6,
                    'sensory_std': 0.1,
                    'best_sensory': max(full_data['Aroma'], full_data['Flavor'], full_data['Aftertaste'], full_data['Acidity'], full_data['Body'], full_data['Balance']),
                    'total_defects': full_data['Category.One.Defects'] + full_data['Category.Two.Defects'],
                    'no_defects': 1 if full_data['Category.One.Defects'] == 0 and full_data['Category.Two.Defects'] == 0 else 0,
                    'moisture_category': 'Óptima' if 0.1 <= full_data['Moisture'] <= 0.12 else 'Aceptable',
                    'processing_simple': 'Washed'
                })
                
                results = predict_quality(full_data)
                
                if results:
                    st.success(f"✅ Análisis completado para: **{coffee['name']}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Puntuación", f"{results['predicted_score']}/100")
                        st.metric("Categoría", results['quality_category'])
                    
                    with col2:
                        st.metric("Precisión", results['accuracy_estimate'])
                        st.metric("Error Modelo", f"±{results['model_rmse']:.2f}")
                    
                    # Radar chart para características sensoriales
                    sensory_features = ['Aroma', 'Flavor', 'Aftertaste', 'Acidity', 'Body', 'Balance']
                    values = [coffee['data'][feature] for feature in sensory_features]
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=sensory_features,
                        fill='toself',
                        name=coffee['name']
                    ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 10]
                            )),
                        showlegend=True,
                        title="Perfil Sensorial"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("📊 Entrada Avanzada (CSV)")
            
            st.write("Sube un archivo CSV con las características del café para análisis por lotes.")
            st.write("El archivo debe contener las siguientes columnas:")
            
            required_columns = [
                'Species', 'Aroma', 'Flavor', 'Aftertaste', 'Acidity', 
                'Body', 'Balance', 'Country.of.Origin', 'altitude_mean_meters',
                'Moisture', 'Category.One.Defects', 'Category.Two.Defects'
            ]
            
            st.code(", ".join(required_columns))
            
            uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Archivo cargado con {len(df)} registros")
                    
                    # Verificar columnas requeridas
                    missing_cols = [col for col in required_columns if col not in df.columns]
                    if missing_cols:
                        st.error(f"❌ Faltan columnas requeridas: {', '.join(missing_cols)}")
                    else:
                        if st.button("🔄 Procesar Lote"):
                            results_list = []
                           
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                           
                            for i, row in df.iterrows():
                                # Completar datos faltantes
                                input_data = row.to_dict()
                                input_data.update({
                                    'Number.of.Bags': 300,
                                    'Cupper.Points': 8.0,
                                    'Variety': 'Unknown',
                                    'Processing.Method': 'Washed / Wet',
                                    'Color': 'Green',
                                    'Owner': 'Unknown Farm',
                                    'Region': 'Unknown',
                                    'altitude_category': 'Media' if input_data['altitude_mean_meters'] < 1600 else 'Alta' if input_data['altitude_mean_meters'] < 2000 else 'Muy-Alta',
                                    'altitude_std': 0.0,
                                    'sensory_avg': (input_data['Aroma'] + input_data['Flavor'] + input_data['Aftertaste'] + input_data['Acidity'] + input_data['Body'] + input_data['Balance']) / 6,
                                    'sensory_std': 0.1,
                                    'best_sensory': max(input_data['Aroma'], input_data['Flavor'], input_data['Aftertaste'], input_data['Acidity'], input_data['Body'], input_data['Balance']),
                                    'total_defects': input_data['Category.One.Defects'] + input_data['Category.Two.Defects'],
                                    'no_defects': 1 if input_data['Category.One.Defects'] == 0 and input_data['Category.Two.Defects'] == 0 else 0,
                                    'moisture_category': 'Óptima' if 0.1 <= input_data['Moisture'] <= 0.12 else 'Aceptable',
                                    'processing_simple': 'Washed'
                                })
                               
                                results = predict_quality(input_data)
                                if results:
                                    results_list.append({
                                        'index': i,
                                        'predicted_score': results['predicted_score'],
                                        'quality_category': results['quality_category'],
                                        'accuracy_estimate': results['accuracy_estimate']
                                    })
                               
                                progress = (i + 1) / len(df)
                                progress_bar.progress(progress)
                                status_text.text(f"Procesando registro {i + 1}/{len(df)}")
                            
                            # Mostrar resultados
                            if results_list:
                                results_df = pd.DataFrame(results_list)
                                st.success("✅ Procesamiento completado!")
                               
                                # Estadísticas
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Promedio", f"{results_df['predicted_score'].mean():.2f}")
                                with col2:
                                    st.metric("Máximo", f"{results_df['predicted_score'].max():.2f}")
                                with col3:
                                    st.metric("Mínimo", f"{results_df['predicted_score'].min():.2f}")
                               
                                # Tabla de resultados
                                st.dataframe(results_df)
                               
                                # Gráfico de distribución
                                fig = px.histogram(
                                    results_df, 
                                    x='predicted_score', 
                                    nbins=20,
                                    title="Distribución de Puntuaciones Predichas"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                               
                                # Botón de descarga
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar Resultados",
                                    data=csv,
                                    file_name="coffee_quality_predictions.csv",
                                    mime="text/csv"
                                )
                
                except Exception as e:
                    st.error(f"❌ Error al procesar el archivo: {e}")

# Página 2: Sistema de Recomendación
elif page == "🔍 Sistema de Recomendación":
    st.header("🔍 Sistema de Recomendación de Café - Best Model")
    
    if recommender is None:
        st.error("❌ No se pudo cargar el sistema de recomendación. Por favor, verifica que los datos existan.")
    else:
        st.info("""
        **¿Cómo funciona?**  
        Ingresa tus preferencias sensoriales y el sistema encontrará cafés similares en la base de datos.
        Puedes especificar todas las características o solo las que más te importen.
        """)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Tu Perfil de Preferencias")
            
            flavor = st.slider("Sabor preferido", 0.0, 10.0, 8.0, 0.1, help="Intensidad del sabor que buscas")
            aftertaste = st.slider("Posgusto preferido", 0.0, 10.0, 8.0, 0.1, help="Persistencia y calidad del posgusto")
            aroma = st.slider("Aroma preferido", 0.0, 10.0, 8.0, 0.1, help="Intensidad y complejidad del aroma")
            acidity = st.slider("Acidez preferida", 0.0, 10.0, 8.0, 0.1, help="Brillantez y vivacidad de la acidez")
            body = st.slider("Cuerpo preferido", 0.0, 10.0, 8.0, 0.1, help="Peso y textura en boca")
            balance = st.slider("Balance preferido", 0.0, 10.0, 8.0, 0.1, help="Equilibrio general entre características")
        
        with col2:
            st.subheader("🔍 Filtros Adicionales")
            
            species_filter = st.selectbox(
                "Filtrar por especie:",
                ["Todas", "Arabica", "Robusta"]
            )
            
            top_n = st.slider(
                "Número de recomendaciones:",
                min_value=5,
                max_value=50,
                value=10,
                step=5
            )
            
            country_filter = st.selectbox(
                "Filtrar por país (opcional):",
                ["Todos"] + [
                    "Brazil", "China", "Colombia", "Costa Rica", "El Salvador", "Ethiopia", 
                    "Guatemala", "Honduras", "Indonesia", "Kenya", "Malawi", "Mexico", 
                    "Nicaragua", "Other", "Peru", "Taiwan", "Tanzania", "Thailand", 
                    "Uganda", "United States (Hawaii)"
                ]
            )
        
        # Botón de recomendación
        if st.button("🔍 Obtener Recomendaciones", type="primary"):
            with st.spinner("🔍 Buscando cafés similares..."):
                try:
                    # Sistema mejorado de recomendación con priorización de países
                    recommendations = get_enhanced_recommendations(
                        recommender,
                        flavor=flavor,
                        aftertaste=aftertaste,
                        aroma=aroma,
                        acidity=acidity,
                        body=body,
                        balance=balance,
                        species=species_filter if species_filter != "Todas" else None,
                        country_filter=country_filter if country_filter != "Todos" else None,
                        top_n=top_n
                    )
                    
                    if recommendations.empty:
                        st.warning("❌ No se encontraron cafés que coincidan con tus preferencias.")
                    else:
                        # Mostrar información de priorización
                        if country_filter and country_filter != "Todos":
                            st.info(f"**Priorización activada para {country_filter}**")
                            st.write("Los resultados están ordenados por:")
                            st.write("1. **Alta prioridad** - Cafés del país seleccionado")
                            st.write("2. **Media prioridad** - Cafés de países con características similares")
                            st.write("3. **Baja prioridad** - Otros países con alta similitud sensorial")
                        
                        st.success(f"✅ Se encontraron {len(recommendations)} cafés recomendados!")
                        
                        # Mostrar recomendaciones con información de priorización
                        for i, (_, coffee) in enumerate(recommendations.iterrows(), 1):
                            priority_emoji = "Alta" if coffee.get('priority') == 'Alta' else "Media" if coffee.get('priority') == 'Media' else "Baja"
                            priority_color = "red" if coffee.get('priority') == 'Alta' else "orange" if coffee.get('priority') == 'Media' else "gray"
                            
                            with st.expander(f"{i}. {coffee.get('Owner', 'Unknown')} - {coffee['Total.Cup.Points']:.2f} pts"):
                                # Mostrar priorización
                                st.markdown(f"**Prioridad:** <span style='color:{priority_color};font-weight:bold'>{priority_emoji}</span>", 
                                          unsafe_allow_html=True)
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write("**📍 Origen**")
                                    st.write(f"País: {coffee['Country.of.Origin']}")
                                    st.write(f"Región: {coffee.get('Region', 'Unknown')}")
                                    st.write(f"Especie: {coffee['Species']}")
                                    
                                    st.write("**☕ Características**")
                                    st.write(f"Variedad: {coffee.get('Variety', 'Unknown')}")
                                    st.write(f"Proceso: {coffee.get('Processing.Method', 'Unknown')}")
                                
                                with col2:
                                    st.write("**🎯 Perfil Sensorial**")
                                    st.metric("Sabor", f"{coffee['Flavor']:.1f}")
                                    st.metric("Aroma", f"{coffee['Aroma']:.1f}")
                                    st.metric("Posgusto", f"{coffee['Aftertaste']:.1f}")
                                    st.metric("Acidez", f"{coffee['Acidity']:.1f}")
                                    st.metric("Cuerpo", f"{coffee['Body']:.1f}")
                                    st.metric("Balance", f"{coffee['Balance']:.1f}")
                                    st.metric("Similitud", f"{coffee['similarity_score']:.3f}")
                        
                        # Tabla resumen
                        st.subheader("📊 Tabla Resumen")
                        display_cols = ['Owner', 'Country.of.Origin', 'Total.Cup.Points', 'Flavor', 'Aroma', 'Aftertaste', 'similarity_score']
                        available_cols = [col for col in display_cols if col in recommendations.columns]
                        st.dataframe(recommendations[available_cols], use_container_width=True)
                        
                        # Gráfico comparativo
                        if len(recommendations) > 1:
                            fig = px.bar(
                                recommendations.head(10),
                                x='Owner',
                                y='Total.Cup.Points',
                                title="Top 10 Cafés Recomendados",
                                labels={'Total.Cup.Points': 'Puntuación', 'Owner': 'Café'}
                            )
                            fig.update_xaxes(tickangle=45)
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error al obtener recomendaciones: {e}")

# Página 3: Análisis de Modelos
elif page == "📊 Análisis de Modelos":
    st.header("📊 Análisis de Modelos - Best Model")
    
    if model is None or metadata is None:
        st.error("❌ No se pudo cargar la información del modelo.")
    else:
        # Información del mejor modelo
        st.subheader("🏆 Mejor Modelo Actual")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Modelo", metadata['best_model_name'])
        with col2:
            st.metric("RMSE", f"{metadata['best_rmse']:.3f}")
        with col3:
            st.metric("R²", f"{metadata['best_r2']:.3f}")
        with col4:
            st.metric("Precisión", f"{(1 - metadata['best_rmse']/100)*100:.1f}%")
        
        # Comparación de modelos
        try:
            comparison_df = pd.read_csv("metrics/model_comparison_ranking.csv")
            st.subheader("📈 Comparación de Todos los Modelos")
            st.dataframe(comparison_df, use_container_width=True)
            
            # Gráfico comparativo
            fig = px.bar(
                comparison_df,
                x="Model",
                y="MAE",
                title="Comparación de Modelos (MAE - menor es mejor)",
                color="MAE",
                color_continuous_scale="RdYlGn_r"
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Gráfico de R²
            fig2 = px.bar(
                comparison_df,
                x="Model",
                y="R2",
                title="R² por Modelo (mayor es mejor)",
                color="R2",
                color_continuous_scale="RdYlGn"
            )
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)
            
        except Exception as e:
            st.warning(f"No se pudo cargar la comparación de modelos: {e}")
        
        # Análisis de características
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Importancia de Características")
            
            try:
                importances = model.feature_importances_
                feature_names = [f"feature_{i}" for i in range(len(importances))]
                
                fi_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Top 15
                top15 = fi_df.head(15)
                
                fig = px.bar(
                    top15,
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 15 Características Más Importantes",
                    labels={'importance': 'Importancia', 'feature': 'Característica'}
                )
                fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"No se pudo mostrar la importancia de características: {e}")
        
        # Información técnica
        with st.expander("ℹ️ Información Técnica del Modelo"):
            feature_cols = st.columns(2)
            
            with feature_cols[0]:
                st.write(f"**Tipo de modelo:** {type(model).__name__}")
                st.write(f"**Total de features:** {len(metadata['feature_names'])}")
                st.write(f"**Mejor modelo:** {metadata['best_model_name']}")
                st.write(f"**Precisión:** {(1 - metadata['best_rmse']/100)*100:.1f}%")
            
            with feature_cols[1]:
                st.write(f"**RMSE:** {metadata['best_rmse']:.3f}")
                st.write(f"**R²:** {metadata['best_r2']:.3f}")
                st.write(f"**MAE:** {metadata['best_mae']:.3f}")
                st.write(f"**MAPE:** {metadata['best_mape']:.3f}")

# Página 5: Demo Interactivo
elif page == "🎮 Demo Interactivo":
    st.header("🎮 Demo Interactivo - Modo Consola")
    
    st.markdown("""
    Esta sección ejecuta los demos originales en modo consola dentro de Streamlit.
    Puedes ver cómo funcionan los sistemas de predicción y recomendación.
    """)
    
    # Tabs para los dos demos
    tab_pred, tab_rec = st.tabs(["🎯 Demo Predicción Interactivo", "🔍 Demo Recomendación Interactivo"])
    
    with tab_pred:
        st.subheader("🎯 Demo de Predicción Interactivo")
        st.write("Simulador interactivo de calidad de café")
        
        # Cargar datos para el demo
        try:
            prefix = configurar_rutas()
            path_data = os.path.join(prefix, "data", "processed", "coffee_data_for_training.csv")
            path_metrics = os.path.join(prefix, "metrics", "model_comparison_ranking.csv")
            path_models = os.path.join(prefix, "models", "prediction")
            
            if os.path.exists(path_data) and os.path.exists(path_metrics):
                df_train = pd.read_csv(path_data)
                df_results = pd.read_csv(path_metrics)
                
                best_model_name = df_results.iloc[0]['Model']
                mae_modelo = df_results.iloc[0]['MAE'] 
                best_model = joblib.load(os.path.join(path_models, f"coffee_model_{best_model_name}.pkl"))
                
                st.info(f"🤖 Motor: {best_model_name} | Error Medio: ±{mae_modelo:.2f}")
                
                # Selección interactiva de país
                paises = sorted(df_train['Country.of.Origin'].unique())
                selected_pais = st.selectbox("🌍 Seleccione País de Origen:", paises)
                
                if selected_pais:
                    regiones = sorted(df_train[df_train['Country.of.Origin'] == selected_pais]['Region'].unique())
                    selected_region = st.selectbox(f"📍 Seleccione Región de {selected_pais}:", regiones)
                    
                    if selected_region:
                        # Selección de otras características
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            variedades = sorted(df_train['Variety'].dropna().unique())
                            selected_variedad = st.selectbox("🌱 Variedad:", variedades)
                            
                            procesos = sorted(df_train['Processing.Method'].dropna().unique())
                            selected_proceso = st.selectbox("⚙️ Proceso:", procesos)
                        
                        with col2:
                            colores = sorted(df_train['Color'].dropna().unique())
                            selected_color = st.selectbox("🎨 Color:", colores)
                        
                        # Altitud
                        df_geo = df_train[(df_train['Country.of.Origin'] == selected_pais) & (df_train['Region'] == selected_region)]
                        alt_min, alt_max = df_geo['altitude_mean_meters'].min(), df_geo['altitude_mean_meters'].max()
                        
                        st.info(f"💡 En {selected_region}, el rango histórico es {alt_min:.0f}m - {alt_max:.0f}m")
                        selected_altitud = st.slider(f"🏔️ Altitud (msnm):", int(alt_min), int(alt_max), 1500)
                        
                        # Botón de predicción
                        if st.button("🔮 Predecir Calidad", type="primary"):
                            try:
                                data_final = pd.DataFrame({
                                    'Country.of.Origin': [selected_pais],
                                    'Region': [selected_region],
                                    'Variety': [selected_variedad],
                                    'Processing.Method': [selected_proceso],
                                    'Color': [selected_color],
                                    'categoria_altitud': ['Alta' if selected_altitud > 1200 else 'Media' if selected_altitud > 800 else 'Baja'],
                                    'Moisture': [0.11],
                                    'Category.One.Defects': [0],
                                    'Category.Two.Defects': [0],
                                    'altitude_mean_meters': [selected_altitud]
                                })

                                pred = best_model.predict(data_final)[0]
                                
                                umbral_mediana = 82.5
                                limite_inf = pred - mae_modelo
                                limite_sup = pred + mae_modelo

                                # Mostrar resultados
                                st.success("✅ Predicción realizada!")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("📊 Puntaje Estimado", f"{pred:.2f}")
                                
                                with col2:
                                    st.metric("📉 Límite Inferior", f"{limite_inf:.2f}")
                                
                                with col3:
                                    st.metric("📈 Límite Superior", f"{limite_sup:.2f}")
                                
                                # Categoría
                                if pred >= umbral_mediana:
                                    st.success(f"🏆 **CATEGORÍA: PREMIUM**")
                                    st.info("(Puntaje por encima de la mediana del mercado)")
                                else:
                                    st.warning(f"📦 **CATEGORÍA: ESTÁNDAR**")
                                    st.info("(Puntaje dentro del rango base de comercialización)")
                                
                            except Exception as e:
                                st.error(f"❌ Error en la predicción: {e}")
            else:
                st.error("❌ No se encontraron los archivos de datos necesarios")
                
        except Exception as e:
            st.error(f"❌ Error al cargar los datos: {e}")
    
    with tab_rec:
        st.subheader("🔍 Demo de Recomendación Interactivo")
        st.write("Buscador de cafés premium por perfil")
        
        try:
            csv_path = './models/recommender/coffee_flavor_segments.csv'
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                perfiles = sorted(list(df['Perfil_Nombre'].unique()))
                
                # Selección interactiva de perfil
                selected_perfil = st.selectbox("🎯 Paso 1: Selecciona un perfil de sabor:", perfiles)
                
                if selected_perfil:
                    # Filtramos países para ese perfil
                    df_perfil = df[df['Perfil_Nombre'] == selected_perfil]
                    paises = sorted(df_perfil['Country.of.Origin'].unique())
                    
                    # Selección de país
                    opciones_pais = paises + ["VER TODOS LOS PAÍSES"]
                    selected_pais_option = st.selectbox(f"🌍 Paso 2: Selecciona un país con perfil '{selected_perfil}':", opciones_pais)
                    
                    # Botón de búsqueda
                    if st.button("🔍 Buscar Cafés", type="primary"):
                        try:
                            # Lógica de Filtrado
                            resultados = df_perfil.copy()
                            
                            if selected_pais_option != "VER TODOS LOS PAÍSES":
                                resultados = resultados[resultados['Country.of.Origin'] == selected_pais_option]
                                st.info(f"🔍 Buscando los mejores de {selected_pais_option}...")
                            else:
                                st.info(f"🔍 Mostrando resultados globales para '{selected_perfil}'...")
                            
                            if resultados.empty:
                                st.warning("❌ No se encontraron resultados.")
                            else:
                                final_top = (resultados
                                             .sort_values(by='Total.Cup.Points', ascending=False)
                                             .drop_duplicates(subset=['Nombre_Comercial'])
                                             [['Nombre_Comercial', 'Total.Cup.Points', 'Categoria_Calidad', 'Region', 'Variety']]
                                             .head(10))

                                st.success(f"🏆 TOP {len(final_top)} CAFÉS RECOMENDADOS:")
                                
                                # Mostrar resultados en tabla
                                display_df = final_top.reset_index(drop=True)
                                display_df.index = display_df.index + 1
                                display_df.columns = ['Marca', 'Puntaje', 'Categoría', 'Región', 'Variedad']
                                
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Gráfico de puntajes
                                fig = px.bar(
                                    final_top.head(5),
                                    x='Nombre_Comercial',
                                    y='Total.Cup.Points',
                                    title=f"Top 5 Cafés - Perfil: {selected_perfil}",
                                    labels={'Nombre_Comercial': 'Café', 'Total.Cup.Points': 'Puntaje'}
                                )
                                fig.update_xaxes(tickangle=45)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.info(f"💡 Perfil sensorial seleccionado: **{selected_perfil}**")
                                
                        except Exception as e:
                            st.error(f"❌ Error en la búsqueda: {e}")
            else:
                st.error("❌ No existe el archivo de segmentos. Ejecuta el Profiler primero.")
                
        except Exception as e:
            st.error(f"❌ Ocurrió un error: {e}")
    
    st.info("""
    💡 **Nota**: Esta es una versión interactiva de los demos originales.
    Puedes seleccionar diferentes opciones y ver los resultados en tiempo real.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📅 Última actualización: {}</p>
        <p>☕ Desarrollado con ❤️ por el equipo de Data Science Henry Bootcamp</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# Página 4: Información
elif page == "ℹ️ Información":
    st.header("ℹ️ Información del Sistema - Best Model")
    
    st.markdown("""
    ## 🎯 **Acerca del Sistema**
    
    Esta aplicación utiliza el **mejor modelo disponible** para predecir la calidad del café
    basándose en sus características sensoriales y de origen. El sistema selecciona automáticamente
    el modelo con mejor rendimiento según las métricas de evaluación.
    
    ## 📊 **Modelos Evaluados**
    
    El sistema ha entrenado y evaluado múltiples modelos de Machine Learning:
    - **SVR** (Support Vector Regression)
    - **CatBoost**
    - **Linear Regression**
    - **Voting Ensemble**
    - **XGBoost**
    - **Gradient Boosting**
    - **Random Forest**
    
    ## 🏆 **Modelo Actual**
    
    El sistema selecciona dinámicamente el modelo con mejor rendimiento
    según el ranking de métricas (MAE, RMSE, R²).
    
    ## 📈 **Métricas de Evaluación**
    
    - **MAE** (Mean Absolute Error): Error promedio absoluto
    - **RMSE** (Root Mean Square Error): Error cuadrático medio
    - **R²** (R-squared): Varianza explicada
    - **MAPE** (Mean Absolute Percentage Error): Error porcentual
    
    ## 🔄 **Funcionalidades**
    
    1. **Predicción de Calidad**: Ingresa las características del café o usa ejemplos predefinidos
    2. **Sistema de Recomendación**: Define tu perfil de preferencias y obtén cafés similares
    3. **Análisis de Modelos**: Explora las métricas y rendimiento de diferentes modelos
    4. **Demo Interactivo**: Prueba los demos originales en modo interactivo
    
    ## 🚀 **Próximas Mejoras**
    
    - [ ] Integración con APIs de café en tiempo real
    - [ ] Sistema de calificación de usuarios
    - [ ] Análisis de tendencias del mercado
    - [ ] Exportación avanzada de reportes
    
    ## 👥 **Equipo de Desarrollo**
    
    **Data Science Henry Bootcamp - Proyecto Final**
    
    *Sistema integrado para análisis y predicción de calidad de café*
    """.format(
        model_type=metadata['best_model_name'] if metadata else 'No disponible'
    ))
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📅 Última actualización: {}</p>
        <p>☕ Desarrollado con ❤️ por el equipo de Data Science Henry Bootcamp</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# Footer general
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>☕ Coffee Quality Predictor & Recommendation System - Best Model | Data Science Henry Bootcamp</p>
</div>
""", unsafe_allow_html=True)
