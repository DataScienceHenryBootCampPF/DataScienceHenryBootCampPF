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
    page_title="☕ Gradient Boosting - Coffee Quality",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🌿 Gradient Boosting Coffee Quality Predictor")
st.markdown("""
*Modelo específico Gradient Boosting para predecir la calidad del café*
""")

# Cargar modelos y datos
@st.cache_resource
def load_gradient_boosting_model():
    """Carga el modelo Gradient Boosting específico"""
    try:
        model_path = "models/prediction/coffee_model_GradientBoosting.pkl"
        preprocessor_path = "models/prediction/preprocessor.pkl"
        metadata_path = "models/prediction/training_metadata.pkl"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else None
        
        return model, preprocessor, metadata
    except Exception as e:
        st.error(f"Error al cargar el modelo Gradient Boosting: {e}")
        return None, None, None

@st.cache_resource
def load_recommender():
    """Carga el sistema de recomendación"""
    try:
        return HybridCoffeeRecommendationSystem()
    except Exception as e:
        st.error(f"Error al cargar el sistema de recomendación: {e}")
        return None

# Cargar modelos
model, preprocessor, metadata = load_gradient_boosting_model()
recommender = load_recommender()

# Sidebar para navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Selecciona una funcionalidad:",
    ["🎯 Predicción de Calidad", "🔍 Sistema de Recomendación", "📊 Análisis del Modelo", "ℹ️ Información"]
)

# Función para predecir calidad
def predict_quality(input_data):
    """Realiza la predicción de calidad del café con Gradient Boosting"""
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocesar datos si el preprocesador está disponible
        if preprocessor is not None:
            processed_data = preprocessor.transform(df)
        else:
            # Si no hay preprocesador, usar los datos directamente
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
        
        # Determinar categoría de calidad
        if prediction >= 85:
            quality_category = "Excelente"
            color = "🟢"
        elif prediction >= 80:
            quality_category = "Muy Bueno"
            color = "🔵"
        elif prediction >= 75:
            quality_category = "Bueno"
            color = "🟡"
        else:
            quality_category = "Regular"
            color = "🟠"
        
        results = {
            'predicted_score': round(prediction, 2),
            'quality_category': quality_category,
            'color': color,
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
    st.header("🎯 Predicción de Calidad - Gradient Boosting")
    
    if model is None:
        st.error("❌ No se pudo cargar el modelo Gradient Boosting. Por favor, verifica que los archivos del modelo existan.")
    else:
        # Información del modelo
        st.info(f"""
        **Modelo:** Gradient Boosting  
        **Tipo:** {type(model).__name__}  
        **Precisión Estimada:** {(1 - 1.472/100)*100:.1f}% (basado en MAE histórico)
        """)
        
        # Tabs para diferentes modos de entrada
        tab1, tab2 = st.tabs(["📝 Entrada Manual", "🎲 Ejemplos Predefinidos"])
        
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
                country = st.selectbox("País de Origen", ["Colombia", "Ethiopia", "Kenya", "Brazil", "Vietnam", "Guatemala", "Costa Rica", "Peru"])
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
                    # Mostrar resultados
                    st.success("✅ Predicción realizada con éxito!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Puntuación Predicha",
                            value=f"{results['predicted_score']}/100",
                            delta=f"{results['color']} {results['quality_category']}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Intervalo de Confianza (95%)",
                            value=f"{results['confidence_interval']['lower']} - {results['confidence_interval']['upper']}"
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
                            'bar': {'color': "darkgreen"},
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
                        st.metric("Categoría", f"{results['color']} {results['quality_category']}")
                    
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

# Página 2: Sistema de Recomendación
elif page == "🔍 Sistema de Recomendación":
    st.header("🔍 Sistema de Recomendación de Café")
    
    if recommender is None:
        st.error("❌ No se pudo cargar el sistema de recomendación. Por favor, verifica que los datos existan.")
    else:
        st.info("""
        **¿Cómo funciona?**  
        Ingresa tus preferencias sensoriales y el sistema encontrará cafés similares en la base de datos.
        Puedes especificar todas las características o solo las que más te importan.
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
                ["Todos"] + ["Colombia", "Ethiopia", "Kenya", "Brazil", "Vietnam", "Guatemala", "Costa Rica", "Peru"]
            )
        
        # Botón de recomendación
        if st.button("🔍 Obtener Recomendaciones", type="primary"):
            with st.spinner("🔍 Buscando cafés similares..."):
                try:
                    recommendations = recommender.recomendar(
                        Flavor=flavor,
                        Aftertaste=aftertaste,
                        Aroma=aroma,
                        Acidity=acidity,
                        Body=body,
                        Balance=balance,
                        species=species_filter if species_filter != "Todas" else None,
                        top_n=top_n
                    )
                    
                    if recommendations.empty:
                        st.warning("❌ No se encontraron cafés que coincidan con tus preferencias.")
                    else:
                        st.success(f"✅ Se encontraron {len(recommendations)} cafés recomendados!")
                        
                        # Mostrar recomendaciones
                        for i, (_, coffee) in enumerate(recommendations.iterrows(), 1):
                            with st.expander(f"{i}. {coffee.get('Owner', 'Unknown')} - {coffee['Total.Cup.Points']:.2f} pts"):
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

# Página 3: Análisis del Modelo
elif page == "📊 Análisis del Modelo":
    st.header("📊 Análisis del Modelo Gradient Boosting")
    
    if model is None:
        st.error("❌ No se pudo cargar la información del modelo.")
    else:
        # Información del modelo
        st.subheader("🌿 Modelo Gradient Boosting")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Modelo", "Gradient Boosting")
        with col2:
            st.metric("Tipo", type(model).__name__)
        with col3:
            st.metric("MAE Histórico", "1.472")
        with col4:
            st.metric("Precisión", f"{(1 - 1.472/100)*100:.1f}%")
        
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
                st.write(f"**Algoritmo:** Gradient Boosting")
                st.write(f"**MAE histórico:** 1.472")
                st.write(f"**Precisión:** {(1 - 1.472/100)*100:.1f}%")
            
            with feature_cols[1]:
                st.write(f"**Características:** {len(model.feature_importances_) if hasattr(model, 'feature_importances_') else 'N/A'}")
                st.write(f"**Modelo específico:** Gradient Boosting")
                st.write(f"**Uso:** Predicción de calidad")

# Página 4: Información
elif page == "ℹ️ Información":
    st.header("ℹ️ Información del Sistema - Gradient Boosting")
    
    st.markdown("""
    ## 🌿 **Acerca del Modelo Gradient Boosting**
    
    Esta página utiliza específicamente el modelo **Gradient Boosting** para predecir la calidad del café
    basándose en sus características sensoriales y de origen.
    
    ## 📊 **Características del Modelo**
    
    - **Algoritmo:** Gradient Boosting Regressor
    - **MAE Histórico:** 1.472
    - **Precisión Estimada:** 85.3%
    - **Uso:** Predicción especializada con Gradient Boosting
    
    ## 🔄 **Funcionalidades**
    
    1. **Predicción de Calidad**: Ingresa las características del café o usa ejemplos predefinidos
    2. **Sistema de Recomendación**: Define tu perfil de preferencias y obtén cafés similares
    3. **Análisis del Modelo**: Explora las características y rendimiento del modelo Gradient Boosting
    
    ## 🎯 **Diferencia con Best Model**
    
    - **Best Model**: Usa dinámicamente el mejor modelo según ranking (actualmente SVR)
    - **Gradient Boosting**: Usa específicamente el modelo Gradient Boosting sin importar el ranking
    
    ## 👥 **Equipo de Desarrollo**
    
    **Data Science Henry Bootcamp - Proyecto Final**
    
    *Sistema integrado para análisis y predicción de calidad de café*
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📅 Última actualización: {}</p>
        <p>🌿 Gradient Boosting Coffee Quality Predictor | Data Science Henry Bootcamp</p>
    </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)

# Footer general
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    <p>🌿 Gradient Boosting Coffee Quality Predictor | Data Science Henry Bootcamp</p>
</div>
""", unsafe_allow_html=True)
