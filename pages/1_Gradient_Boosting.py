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
    page_title="☕ Coffee Quality Predictor",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("☕ Coffee Quality Predictor & Recommendation System")
st.markdown("""
*Aplicación interactiva para predecir la calidad del café y encontrar recomendaciones personalizadas*
""")

# Cargar modelos y datos
@st.cache_resource
def load_models():
    """Carga el modelo predictivo y componentes relacionados"""
    try:
        model_path = "models/prediction/best_model.pkl"
        preprocessor_path = "models/prediction/preprocessor.pkl"
        metadata_path = "models/prediction/training_metadata.pkl"
        
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        metadata = joblib.load(metadata_path)
        
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

# Cargar modelos
model, preprocessor, metadata = load_models()
recommender = load_recommender()

# Sidebar para navegación
st.sidebar.title("📋 Navegación")
page = st.sidebar.selectbox(
    "Selecciona una funcionalidad:",
    ["🎯 Predicción de Calidad", "🔍 Sistema de Recomendación", "📊 Análisis de Modelos", "ℹ️ Información"]
)

# Función para predecir calidad
def predict_quality(input_data):
    """Realiza la predicción de calidad del café"""
    try:
        # Convertir a DataFrame
        df = pd.DataFrame([input_data])
        
        # Preprocesar datos
        processed_data = preprocessor.transform(df)
        
        # Hacer predicción
        prediction = model.predict(processed_data)[0]
        
        # Calcular intervalo de confianza
        rmse = metadata['best_rmse']
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
    st.header("🎯 Predicción de Calidad de Café")
    
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
        tab1, tab2, tab3 = st.tabs(["📝 Entrada Manual", "🎲 Ejemplos Predefinidos", "📊 Entrada Avanzada"])
        
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
    st.header("🔍 Sistema de Recomendación de Café")
    
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
            
            similarity_threshold = st.slider(
                "Umbral de similitud mínimo:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Similitud mínima para incluir una recomendación"
            )
        
        if st.button("🔮 Encontrar Cafés Similares", type="primary"):
            try:
                with st.spinner("🔍 Buscando cafés similares..."):
                    # Preparar filtros
                    species = None if species_filter == "Todas" else species_filter
                    
                    # Ejecutar recomendación
                    recommendations = recommender.recomendar(
                        Flavor=flavor,
                        Aftertaste=aftertaste,
                        Aroma=aroma,
                        Acidity=acidity,
                        Body=body,
                        Balance=balance,
                        species=species
                    )
                    
                    if recommendations is not None and not recommendations.empty:
                        # Filtrar por umbral de similitud
                        recommendations = recommendations[
                            recommendations['similarity_score'] >= similarity_threshold
                        ].head(top_n)
                        
                        if not recommendations.empty:
                            st.success(f"✅ Se encontraron {len(recommendations)} cafés similares!")
                            
                            # Tabla de resultados
                            display_cols = [
                                'Country.of.Origin', 'Region', 'Species', 'Total.Cup.Points',
                                'similarity_score', 'Flavor', 'Aftertaste', 'Aroma', 
                                'Acidity', 'Body', 'Balance'
                            ]
                            
                            # Limpiar columnas que no existen
                            display_cols = [c for c in display_cols if c in recommendations.columns]
                            
                            st.dataframe(
                                recommendations[display_cols].round(2),
                                use_container_width=True
                            )
                            
                            # Visualizaciones
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Gráfico de similitud
                                fig = px.bar(
                                    recommendations.head(10),
                                    x='similarity_score',
                                    y='Country.of.Origin',
                                    orientation='h',
                                    title="Top 10 Cafés por Similitud",
                                    labels={'similarity_score': 'Puntuación de Similitud'}
                                )
                                fig.update_layout(height=400)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col2:
                                # Radar chart comparativo
                                if len(recommendations) > 0:
                                    top_coffee = recommendations.iloc[0]
                                    
                                    fig = go.Figure()
                                    
                                    # Perfil del usuario
                                    user_profile = [flavor, aftertaste, aroma, acidity, body, balance]
                                    fig.add_trace(go.Scatterpolar(
                                        r=user_profile,
                                        theta=['Flavor', 'Aftertaste', 'Aroma', 'Acidity', 'Body', 'Balance'],
                                        fill='toself',
                                        name='Tu Perfil',
                                        line_color='blue'
                                    ))
                                    
                                    # Perfil del café recomendado
                                    coffee_profile = [
                                        top_coffee.get('Flavor', 0),
                                        top_coffee.get('Aftertaste', 0),
                                        top_coffee.get('Aroma', 0),
                                        top_coffee.get('Acidity', 0),
                                        top_coffee.get('Body', 0),
                                        top_coffee.get('Balance', 0)
                                    ]
                                    fig.add_trace(go.Scatterpolar(
                                        r=coffee_profile,
                                        theta=['Flavor', 'Aftertaste', 'Aroma', 'Acidity', 'Body', 'Balance'],
                                        fill='toself',
                                        name=f"Mejor Match: {top_coffee.get('Country.of.Origin', 'Unknown')}",
                                        line_color='red'
                                    ))
                                    
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(visible=True, range=[0, 10])
                                        ),
                                        showlegend=True,
                                        title="Comparación de Perfiles Sensoriales"
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Detalles del mejor match
                            st.subheader("🏆 Mejor Recomendación")
                            best_match = recommendations.iloc[0]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("País", best_match.get('Country.of.Origin', 'Unknown'))
                                st.metric("Región", best_match.get('Region', 'Unknown'))
                            
                            with col2:
                                st.metric("Puntuación Total", f"{best_match.get('Total.Cup.Points', 0):.2f}")
                                st.metric("Similitud", f"{best_match.get('similarity_score', 0):.3f}")
                            
                            with col3:
                                st.metric("Especie", best_match.get('Species', 'Unknown'))
                                st.metric("Variedad", best_match.get('Variety', 'Unknown'))
                            
                        else:
                            st.warning("⚠️ No se encontraron cafés que cumplan con los criterios de similitud. Intenta reducir el umbral de similitud.")
                    else:
                        st.error("❌ No se encontraron recomendaciones con los criterios especificados.")
                        
            except Exception as e:
                st.error(f"❌ Error en el sistema de recomendación: {e}")

# Página 3: Análisis de Modelos
elif page == "📊 Análisis de Modelos":
    st.header("📊 Análisis de Modelos y Métricas")
    
    if metadata is None:
        st.error("❌ No se pudieron cargar los metadatos del modelo.")
    else:
        # Información general del modelo
        st.subheader("🎯 Información del Modelo Actual")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Modelo", metadata['best_model_name'])
        with col2:
            st.metric("RMSE", f"{metadata['best_rmse']:.3f}")
        with col3:
            st.metric("R²", f"{metadata['best_r2']:.3f}")
        with col4:
            st.metric("Features", len(metadata['feature_names']))
        
        # Lista de features
        st.subheader("📋 Características del Modelo")
        
        feature_cols = st.columns(2)
        with feature_cols[0]:
            st.write("**Features principales:**")
            for i, feature in enumerate(metadata['feature_names'][:10]):
                st.write(f"• {feature}")
        
        with feature_cols[1]:
            st.write(f"**Total de features:** {len(metadata['feature_names'])}")
            st.write(f"**Mejor modelo:** {metadata['best_model_name']}")
            st.write(f"**Precisión:** {(1 - metadata['best_rmse']/100)*100:.1f}%")
        
        # Análisis comparativo si hay múltiples modelos
        try:
            analysis_summary = joblib.load("models/prediction/analysis_summary.pkl")
            
            st.subheader("🏆 Comparación de Modelos")
            
            if 'model_comparison' in analysis_summary:
                comparison_df = pd.DataFrame(analysis_summary['model_comparison'])
                st.dataframe(comparison_df.round(4), use_container_width=True)
                
                # Gráfico comparativo
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('RMSE (menor es mejor)', 'R² (mayor es mejor)'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(
                    go.Bar(x=comparison_df['Model'], y=comparison_df['RMSE'], name='RMSE'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=comparison_df['Model'], y=comparison_df['R²'], name='R²'),
                    row=1, col=2
                )
                
                fig.update_layout(title_text="Comparación de Rendimiento de Modelos")
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.warning(f"No se pudo cargar el análisis comparativo: {e}")
        
        # Importancia de características (si está disponible)
        try:
            if hasattr(model, 'feature_importances_'):
                st.subheader("🎯 Importancia de Características")
                
                feature_importance = pd.DataFrame({
                    'feature': metadata['feature_names'],
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                
                fig = px.bar(
                    feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Características Más Importantes"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.info("El modelo actual no proporciona información de importancia de características.")

# Página 4: Información
elif page == "ℹ️ Información":
    st.header("ℹ️ Información del Sistema")
    
    st.markdown("""
    ## 🎯 Propósito del Sistema
    
    Esta aplicación web interactiva permite:
    - **Predecir la calidad del café** utilizando machine learning
    - **Encontrar cafés similares** basados en preferencias personales
    - **Analizar diferentes modelos** y sus métricas de rendimiento
    
    ## 🤖 Tecnologías Utilizadas
    
    - **Streamlit**: Framework para aplicaciones web interactivas
    - **Scikit-learn**: Biblioteca de machine learning
    - **Plotly**: Visualizaciones interactivas
    - **Pandas**: Manipulación de datos
    - **Joblib**: Serialización de modelos
    
    ## 📊 Modelo Predictivo
    
    El sistema utiliza el mejor modelo entrenado con las siguientes características:
    - **Algoritmo**: {model_type if 'model_type' in locals() else 'Variados (selecciona el mejor)'}
    - **Métricas**: RMSE, R², precisión estimada
    - **Features**: Múltiples características sensoriales y de origen
    
    ## 🔍 Sistema de Recomendación
    
    Sistema híbrido que combina:
    - **Filtrado basado en contenido**: Similitud de características sensoriales
    - **Filtros personalizados**: Por especie, país, región
    - **Puntuación de similitud**: Métrica de distancia euclidiana normalizada
    
    ## 📝 Cómo Usar
    
    1. **Predicción de Calidad**: Ingresa las características del café o usa ejemplos predefinidos
    2. **Sistema de Recomendación**: Define tu perfil de preferencias y obtén cafés similares
    3. **Análisis de Modelos**: Explora las métricas y rendimiento de diferentes modelos
    
    ## 🚀 Próximas Mejoras
    
    - [ ] Integración con APIs de café en tiempo real
    - [ ] Sistema de calificación de usuarios
    - [ ] Análisis de tendencias del mercado
    - [ ] Exportación avanzada de reportes
    
    ## 👥 Equipo de Desarrollo
    
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
    <p>☕ Coffee Quality Predictor & Recommendation System | Data Science Henry Bootcamp</p>
</div>
""", unsafe_allow_html=True)
