import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Carga y prepara los datos para entrenamiento."""
    print("📁 Cargando dataset procesado...")
    
    # Ruta al archivo de datos procesados
    data_path = "./data/processed/coffee_data_cleaned_final.csv"
    
    try:
        df = pd.read_csv(data_path)
        print(f"✅ Dataset cargado exitosamente: {df.shape}")
        print(f"   - Filas: {df.shape[0]}")
        print(f"   - Columnas: {df.shape[1]}")
        print(f"   - Variable objetivo: Total.Cup.Points")
        
        # Verificar que la variable objetivo existe
        if 'Total.Cup.Points' not in df.columns:
            print("❌ Error: No se encontró la columna 'Total.Cup.Points'")
            return None, None, None, None, None
        
        # Análisis preliminar
        print("\n📊 Análisis preliminar de datos:")
        print(f"   - Puntaje mínimo: {df['Total.Cup.Points'].min():.2f}")
        print(f"   - Puntaje máximo: {df['Total.Cup.Points'].max():.2f}")
        print(f"   - Puntaje promedio: {df['Total.Cup.Points'].mean():.2f}")
        print(f"   - Desviación estándar: {df['Total.Cup.Points'].std():.2f}")
        
        # Identificar variables categóricas y numéricas
        categorical_features = []
        numerical_features = []
        
        for col in df.columns:
            if col == 'Total.Cup.Points':
                continue
            elif df[col].dtype == 'object':
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        print(f"\n🔍 Variables identificadas:")
        print(f"   - Categóricas ({len(categorical_features)}): {categorical_features[:3]}...")
        print(f"   - Numéricas ({len(numerical_features)}): {numerical_features[:3]}...")
        
        # Separar features y target
        X = df.drop('Total.Cup.Points', axis=1)
        y = df['Total.Cup.Points']
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Crear preprocesador
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Fit y transform
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        print(f"✅ Datos preparados: Train={X_train_processed.shape}, Test={X_test_processed.shape}")
        
        return X_train_processed, X_test_processed, y_train, y_test, preprocessor
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {data_path}")
        print("   Asegúrate de haber ejecutado primero el notebook de limpieza")
        return None, None, None, None, None
    except Exception as e:
        print(f"❌ Error al cargar datos: {str(e)}")
        return None, None, None, None, None

def calculate_metrics(y_true, y_pred):
    """Calcula métricas de evaluación."""
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def train_all_models(X_train, X_test, y_train, y_test):
    """Entrena todos los modelos y devuelve resultados."""
    print("\n🚀 Entrenando modelos...")
    
    # Definir modelos
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR()
    }
    
    # Hiperparámetros para optimización
    param_grids = {
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.05],
            'max_depth': [3, 5]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [5, 10]
        }
    }
    
    results = []
    trained_models = {}
    
    # Entrenar modelos base
    for name, model in models.items():
        print(f"\n📊 Entrenando {name}...")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Métricas
        train_metrics = calculate_metrics(y_train, y_train_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                   scoring='neg_mean_squared_error')
        cv_rmse = np.sqrt(-cv_scores)
        
        # Guardar modelo
        trained_models[name] = model
        
        print(f"   ✅ RMSE Test: {test_metrics['rmse']:.3f}")
        print(f"   ✅ R² Test: {test_metrics['r2']:.3f}")
        print(f"   ✅ CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
        
        results.append({
            'Model': name,
            'Train_RMSE': train_metrics['rmse'],
            'Test_RMSE': test_metrics['rmse'],
            'Train_R2': train_metrics['r2'],
            'Test_R2': test_metrics['r2'],
            'CV_RMSE_Mean': cv_rmse.mean(),
            'CV_RMSE_Std': cv_rmse.std()
        })
    
    # Optimizar mejores modelos
    print("\n🔧 Optimizando hiperparámetros...")
    
    results_df = pd.DataFrame(results)
    best_models = results_df.nsmallest(3, 'Test_RMSE')['Model'].tolist()
    
    for model_name in best_models:
        if model_name in param_grids:
            print(f"\n⚡ Optimizando {model_name}...")
            
            base_model = models[model_name]
            param_grid = param_grids[model_name]
            
            # Grid Search
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Mejor modelo
            best_model = grid_search.best_estimator_
            
            # Evaluación
            y_test_pred = best_model.predict(X_test)
            test_metrics = calculate_metrics(y_test, y_test_pred)
            
            # Guardar modelo optimizado
            optimized_name = f"{model_name}_Optimized"
            trained_models[optimized_name] = best_model
            
            results.append({
                'Model': optimized_name,
                'Train_RMSE': test_metrics['rmse'],
                'Test_RMSE': test_metrics['rmse'],
                'Train_R2': test_metrics['r2'],
                'Test_R2': test_metrics['r2'],
                'CV_RMSE_Mean': 0,
                'CV_RMSE_Std': 0
            })
            
            print(f"   ✅ Mejor RMSE Test: {test_metrics['rmse']:.3f}")
            print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")
    
    # Seleccionar mejor modelo
    results_df = pd.DataFrame(results)
    best_model_info = results_df.loc[results_df['Test_RMSE'].idxmin()]
    best_model = trained_models[best_model_info['Model']]
    
    print(f"\n🏆 Mejor modelo: {best_model_info['Model']}")
    print(f"📊 RMSE Test: {best_model_info['Test_RMSE']:.3f}")
    print(f"📊 R² Test: {best_model_info['Test_R2']:.3f}")
    
    return results_df, trained_models, best_model, best_model_info

def create_comparison_plot(results_df, save_path='./models/model_comparison.png'):
    """Crea gráfico de comparación simple y claro."""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Ordenar por RMSE
    df_sorted = results_df.sort_values('Test_RMSE')
    
    # Crear gráfico de barras
    bars = ax.barh(df_sorted['Model'], df_sorted['Test_RMSE'], 
                   color='#2E86AB', alpha=0.8)
    
    # Personalizar
    ax.set_xlabel('RMSE (Test) - Menor es mejor', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Modelos - Error de Predicción', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (bar, value) in enumerate(zip(bars, df_sorted['Test_RMSE'])):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', fontweight='bold')
    
    # Resaltar el mejor modelo
    best_bar = bars[0]
    best_bar.set_color('#F18F01')
    best_bar.set_alpha(1.0)
    
    # Añadir anotación para el mejor modelo
    best_model = df_sorted.iloc[0]
    ax.annotate(f'🏆 Mejor: {best_model["Model"]}', 
                xy=(best_bar.get_width(), best_bar.get_y() + best_bar.get_height()/2),
                xytext=(best_bar.get_width() + 0.5, best_bar.get_y() + best_bar.get_height()/2),
                fontweight='bold', color='#F18F01')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"📊 Gráfico guardado en: {save_path}")

def save_results(results_df, trained_models, best_model, preprocessor):
    """Guarda todos los resultados y modelos."""
    print(f"\n💾 Guardando resultados...")
    
    # Crear directorio si no existe
    os.makedirs('./models', exist_ok=True)
    
    # Guardar mejor modelo
    joblib.dump(best_model, './models/best_coffee_quality_model.pkl')
    print(f"   ✅ Mejor modelo guardado: best_coffee_quality_model.pkl")
    
    # Guardar preprocesador
    joblib.dump(preprocessor, './models/preprocessor.pkl')
    print(f"   ✅ Preprocesador guardado: preprocessor.pkl")
    
    # Guardar resultados
    results_df.to_csv('./models/model_results.csv', index=False)
    print(f"   ✅ Resultados guardados: model_results.csv")
    
    # Guardar importancia de features si está disponible
    feature_importance = {}
    for name, model in trained_models.items():
        if hasattr(model, 'feature_importances_'):
            feature_importance[name] = model.feature_importances_
    
    if feature_importance:
        importance_df = pd.DataFrame(feature_importance)
        importance_df.to_csv('./models/feature_importance.csv', index=False)
        print(f"   ✅ Importancia de features guardada: feature_importance.csv")

def main():
    """Función principal que ejecuta todo el proceso."""
    print("🚀 SISTEMA DE ENTRENAMIENTO DE MODELOS DE CALIDAD DE CAFÉ")
    print("=" * 60)
    
    # 1. Cargar y preparar datos
    X_train, X_test, y_train, y_test, preprocessor = load_and_prepare_data()
    if X_train is None:
        return
    
    # 2. Entrenar todos los modelos
    results_df, trained_models, best_model, best_model_info = train_all_models(
        X_train, X_test, y_train, y_test
    )
    
    # 3. Crear gráfico de comparación
    create_comparison_plot(results_df)
    
    # 4. Guardar resultados
    save_results(results_df, trained_models, best_model, preprocessor)
    
    # 5. Mostrar resumen final
    print("\n" + "=" * 60)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("=" * 60)
    
    print(f"\n🥇 MEJOR MODELO: {best_model_info['Model']}")
    print(f"📊 RMSE (Test): {best_model_info['Test_RMSE']:.3f}")
    print(f"📊 R² (Test): {best_model_info['Test_R2']:.3f}")
    
    print(f"\n📈 TOP 3 MODELOS:")
    top_models = results_df.nsmallest(3, 'Test_RMSE')
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        print(f"   {i}. {row['Model']}: RMSE={row['Test_RMSE']:.3f}, R²={row['Test_R2']:.3f}")
    
    print(f"\n💾 ARCHIVOS GUARDADOS:")
    print(f"   - Mejor modelo: ./models/best_coffee_quality_model.pkl")
    print(f"   - Preprocesador: ./models/preprocessor.pkl")
    print(f"   - Resultados: ./models/model_results.csv")
    print(f"   - Importancia: ./models/feature_importance.csv")
    print(f"   - Gráfico: ./models/model_comparison.png")
    
    print(f"\n🔮 PARA USAR EL MODELO:")
    print(f"   >>> import joblib")
    print(f"   >>> model = joblib.load('./models/best_coffee_quality_model.pkl')")
    print(f"   >>> preprocessor = joblib.load('./models/preprocessor.pkl')")
    print(f"   >>> prediction = model.predict(preprocessor.transform(X_new))")
    
    print(f"\n✨ ¡Los modelos están listos para producción!")

if __name__ == "__main__":
    main()
