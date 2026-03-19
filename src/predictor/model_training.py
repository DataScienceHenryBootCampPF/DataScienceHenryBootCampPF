"""
Entrenamiento de Modelos de Predicción de Calidad de Café
========================================================

Este módulo se encarga de:
- Cargar datos procesados por feature engineering
- Entrenar múltiples modelos de Machine Learning
- Optimizar hiperparámetros
- Evaluar y comparar modelos
- Guardar resultados y mejores modelos

Author: Data Science Henry Bootcamp
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')

class CoffeeModelTrainer:
    """
    Clase para entrenar y evaluar modelos de predicción de calidad de café.
    
    Esta clase maneja el ciclo completo de entrenamiento:
    - Carga de datos procesados
    - Entrenamiento de múltiples algoritmos
    - Optimización de hiperparámetros
    - Evaluación comparativa
    - Guardado de resultados
    """
    
    def __init__(self, data_path, output_dir):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            data_path (str): Ruta al archivo CSV de datos procesados
            output_dir (str): Directorio donde guardar los resultados
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
        # Asegurar que el directorio de salida exista
        os.makedirs(output_dir, exist_ok=True)
        
        # Mapeo de modelos
        self.model_classes = {
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'Lasso': Lasso,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'SVR': SVR
        }
        
        print(f"🤖 CoffeeModelTrainer inicializado")
        print(f"📁 Datos procesados: {data_path}")
        print(f"📁 Directorio de salida: {output_dir}")
    
    def load_processed_data(self):
        """
        Carga los datos procesados desde el archivo CSV.
        
        Returns:
            dict: Diccionario con datos de entrenamiento y prueba
        """
        try:
            print(f"\n📁 Cargando datos procesados desde: {self.data_path}")
            
            # Cargar datos completos
            df = pd.read_csv(self.data_path)
            print(f"✅ Datos cargados: {df.shape}")
            
            # Separar train/test
            train_df = df[df['dataset'] == 'train'].drop(columns=['dataset'])
            test_df = df[df['dataset'] == 'test'].drop(columns=['dataset'])
            
            # Separar features y target
            X_train = train_df.drop(columns=['Total.Cup.Points'])
            y_train = train_df['Total.Cup.Points']
            
            X_test = test_df.drop(columns=['Total.Cup.Points'])
            y_test = test_df['Total.Cup.Points']
            
            # Guardar nombres de features
            self.feature_names = X_train.columns.tolist()
            
            print(f"📊 Datos de entrenamiento: {X_train.shape}")
            print(f"📊 Datos de prueba: {X_test.shape}")
            print(f"📋 Features: {len(self.feature_names)}")
            
            return {
                'X_train': X_train.values,
                'X_test': X_test.values,
                'y_train': y_train.values,
                'y_test': y_test.values,
                'feature_names': self.feature_names
            }
            
        except FileNotFoundError:
            print(f"❌ Error: No se encontró el archivo {self.data_path}")
            return None
        except Exception as e:
            print(f"❌ Error al cargar datos: {str(e)}")
            return None
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calcula métricas de evaluación para regresión.
        
        Args:
            y_true (array): Valores reales
            y_pred (array): Valores predichos
            
        Returns:
            dict: Diccionario con métricas calculadas
        """
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics
    
    def train_model(self, model_name, model_config, X_train, X_test, y_train, y_test):
        """
        Entrena un modelo específico y calcula métricas.
        
        Args:
            model_name (str): Nombre del modelo
            model_config (dict): Configuración del modelo
            X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba
            
        Returns:
            dict: Resultados del modelo
        """
        print(f"\n🤖 Entrenando: {model_name}")
        start_time = time.time()
        
        # Obtener clase del modelo
        model_class = self.model_classes[model_config['class']]
        
        # Crear instancia con parámetros base
        model_params = {}
        
        # Añadir random_state solo a modelos que lo soportan
        if model_config['class'] in ['DecisionTreeRegressor', 'RandomForestRegressor', 
                                   'GradientBoostingRegressor']:
            model_params['random_state'] = 42
        
        # Crear modelo
        model = model_class(**model_params)
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Hacer predicciones
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calcular métricas
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_test_pred)
        
        # Validación cruzada
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        # Tiempo de entrenamiento
        training_time = time.time() - start_time
        
        # Guardar modelo
        self.models[model_name] = model
        
        # Preparar resultados
        result = {
            'model_name': model_name,
            'model_class': model_config['class'],
            'train_rmse': train_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'train_mape': train_metrics['mape'],
            'test_mape': test_metrics['mape'],
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'training_time': training_time,
            'model': model
        }
        
        # Imprimir resultados
        print(f"   ✅ RMSE Train: {train_metrics['rmse']:.3f}")
        print(f"   ✅ RMSE Test: {test_metrics['rmse']:.3f}")
        print(f"   ✅ R² Test: {test_metrics['r2']:.3f}")
        print(f"   ✅ CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
        print(f"   ⏱️ Tiempo: {training_time:.2f}s")
        
        return result
    
    def optimize_hyperparameters(self, model_name, model_config, X_train, y_train):
        """
        Optimiza hiperparámetros usando GridSearchCV.
        
        Args:
            model_name (str): Nombre del modelo
            model_config (dict): Configuración del modelo
            X_train, y_train: Datos de entrenamiento
            
        Returns:
            dict: Resultados de la optimización
        """
        print(f"\n⚡ Optimizando hiperparámetros: {model_name}")
        start_time = time.time()
        
        # Obtener clase del modelo
        model_class = self.model_classes[model_config['class']]
        
        # Parámetros base
        model_params = {}
        
        # Añadir random_state solo a modelos que lo soportan
        if model_config['class'] in ['DecisionTreeRegressor', 'RandomForestRegressor', 
                                   'GradientBoostingRegressor']:
            model_params['random_state'] = 42
        
        # Crear modelo base
        model = model_class(**model_params)
        
        # Grid Search
        grid_search = GridSearchCV(
            model,
            model_config['params'],
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        # Entrenar Grid Search
        grid_search.fit(X_train, y_train)
        
        # Mejor modelo
        best_model = grid_search.best_estimator_
        
        # Guardar modelo optimizado
        optimized_name = f"{model_name}_Optimized"
        self.models[optimized_name] = best_model
        
        # Tiempo de optimización
        optimization_time = time.time() - start_time
        
        # Preparar resultados
        result = {
            'model_name': optimized_name,
            'model_class': model_config['class'],
            'best_params': grid_search.best_params_,
            'best_score': np.sqrt(-grid_search.best_score_),
            'optimization_time': optimization_time,
            'model': best_model
        }
        
        print(f"   ✅ Mejor RMSE CV: {result['best_score']:.3f}")
        print(f"   ✅ Mejores parámetros: {result['best_params']}")
        print(f"   ⏱️ Tiempo optimización: {optimization_time:.2f}s")
        
        return result
    
    def evaluate_optimized_model(self, optimized_result, X_test, y_test):
        """
        Evalúa el modelo optimizado en datos de prueba.
        
        Args:
            optimized_result (dict): Resultados de la optimización
            X_test, y_test: Datos de prueba
            
        Returns:
            dict: Métricas de evaluación
        """
        model = optimized_result['model']
        y_pred = model.predict(X_test)
        
        metrics = self.calculate_metrics(y_test, y_pred)
        
        evaluation = {
            'model_name': optimized_result['model_name'],
            'test_rmse': metrics['rmse'],
            'test_mae': metrics['mae'],
            'test_r2': metrics['r2'],
            'test_mape': metrics['mape']
        }
        
        print(f"   📊 RMSE Test optimizado: {metrics['rmse']:.3f}")
        print(f"   📊 R² Test optimizado: {metrics['r2']:.3f}")
        
        return evaluation
    
    def create_roc_curve_data(self, X_test, y_test):
        """
        Crea datos para curva ROC (adaptando regresión a clasificación binaria).
        
        Args:
            X_test, y_test: Datos de prueba
            
        Returns:
            dict: Datos para curva ROC
        """
        print(f"\n📈 Creando datos para Curva ROC")
        
        # Convertir a clasificación binaria (café de alta calidad vs baja calidad)
        # Usamos la mediana como umbral
        threshold = np.median(y_test)
        y_test_binary = (y_test > threshold).astype(int)
        
        roc_data = {}
        
        for model_name, model in self.models.items():
            try:
                # Obtener predicciones
                y_pred = model.predict(X_test)
                
                # Convertir predicciones a probabilidades (normalización simple)
                y_pred_proba = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
                
                # Calcular curva ROC
                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                
                roc_data[model_name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': roc_auc
                }
                
                print(f"   ✅ {model_name}: AUC = {roc_auc:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error en {model_name}: {str(e)}")
        
        return roc_data
    
    def train_all_models(self, optimize_top_n=3):
        """
        Entrena todos los modelos configurados.
        
        Args:
            optimize_top_n (int): Número de mejores modelos a optimizar
            
        Returns:
            pd.DataFrame: DataFrame con todos los resultados
        """
        print("🚀 INICIANDO ENTRENAMIENTO DE MODELOS")
        print("="*50)
        
        # Cargar datos
        data_dict = self.load_processed_data()
        if data_dict is None:
            return None
        
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        
        # Configuración de modelos
        models_config = {
            'LinearRegression': {
                'class': 'LinearRegression',
                'params': {}
            },
            'Ridge': {
                'class': 'Ridge',
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'Lasso': {
                'class': 'Lasso', 
                'params': {'alpha': [0.001, 0.01, 0.1]}
            },
            'DecisionTree': {
                'class': 'DecisionTreeRegressor',
                'params': {
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [5, 10, 15]
                }
            },
            'RandomForest': {
                'class': 'RandomForestRegressor',
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [5, 10]
                }
            },
            'GradientBoosting': {
                'class': 'GradientBoostingRegressor',
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'SVR': {
                'class': 'SVR',
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear']
                }
            }
        }
        
        # Entrenar todos los modelos base
        for model_name, model_config in models_config.items():
            result = self.train_model(
                model_name, model_config, X_train, X_test, y_train, y_test
            )
            self.results.append(result)
        
        # Convertir a DataFrame para análisis
        results_df = pd.DataFrame(self.results)
        
        # Seleccionar mejores modelos para optimización
        top_models = results_df.nsmallest(optimize_top_n, 'test_rmse')
        print(f"\n🏆 Top {optimize_top_n} modelos para optimizar:")
        for i, (_, row) in enumerate(top_models.iterrows(), 1):
            print(f"   {i}. {row['model_name']}: RMSE={row['test_rmse']:.3f}")
        
        # Optimizar los mejores modelos
        optimization_results = []
        for _, row in top_models.iterrows():
            model_name = row['model_name']
            model_config = models_config[model_name]
            
            # Optimizar solo si tiene parámetros para optimizar
            if model_config['params']:
                optimized_result = self.optimize_hyperparameters(
                    model_name, model_config, X_train, y_train
                )
                
                # Evaluar modelo optimizado
                evaluation = self.evaluate_optimized_model(optimized_result, X_test, y_test)
                
                # Combinar resultados
                combined_result = {**optimized_result, **evaluation}
                optimization_results.append(combined_result)
                self.results.append(combined_result)
        
        # DataFrame final con todos los resultados
        final_results_df = pd.DataFrame(self.results)
        
        # Seleccionar mejor modelo
        best_idx = final_results_df['test_rmse'].idxmin()
        best_result = final_results_df.loc[best_idx]
        self.best_model = best_result['model']
        self.best_model_name = best_result['model_name']
        
        # Crear datos para curva ROC
        roc_data = self.create_roc_curve_data(X_test, y_test)
        
        print(f"\n🥇 MEJOR MODELO: {self.best_model_name}")
        print(f"📊 RMSE Test: {best_result['test_rmse']:.3f}")
        print(f"📊 R² Test: {best_result['test_r2']:.3f}")
        
        return final_results_df, roc_data
    
    def save_results(self, results_df, roc_data):
        """
        Guarda los resultados y modelos entrenados.
        
        Args:
            results_df (pd.DataFrame): DataFrame con resultados
            roc_data (dict): Datos para curva ROC
        """
        print(f"\n💾 GUARDANDO RESULTADOS")
        print(f"="*30)
        
        # 1. Guardar resultados CSV
        results_path = os.path.join(self.output_dir, 'training_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"✅ Resultados guardados: {results_path}")
        
        # 2. Guardar mejor modelo
        if self.best_model is not None:
            best_model_path = os.path.join(self.output_dir, 'best_model.pkl')
            joblib.dump(self.best_model, best_model_path)
            print(f"✅ Mejor modelo guardado: {best_model_path}")
        
        # 3. Guardar todos los modelos
        models_dir = os.path.join(self.output_dir, 'all_models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            safe_name = model_name.replace(' ', '_').replace('á', 'a').replace('é', 'e')
            model_path = os.path.join(models_dir, f'{safe_name}.pkl')
            joblib.dump(model, model_path)
        
        print(f"✅ {len(self.models)} modelos guardados en {models_dir}")
        
        # 4. Guardar datos ROC
        roc_path = os.path.join(self.output_dir, 'roc_data.pkl')
        joblib.dump(roc_data, roc_path)
        print(f"✅ Datos ROC guardados: {roc_path}")
        
        # 5. Guardar metadatos
        metadata = {
            'best_model_name': self.best_model_name,
            'best_rmse': results_df.loc[results_df['test_rmse'].idxmin(), 'test_rmse'],
            'best_r2': results_df.loc[results_df['test_rmse'].idxmin(), 'test_r2'],
            'total_models': len(results_df),
            'feature_names': self.feature_names,
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_path = os.path.join(self.output_dir, 'training_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"✅ Metadatos guardados: {metadata_path}")
        
        return results_path, roc_path
    
    def create_visualization_plots(self, results_df, roc_data):
        """
        Crea gráficos de visualización para los resultados.
        
        Args:
            results_df (pd.DataFrame): DataFrame con resultados
            roc_data (dict): Datos para curva ROC
        """
        print(f"\n📊 CREANDO GRÁFICOS VISUALES")
        print(f"="*35)
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        
        # 1. Gráfico comparativo de modelos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análisis Comparativo de Modelos - Calidad de Café', fontsize=16, fontweight='bold')
        
        # RMSE Test
        results_sorted_rmse = results_df.sort_values('test_rmse')
        bars1 = ax1.barh(range(len(results_sorted_rmse)), results_sorted_rmse['test_rmse'], 
                         color='#2E86AB', alpha=0.8)
        ax1.set_xlabel('RMSE Test (menor es mejor)', fontweight='bold')
        ax1.set_title('Error Cuadrático Medio', fontweight='bold')
        ax1.set_yticks(range(len(results_sorted_rmse)))
        ax1.set_yticklabels(results_sorted_rmse['model_name'], fontsize=9)
        ax1.grid(axis='x', alpha=0.3)
        
        # Resaltar mejor modelo
        bars1[0].set_color('#F18F01')
        bars1[0].set_alpha(1.0)
        
        # R² Test
        results_sorted_r2 = results_df.sort_values('test_r2', ascending=True)
        bars2 = ax2.barh(range(len(results_sorted_r2)), results_sorted_r2['test_r2'], 
                         color='#A23B72', alpha=0.8)
        ax2.set_xlabel('R² Test (mayor es mejor)', fontweight='bold')
        ax2.set_title('Coeficiente de Determinación', fontweight='bold')
        ax2.set_yticks(range(len(results_sorted_r2)))
        ax2.set_yticklabels(results_sorted_r2['model_name'], fontsize=9)
        ax2.grid(axis='x', alpha=0.3)
        
        # Tiempo de entrenamiento
        results_sorted_time = results_df.sort_values('training_time', ascending=True)
        bars3 = ax3.barh(range(len(results_sorted_time)), results_sorted_time['training_time'], 
                         color='#592E83', alpha=0.8)
        ax3.set_xlabel('Tiempo (segundos)', fontweight='bold')
        ax3.set_title('Tiempo de Entrenamiento', fontweight='bold')
        ax3.set_yticks(range(len(results_sorted_time)))
        ax3.set_yticklabels(results_sorted_time['model_name'], fontsize=9)
        ax3.grid(axis='x', alpha=0.3)
        
        # Curva ROC
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        for model_name, roc_info in roc_data.items():
            ax4.plot(roc_info['fpr'], roc_info['tpr'], 
                    label=f'{model_name} (AUC = {roc_info["auc"]:.3f})')
        
        ax4.set_xlabel('False Positive Rate', fontweight='bold')
        ax4.set_ylabel('True Positive Rate', fontweight='bold')
        ax4.set_title('Curva ROC', fontweight='bold')
        ax4.legend(loc='lower right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar gráfico
        plot_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Gráfico guardado: {plot_path}")
        
        return plot_path
    
    def run_full_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            tuple: (results_df, roc_data, results_path, roc_path)
        """
        print("🚀 INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO")
        print("="*60)
        
        # 1. Entrenar todos los modelos
        results_df, roc_data = self.train_all_models(optimize_top_n=3)
        
        if results_df is None:
            return None, None, None, None
        
        # 2. Guardar resultados
        results_path, roc_path = self.save_results(results_df, roc_data)
        
        # 3. Crear visualizaciones
        plot_path = self.create_visualization_plots(results_df, roc_data)
        
        print(f"\n✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"🥇 Mejor modelo: {self.best_model_name}")
        print(f"📊 RMSE: {results_df.loc[results_df['test_rmse'].idxmin(), 'test_rmse']:.3f}")
        print(f"📊 R²: {results_df.loc[results_df['test_rmse'].idxmin(), 'test_r2']:.3f}")
        
        return results_df, roc_data, results_path, plot_path

if __name__ == "__main__":
    # Ejemplo de uso
    import os
    
    # Obtener el directorio actual del script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    data_path = os.path.join(project_root, "models", "prediction", "processed_data.csv")
    output_dir = os.path.join(project_root, "models", "prediction")
    
    print(f"📁 Ruta corregida de datos: {data_path}")
    print(f"📁 Ruta corregida de salida: {output_dir}")
    
    trainer = CoffeeModelTrainer(data_path, output_dir)
    results_df, roc_data, results_path, plot_path = trainer.run_full_training()
    
    if results_df is not None:
        print(f"\n🎉 Entrenamiento completado!")
        print(f"📁 Resultados guardados en: {results_path}")
        print(f"📊 Gráficos guardados en: {plot_path}")
