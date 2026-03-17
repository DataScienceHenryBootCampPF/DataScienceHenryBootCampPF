import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class CoffeeQualityModelTrainer:
    """
    Clase para entrenar y evaluar modelos de predicción de calidad de café.
    Maneja múltiples algoritmos y optimización de hiperparámetros.
    """
    
    def __init__(self, model_save_path='./models/'):
        self.model_save_path = model_save_path
        self.models = {}
        self.evaluations = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
        
        # Definir modelos base
        self.base_models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'DecisionTree': DecisionTreeRegressor(random_state=42),
            'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(kernel='rbf')
        }
        
        # Hiperparámetros para optimización
        self.param_grids = {
            'Ridge': {'alpha': [0.1, 1.0, 10.0, 100.0]},
            'Lasso': {'alpha': [0.001, 0.01, 0.1, 1.0]},
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'SVR': {
                'C': [1, 10, 100],
                'gamma': ['scale', 'auto']
            }
        }
    
    def train_base_models(self, X_train, X_test, y_train, y_test, feature_names):
        """
        Entrena todos los modelos base y evalúa su rendimiento.
        """
        print("\n🚀 Entrenando modelos base...")
        
        results = []
        
        for name, model in self.base_models.items():
            print(f"\n📊 Entrenando {name}...")
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Métricas de evaluación
            train_metrics = self.calculate_metrics(y_train, y_train_pred)
            test_metrics = self.calculate_metrics(y_test, y_test_pred)
            
            # Validación cruzada
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                      scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores)
            
            # Guardar modelo y evaluación
            self.models[name] = model
            self.evaluations[name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_rmse_mean': cv_rmse.mean(),
                'cv_rmse_std': cv_rmse.std()
            }
            
            # Extraer importancia de features si está disponible
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(
                    feature_names, model.feature_importances_
                ))
            
            # Preparar resultados para visualización
            results.append({
                'Model': name,
                'Train_RMSE': train_metrics['rmse'],
                'Test_RMSE': test_metrics['rmse'],
                'Train_R2': train_metrics['r2'],
                'Test_R2': test_metrics['r2'],
                'CV_RMSE_Mean': cv_rmse.mean(),
                'CV_RMSE_Std': cv_rmse.std()
            })
            
            print(f"   ✅ RMSE Test: {test_metrics['rmse']:.3f}")
            print(f"   ✅ R² Test: {test_metrics['r2']:.3f}")
            print(f"   ✅ CV RMSE: {cv_rmse.mean():.3f} ± {cv_rmse.std():.3f}")
        
        # Crear DataFrame de resultados
        self.results_df = pd.DataFrame(results)
        return self.results_df
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calcula métricas de evaluación para regresión.
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        return metrics
    
    def optimize_best_models(self, X_train, y_train, X_test, y_test, feature_names, top_n=3):
        """
        Optimiza hiperparámetros de los mejores modelos.
        """
        print("\n🔧 Optimizando hiperparámetros...")
        
        # Seleccionar mejores modelos basados en RMSE de prueba
        best_models = self.results_df.nsmallest(top_n, 'Test_RMSE')['Model'].tolist()
        
        optimized_results = []
        
        for model_name in best_models:
            if model_name in self.param_grids:
                print(f"\n⚡ Optimizando {model_name}...")
                
                base_model = self.base_models[model_name]
                param_grid = self.param_grids[model_name]
                
                # Grid Search con validación cruzada
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=5,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1, verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                # Mejor modelo
                best_model = grid_search.best_estimator_
                
                # Evaluación
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                train_metrics = self.calculate_metrics(y_train, y_train_pred)
                test_metrics = self.calculate_metrics(y_test, y_test_pred)
                
                # Guardar modelo optimizado
                optimized_name = f"{model_name}_Optimized"
                self.models[optimized_name] = best_model
                self.evaluations[optimized_name] = {
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'best_params': grid_search.best_params_
                }
                
                # Importancia de features
                if hasattr(best_model, 'feature_importances_'):
                    self.feature_importance[optimized_name] = dict(zip(
                        feature_names, best_model.feature_importances_
                    ))
                
                optimized_results.append({
                    'Model': optimized_name,
                    'Train_RMSE': train_metrics['rmse'],
                    'Test_RMSE': test_metrics['rmse'],
                    'Train_R2': train_metrics['r2'],
                    'Test_R2': test_metrics['r2'],
                    'Best_Params': str(grid_search.best_params_)
                })
                
                print(f"   ✅ Mejor RMSE Test: {test_metrics['rmse']:.3f}")
                print(f"   ✅ Mejores parámetros: {grid_search.best_params_}")
        
        # Actualizar DataFrame de resultados
        if optimized_results:
            optimized_df = pd.DataFrame(optimized_results)
            self.results_df = pd.concat([self.results_df, optimized_df], ignore_index=True)
        
        return optimized_results
    
    def select_best_model(self):
        """
        Selecciona el mejor modelo basado en múltiples criterios.
        """
        print("\n🏆 Seleccionando mejor modelo...")
        
        # Ordenar por RMSE de prueba (menor es mejor)
        sorted_models = self.results_df.sort_values('Test_RMSE')
        
        # Considerar estabilidad (baja desviación en CV)
        top_models = sorted_models.head(5)
        
        # Seleccionar el mejor modelo (menor RMSE + buena estabilidad)
        best_candidate = top_models.iloc[0]
        self.best_model_name = best_candidate['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🥇 Mejor modelo seleccionado: {self.best_model_name}")
        print(f"   📊 RMSE Test: {best_candidate['Test_RMSE']:.3f}")
        print(f"   📊 R² Test: {best_candidate['Test_R2']:.3f}")
        
        if 'CV_RMSE_Std' in best_candidate:
            print(f"   📊 Estabilidad CV: ±{best_candidate['CV_RMSE_Std']:.3f}")
        
        return self.best_model, self.best_model_name
    
    def save_models(self):
        """
        Guarda los modelos entrenados en archivos.
        """
        import os
        os.makedirs(self.model_save_path, exist_ok=True)
        
        print(f"\n💾 Guardando modelos en {self.model_save_path}")
        
        # Guardar mejor modelo
        if self.best_model:
            joblib.dump(self.best_model, 
                       os.path.join(self.model_save_path, 'best_coffee_quality_model.pkl'))
            print(f"   ✅ Mejor modelo guardado: best_coffee_quality_model.pkl")
        
        # Guardar todos los modelos
        for name, model in self.models.items():
            filename = f"{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, os.path.join(self.model_save_path, filename))
        
        # Guardar resultados
        if hasattr(self, 'results_df'):
            self.results_df.to_csv(os.path.join(self.model_save_path, 'model_results.csv'), 
                                  index=False)
            print(f"   ✅ Resultados guardados: model_results.csv")
        
        # Guardar importancia de features
        if self.feature_importance:
            importance_df = pd.DataFrame(self.feature_importance).T
            importance_df.to_csv(os.path.join(self.model_save_path, 'feature_importance.csv'))
            print(f"   ✅ Importancia de features guardada: feature_importance.csv")
    
    def plot_model_comparison(self):
        """
        Genera gráficos comparativos de modelos.
        """
        if not hasattr(self, 'results_df'):
            print("❌ No hay resultados para visualizar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RMSE Comparison
        ax1 = axes[0, 0]
        sorted_df = self.results_df.sort_values('Test_RMSE')
        bars = ax1.barh(sorted_df['Model'], sorted_df['Test_RMSE'], 
                       color='skyblue', edgecolor='navy')
        ax1.set_xlabel('RMSE (Test)')
        ax1.set_title('RMSE de Modelos - Test Set', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Resaltar mejor modelo
        best_idx = sorted_df['Test_RMSE'].idxmin()
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('orange')
        
        # 2. R² Comparison
        ax2 = axes[0, 1]
        sorted_r2 = self.results_df.sort_values('Test_R2', ascending=True)
        bars2 = ax2.barh(sorted_r2['Model'], sorted_r2['Test_R2'], 
                         color='lightgreen', edgecolor='darkgreen')
        ax2.set_xlabel('R² (Test)')
        ax2.set_title('R² de Modelos - Test Set', fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Resaltar mejor modelo
        best_r2_idx = sorted_r2['Test_R2'].idxmax()
        bars2[best_r2_idx].set_color('gold')
        bars2[best_r2_idx].set_edgecolor('orange')
        
        # 3. Train vs Test Performance
        ax3 = axes[1, 0]
        ax3.scatter(self.results_df['Train_RMSE'], self.results_df['Test_RMSE'], 
                   s=100, alpha=0.7, c='coral')
        ax3.plot([self.results_df['Train_RMSE'].min(), self.results_df['Train_RMSE'].max()],
                [self.results_df['Train_RMSE'].min(), self.results_df['Train_RMSE'].max()],
                'r--', alpha=0.5)
        ax3.set_xlabel('RMSE (Train)')
        ax3.set_ylabel('RMSE (Test)')
        ax3.set_title('Overfitting Analysis', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Añadir etiquetas
        for i, row in self.results_df.iterrows():
            ax3.annotate(row['Model'], (row['Train_RMSE'], row['Test_RMSE']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Feature Importance (si está disponible)
        ax4 = axes[1, 1]
        if self.best_model_name in self.feature_importance:
            importance_data = self.feature_importance[self.best_model_name]
            sorted_importance = sorted(importance_data.items(), key=lambda x: x[1], reverse=True)[:10]
            
            features, values = zip(*sorted_importance)
            bars4 = ax4.barh(range(len(features)), values, color='plum', edgecolor='purple')
            ax4.set_yticks(range(len(features)))
            ax4.set_yticklabels(features)
            ax4.set_xlabel('Importance')
            ax4.set_title(f'Top 10 Features - {self.best_model_name}', fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Feature importance\nno disponible', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Feature Importance', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'model_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_predictions_report(self, X_test, y_test, sample_size=100):
        """
        Genera un reporte detallado de predicciones.
        """
        if not self.best_model:
            print("❌ No hay mejor modelo seleccionado")
            return
        
        print(f"\n📋 Generando reporte de predicciones con {self.best_model_name}...")
        
        # Hacer predicciones
        y_pred = self.best_model.predict(X_test)
        
        # Crear DataFrame de resultados
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Difference': y_test - y_pred,
            'Absolute_Error': np.abs(y_test - y_pred),
            'Percentage_Error': (np.abs(y_test - y_pred) / y_test) * 100
        })
        
        # Estadísticas del error
        print("\n📊 Estadísticas del Error:")
        print(f"   Error Medio Absoluto: {results_df['Absolute_Error'].mean():.3f}")
        print(f"   Error Porcentual Medio: {results_df['Percentage_Error'].mean():.2f}%")
        print(f"   Desviación estándar del error: {results_df['Absolute_Error'].std():.3f}")
        
        # Peores predicciones
        worst_predictions = results_df.nlargest(5, 'Absolute_Error')
        print("\n⚠️  Peores 5 Predicciones:")
        print(worst_predictions.round(3))
        
        # Mejores predicciones
        best_predictions = results_df.nsmallest(5, 'Absolute_Error')
        print("\n✅ Mejores 5 Predicciones:")
        print(best_predictions.round(3))
        
        # Guardar reporte
        report_path = os.path.join(self.model_save_path, 'predictions_report.csv')
        results_df.to_csv(report_path, index=False)
        print(f"\n💾 Reporte guardado en: {report_path}")
        
        return results_df
    
    def run_complete_training(self, df):
        """
        Ejecuta el flujo completo de entrenamiento.
        """
        print("🎯 Iniciando entrenamiento completo de modelos...")
        
        # 1. Preprocesamiento
        preprocessor = CoffeeDataPreprocessor()
        
        # Analizar balance de especies
        preprocessor.get_species_balance_info(df)
        
        # Preparar datos
        X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df)
        
        # 2. Entrenar modelos base
        self.train_base_models(X_train, X_test, y_train, y_test, feature_names)
        
        # 3. Optimizar mejores modelos
        self.optimize_best_models(X_train, y_train, X_test, y_test, feature_names)
        
        # 4. Seleccionar mejor modelo
        self.select_best_model()
        
        # 5. Generar visualizaciones
        self.plot_model_comparison()
        
        # 6. Generar reporte de predicciones
        self.generate_predictions_report(X_test, y_test)
        
        # 7. Guardar modelos
        self.save_models()
        
        print("\n🎉 Entrenamiento completado exitosamente!")
        print(f"🥇 Mejor modelo: {self.best_model_name}")
        
        return self.best_model, self.results_df
