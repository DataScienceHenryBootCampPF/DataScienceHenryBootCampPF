import pandas as pd
import numpy as np
import joblib
import os
import time
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

# Modelos
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

# 1. Definición de carpetas
MODELS_DIR = './models/prediction'
METRICS_DIR = './metrics'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# 2. Cargar datos
df = pd.read_csv('./data/processed/coffee_data_for_training.csv')

# --- LIMPIEZA DE OUTLIERS REFINADA ---
# Filtramos para enfocarnos en el rango donde vive el 95% de los datos reales
df = df[(df['Total.Cup.Points'] > 70) & (df['Total.Cup.Points'] < 92)]

# 3. Definir X e y
target = 'Total.Cup.Points'
features_cat = ['Country.of.Origin', 'Region', 'Variety', 'Processing.Method', 'Color', 'categoria_altitud']
features_num = ['Moisture', 'Category.One.Defects', 'Category.Two.Defects', 'altitude_mean_meters']

X = df[features_cat + features_num]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        # handle_unknown='ignore' es vital para que la Demo no rompa con datos nuevos
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat),
        ('num', StandardScaler(), features_num)
    ])

# 5. Modelos (Hiperparámetros ajustados para mayor capacidad de aprendizaje)
rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=2, random_state=42)
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)

# Directorio para logs de CatBoost
cat_logs_dir = os.path.join(MODELS_DIR, 'catboost_info')
os.makedirs(cat_logs_dir, exist_ok=True)

models = {
    'LinearRegression': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=1.0, epsilon=0.1),
    'RandomForest': rf,
    'GradientBoosting': gb,
    'XGBoost': xgb,
    'CatBoost': CatBoostRegressor(iterations=300, learning_rate=0.05, verbose=0, 
                                  random_state=42, train_dir=cat_logs_dir),
    'Voting_Ensemble': VotingRegressor(estimators=[('rf', rf), ('gb', gb), ('xgb', xgb)])
}

# 6. Torneo de Modelos
results = []

for name, model in models.items():
    start_time = time.time()
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Cálculo de métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    duration = time.time() - start_time
    
    results.append({
        'Model': name, 
        'MAE': mae, 
        'RMSE': rmse, 
        'R2': r2, 
        'MAPE': mape,
        'Time_sec': duration
    })
    
    # Guardar cada modelo individualmente
    joblib.dump(pipeline, os.path.join(MODELS_DIR, f'coffee_model_{name}.pkl'))
    print(f"✅ {name:<18} | MAE: {mae:.3f} | R2: {r2:.3f}")

# 7. Guardar y Reportar Resultados
df_results = pd.DataFrame(results).sort_values(by='MAE')
df_results.to_csv(os.path.join(METRICS_DIR, 'model_comparison_ranking.csv'), index=False)

# Guardar resumen del mejor modelo
with open(os.path.join(METRICS_DIR, 'best_model_summary.txt'), 'w') as f:
    f.write("RESUMEN DEL MEJOR MODELO OPTIMIZADO\n")
    f.write("="*40 + "\n")
    f.write(str(df_results.iloc[0]))

print(f"\n🏆 Ganador por MAE: {df_results.iloc[0]['Model']}")