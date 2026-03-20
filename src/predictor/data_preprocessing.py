import pandas as pd
import json

def procesar_dataset_entrenamiento(path_input, path_output):
    df = pd.read_csv(path_input)
    df_model = df[df['Species'] == 'Arabica'].copy()
    
    # Reducimos el ruido: eliminamos lo que esté MUY por fuera del rango normal
    # Basado en tu describe(), el 95% de los datos está entre 75 y 90
    df_model = df_model[(df_model['Total.Cup.Points'] > 70) & (df_model['Total.Cup.Points'] < 92)]
    
    cols_categoricas = ['Country.of.Origin', 'Region', 'Variety', 'Processing.Method', 'Color']
    umbral = 10
    dict_categorias_validas = {}

    for col in cols_categoricas:
        df_model[col] = df_model[col].astype(str).str.strip()
        counts = df_model[col].value_counts()
        
        # Guardamos las que NO son Other para la Demo
        validas = counts[counts >= umbral].index.tolist()
        dict_categorias_validas[col] = validas
        
        mask_minoritarios = ~df_model[col].isin(validas)
        mask_already_other = df_model[col].str.lower() == 'other'
        
        df_model.loc[mask_minoritarios | mask_already_other, col] = 'Other'

    # Guardamos el diccionario para que la Demo sepa qué es "Other"
    with open('./models/prediction/categorias_validas.json', 'w') as f:
        json.dump(dict_categorias_validas, f)

    df_model.to_csv(path_output, index=False)

# Ejecución
procesar_dataset_entrenamiento(
    './data/processed/coffee_data_cleaned_final.csv', 
    './data/processed/coffee_data_for_training.csv'
)