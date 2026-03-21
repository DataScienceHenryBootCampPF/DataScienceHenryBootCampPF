# test_recommendation_system.py

import sys
import os

# Asegurar que el src esté en el path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from src.predictor.recommendation_system import HybridCoffeeRecommendationSystem
except ImportError as e:
    print(f"❌ Error importando el sistema: {e}")
    print("Asegúrate de que estás en el directorio raíz del proyecto.")
    sys.exit(1)

def test_basic():
    """Test básico del sistema"""
    print("🚀 INICIANDO TESTS DEL SISTEMA DE RECOMENDACIÓN")
    print("="*50)

    print("\n[TEST 1] Inicialización...")
    try:
        sistema = HybridCoffeeRecommendationSystem()
        print("✅ Sistema inicializado correctamente")
    except Exception as e:
        print(f"❌ Falló la inicialización: {e}")
        return
    
    print("\n[TEST 2] Recomendación con 1 variable (Flavor=8.3)...")
    try:
        resultado = sistema.recomendar(Flavor=8.3)
        if len(resultado) == 10:
            print(f"✅ Retorna 10 resultados: {len(resultado)}")
        else:
            print(f"❌ Retorna {len(resultado)} resultados (esperaba 10)")
    except Exception as e:
        print(f"❌ Error en recomendación simple: {e}")
    
    print("\n[TEST 3] Recomendación con 2 variables (Flavor=8.3, Aftertaste=8.1)...")
    try:
        resultado = sistema.recomendar(Flavor=8.3, Aftertaste=8.1)
        if len(resultado) == 10:
            print("✅ Funciona con múltiples variables")
        else:
            print(f"❌ Retorna {len(resultado)} resultados")
    except Exception as e:
        print(f"❌ Error en recomendación múltiple: {e}")
    
    print("\n[TEST 4] Validación de ordenamiento por Calidad...")
    try:
        scores = resultado['Total.Cup.Points'].values
        # Verificar orden descendente (o igual)
        is_sorted = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
        
        if is_sorted:
            print("✅ Correctamente ordenado por Total.Cup.Points (DESC)")
            print(f"   Scores: {scores}")
        else:
            print("❌ No está ordenado por puntuación descendente")
            print(f"   Scores: {scores}")
    except Exception as e:
        print(f"❌ Error validando ordenamiento: {e}")
    
    print("\n[TEST 5] Validación de similitud...")
    try:
        if 'similarity_score' in resultado.columns:
            sim_scores = resultado['similarity_score'].values
            valid_range = all(0 <= s <= 1.0001 for s in sim_scores) # 1.0001 por error flotante
            if valid_range:
                print("✅ Similarity scores válidos (0-1)")
                print(f"   Top Similarity: {sim_scores[0]:.4f}")
            else:
                print(f"❌ Scores fuera de rango: {sim_scores}")
        else:
            print("❌ Columna 'similarity_score' no encontrada")
    except Exception as e:
        print(f"❌ Error validando similitud: {e}")
    
    print("\n[TEST 6] Filtro por especie (Arabica)...")
    try:
        resultado_arabica = sistema.recomendar(Flavor=8.3, species='Arabica')
        unique_species = resultado_arabica['Species'].unique()
        if len(unique_species) == 1 and unique_species[0] == 'Arabica':
            print("✅ Filtro por especie funciona (Solo Arabica)")
        else:
            print(f"❌ Filtro falló. Especies encontradas: {unique_species}")
    except Exception as e:
        print(f"❌ Error probando filtro: {e}")

    print("\n[TEST 7] Manejo de entradas inválidas...")
    try:
        res_invalid = sistema.recomendar() # Sin argumentos
        if res_invalid is None:
            print("✅ Correctamente rechazó entrada vacía")
        else:
            print("❌ Debería retornar None para entrada vacía")
    except Exception as e:
        print(f"❌ Error en test de validación: {e}")
    
    print("\n✅ TODOS LOS TESTS COMPLETADOS")

if __name__ == "__main__":
    test_basic()
