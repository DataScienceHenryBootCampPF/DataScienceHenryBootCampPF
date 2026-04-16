Plan de Validación de Modelos

Este documento detalla la estrategia científica y técnica utilizada para garantizar que los modelos de predicción de calidad de café (puntuación de taza) sean robustos, generalizables y libres de sesgos.

1. Estrategia de Curación y Filtrado de Datos

Antes del entrenamiento, el pipeline aplica reglas de negocio para asegurar la calidad de la muestra:
 - Segmentación por Especie: Se filtran únicamente registros de la especie Arabica, garantizando homogeneidad en las variables biofísicas.
 - Limpieza de Outliers de Negocio: Se acota el target (Total.Cup.Points) al rango [70, 92]. Esto elimina ruidos estadísticos de cafés con defectos extremos o puntuaciones atípicas que no representan el mercado de café de especialidad.
 - Reducción de Cardinalidad: Se aplica un umbral de frecuencia (n ≥ 10) para variables categóricas (País, Variedad, Región). Las categorías con baja representatividad se agrupan en la etiqueta 'Other', evitando que el modelo aprenda ruidos de muestras aisladas (Overfitting).

2. Metodología de Partición y Entrenamiento (Hold-out)

Para validar la capacidad de generalización, se implementa una división de datos estricta:

 - Training Set (80%): Utilizado para el ajuste de pesos y el aprendizaje de patrones.
 - Test Set (20%): Un conjunto de datos "ciego" que el modelo nunca ve durante el entrenamiento. Se utiliza para generar las métricas finales de rendimiento.
 - Reproducibilidad: Se utiliza una semilla aleatoria fija (random_state=42) para asegurar que el experimento sea consistente en cualquier entorno (Local, GitHub Actions o Producción).

3. Validación Cruzada (K-Fold Cross-Validation)

Para confirmar la estabilidad del modelo elegido (SVR, XGBoost o Ensemble), el plan incluye una Validación Cruzada con K=5:
 - El set de entrenamiento se divide en 5 pliegues.
 - Se calcula el MAE (Mean Absolute Error) promedio y su desviación estándar.
 - Criterio de Aceptación: Un modelo es validado solo si la varianza del error entre pliegues es baja, lo que indica que el modelo no depende de una partición específica de los datos.

4. Definición y Justificación de Métricas Clave

El éxito del modelo se mide a través de un enfoque multimetral:
 - Métrica principal de decisión: MAE.	Indica cuántos "puntos de taza" le pifia el modelo en promedio. Es fácil de comunicar a los catadores.
 - RMSE:	Penaliza errores de gran magnitud.	Nos alerta si el modelo comete errores graves en cafés específicos, lo cual podría arruinar una transacción.
 - R² o Coeficiente de determinación.	Explica qué porcentaje de la calidad del café es capturado por las variables (Altitud, Humedad, etc.).
 - MAPE	o Error porcentual absoluto medio.	Permite entender el error en términos relativos respecto al puntaje total.

5. Pipeline de Validación Automatizado (MLOps)

La validación no es un proceso estático, sino que está integrada en el ciclo de vida del software:
 - MLflow Tracking: Cada entrenamiento registra automáticamente parámetros y métricas. Si un nuevo modelo no supera el MAE del anterior, se descarta.
 - GitHub Actions CI: El archivo .github/workflows/main.yml garantiza que cualquier cambio en el código pase por el proceso de validación completo en un entorno limpio antes de ser desplegado.
 - Consistencia de Inferencia: El uso de categorias_validas.json asegura que la lógica de validación de datos sea idéntica tanto en el entrenamiento como en la App (Streamlit).
________________________________________
🏆 Justificación del Modelo Ganador

El modelo seleccionado como finalista es aquel que logra el equilibrio óptimo entre un MAE bajo y una complejidad computacional moderada, priorizando siempre la estabilidad demostrada en la fase de validación cruzada.
