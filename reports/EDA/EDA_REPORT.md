1. Introducción y Contexto del Proyecto

Este reporte analiza el dataset de Coffee Quality Institute, que contiene evaluaciones sensoriales y físicas de lotes de café de todo el mundo. El objetivo principal es entender qué factores determinan el Total Cup Points (nuestra variable objetivo) para predecir o clasificar la calidad del grano.

2. Diccionario de Datos
DICCIONARIO DE DATOS: COFFE QUALITY DATASET

A. Identidad y Trazabilidad

 - Unnamed: 0: Índice numérico residual del archivo original; no tiene valor analítico.
 - Species: Tipo de especie botánica (Arabica o Robusta).
 - Owner / Owner.1: Persona o entidad dueña del lote de café.
 - Country.of.Origin: País de procedencia del grano.
 - Farm.Name: Nombre de la finca productora.
 - Lot.Number: Código de identificación del lote específico.
 - Mill: Nombre del beneficio o planta donde se procesó el café.
 - ICO.Number: Registro internacional ante la International Coffee Organization.
 - Company: Empresa exportadora o comercializadora.
 - Region: Zona geográfica específica dentro del país de origen.
 - Producer: Nombre del productor o caficultor.
 - In.Country.Partner: Organización que coordina la certificación en el país origen (ej. Almacafé).

B. Logística y Certificación

 - Number.of.Bags: Cantidad total de sacos en el lote.
 - Bag.Weight: Peso de cada saco individual.
 - Harvest.Year: Periodo o año de la cosecha.
 - Grading.Date: Fecha en la que se realizó la evaluación técnica.
 - Expiration: Fecha límite de validez de la certificación.
 - Certification.Body: Entidad encargada de emitir el certificado de calidad.
 - Certification.Address / Contact: Datos de ubicación y contacto de la certificadora.

C. Atributos Sensoriales

 - Aroma: Fragancia del café en seco y su olor al contacto con agua caliente.
 - Flavor: Intensidad y calidad del sabor percibido.
 - Aftertaste: Calidad del sabor remanente tras la degustación.
 - Acidity: Nivel de brillantez y viveza del grano.
 - Body: Textura y sensación de peso en el paladar.
 - Balance: Equilibrio entre aroma, sabor, acidez y cuerpo.
 - Uniformity: Consistencia de sabor entre las distintas tazas de la muestra.
 - Clean.Cup: Ausencia de sabores defectuosos o "sucios".
 - Sweetness: Presencia de dulzor natural en el grano.
 - Cupper.Points: Calificación global otorgada por el catador.
 - Total.Cup.Points: Variable objetivo (Target). Puntaje final sobre 100 puntos.

D. Atributos Físicos y Defectos

 - Moisture: Grado de humedad del grano verde.
 - Color: Tono visual del café (Verde, Azulado, etc.).
 - Category.One.Defects: Conteo de defectos primarios (los más graves).
 - Category.Two.Defects: Conteo de defectos secundarios (leves).
 - Quakers: Granos que no se tuestan bien por falta de madurez.

F. Altitud

 - Altitude: Texto original con la información de altura (sin procesar).
 - unit_of_measurement: Unidad utilizada en el registro original (ft o m).
 - altitude_low_meters: Valor mínimo del rango de altitud en metros.
 - altitude_high_meters: Valor máximo del rango de altitud en metros.
 - altitude_mean_meters: Promedio de altitud calculado y normalizado

3. Data Preprocessing
 
La etapa de preprocesamiento consistió en la transformación de un archivo con errores, nulos y unidades mezcladas en una matriz numérica limpia lista para que los modelos de ML puedan aprender. Para transformar el dataset original en un conjunto de datos apto para el modelado, se aplicaron las siguientes etapas:

A. Vista inicial del Dataset 

Se identifico el número de registros y columnas (1339 y 44 respectivamente). Además del analisis estadistico por columna se detectó lo siguiente:

 - Inconsistencia en Altitud: Presencia de outliers extremos (190km+) que distorsionan la media. Requiere tratamiento por mediana o filtrado físico.
 - Registros Truncados: Valores mínimos de 0.0 en variables sensoriales indican datos faltantes o errores de carga que deben ser removidos.
 - Baja Variabilidad: Atributos como 'Sweetness' y 'Clean.Cup' presentan una concentración excesiva en el valor máximo (10.0), aportando poco valor predictivo.

Otros puntos importantes:

 - No se detectaron registros duplicados
 - Se identificó un fuerte desbalance de clases entre Arabica (n=1311) y Robusta (n=28), lo que justificó un tratamiento diferenciado de variables como la altitud y el método de procesamiento durante la etapa de limpieza, evitando que la mayoría (Arabica) diluya las particularidades de la minoría (Robusta).
 - El análisis de nulos reveló una criticidad extrema en Lot.Number (79% de faltantes), justificando su descarte inmediato. Para el resto de las variables con nulos moderados (12% al 20%), como la altitud y variedades, se optó por una estrategia de imputación jerárquica para preservar la integridad del volumen de datos (1339 muestras) sin introducir sesgos significativos.

 --- Columnas con Valores Faltantes ---
                      Cantidad  Porcentaje (%)
Lot.Number                1063           79.39
Farm.Name                  359           26.81
Mill                       318           23.75
Color                      270           20.16
Producer                   232           17.33
altitude_high_meters       230           17.18
altitude_low_meters        230           17.18
altitude_mean_meters       230           17.18
Altitude                   226           16.88
Variety                    226           16.88
Company                    209           15.61
Processing.Method          170           12.70
ICO.Number                 159           11.87
Region                      59            4.41
Harvest.Year                47            3.51
Owner.1                      7            0.52
Owner                        7            0.52
Country.of.Origin            1            0.07
Quakers                      1            0.07

B. Limpieza y normalización de la estructura

Mediante el archivo "cleaning.py" se eliminaron casi 20 columnas que no tenían información relevante (IDs, fechas de expiración, contactos, etc.), reduciendo la dimensionalidad. Así mismo, se eliminaron registros de Total.Cup.Points (la variable objetivo) == 0 ya que arruinarian cualquier tipo de entrenamiento.

La imputación inteligente se realizó por grupos:

 - Para el Color o el Método de Procesamiento, no usas la moda de todo el dataset. El código mira la moda por País y por Especie.
 - Si falta la altitud, primero busca la mediana del país/especie. Si aun así no la encuentra, usa la mediana global de la especie.
 - Se detectaron unidades de medidas de altura sobre el nivel del mar diferentes (m y ft), con lo cual la normalización se realizó modificando los valores que estaban en pies (ft) para pasarlos a metros, multiplicando este valor por 0.3048

Por ultimo se eliminaron 6 registros con altitudes físicamente imposibles (>4000m) que representaban errores de carga y distorsionaban el análisis estadístico.


4. Análisis Exploratorio de Datos

En el archivo EDA se verifico que los datos esten normalizados correctamente mediante 3 tipos de analisis de variables: Univariado, Bivariado y Multivariado.

1. Distribución del Target (Histograma): Mostrar cómo se distribuye el Total.Cup.Points (que en tu tabla vemos que va de 59 a 90, con mediana de 82.5). Esto prueba que tenés datos variados para entrenar.

2. Heatmap de Correlación: Es obligatoria. Muestra qué variables (Aroma, Flavor, Aftertaste, Altitud) están más ligadas al puntaje. Con los datos limpios, vas a ver que la altitud ahora sí "pesa" en la relación.

3. Scatter Plot (Altitud vs Total Cup Points): Para mostrar la tendencia. Con 1,332 datos, verás una nube densa que sube ligeramente a medida que aumenta la altura.

4. Scatter Plot (Humedad vs Total Cup Points): Para mostrar la tendencia.

5. Graficos descriptivos adicionales







2. ¿Por qué se producen esos errores? (Los 3 culpables)
Ese 27% de veces que el modelo falla, suele ser por estas razones:

Variables "Invisibles": Dos cafés de la misma región, misma variedad y misma altitud pueden tener puntajes distintos porque uno se secó al sol 2 días más que el otro. Como esa info no está en tu Excel, el modelo les asigna el mismo puntaje a ambos, y a uno le erra.

El factor "Catador": La diferencia entre 82.3 y 82.7 es subjetiva. Un catador puede estar de buen humor y poner un 83, y otro un 82. Tu modelo intenta buscar una lógica matemática donde a veces hay una opinión humana.

Outliers Geográficos: Alguna finca en una zona baja que, por un microclima especial, sacó un puntaje altísimo. El modelo, al ver que la altitud es baja, va a predecir un puntaje bajo. Ese punto va a aparecer muy lejos de la línea de perfección en tu gráfico.

El error del modelo es marginal; se concentra en la zona de transición de la mediana. El modelo no confunde un café excelente con uno malo, sino que la incertidumbre aparece en la frontera donde la diferencia sensorial es mínima para los datos disponibles."


En la industria del café (SCA), la diferencia entre un 82 y un 84 no es solo un número; es un salto de precio, de mercado y de prestigio. Por eso, que tu modelo tenga un MAE de 1.4 es a la vez un logro y una limitación que tenés que saber "vender".

1. La "Lupa" del Especialista
Si un termómetro tiene un error de 1 grado, para saber si tenés fiebre (37°C vs 38°C) es un problema enorme. En el café pasa lo mismo.

Para que tu Demo no parezca "plana" o "tibia", tenés que explicar que el modelo está operando en la zona más difícil de la curva.

Casi todos los cafés del dataset son "buenos".

Diferenciar "lo bueno" de "lo muy bueno" requiere una precisión quirúrgica que, sin datos de laboratorio (química del suelo, grados Brix del fruto), es casi imposible de obtener.