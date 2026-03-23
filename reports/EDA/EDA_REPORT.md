1. Introducción y Contexto del Proyecto

Este reporte analiza el dataset del Coffee Quality Institute, que contiene evaluaciones técnicas de lotes de café de todo el mundo. El objetivo es desentrañar los factores físicos (altitud, país, especie) y sensoriales (aroma, cuerpo, acidez) que determinan el Total Cup Points (variable objetivo) para predecir la calidad del grano en el mercado de especialidad.

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

B. Metodología de Data Preprocessing

Para transformar un archivo crudo de 1339 registros y 44 columnas en una matriz apta para Machine Learning, se aplicó un pipeline de limpieza bajo un enfoque de integridad estadística:

 - Reducción de Dimensionalidad: Se descartaron columnas con nulos masivos (Lot.Number 79%) o sin valor predictivo (IDs, contactos).
 
 - Tratamiento de la Altitud: Se unificaron unidades (pies a metros) y se eliminaron outliers espaciales (>4000m) que distorsionaban las medias.
 
 - Imputación Jerárquica: Los valores faltantes se completaron usando la Moda (categóricas) y Mediana (numéricas) basadas en la combinación País + Especie, respetando la biología de la planta (Arabica vs. Robusta).

 - Feature Engineering: 
    * Se creó Nombre_Comercial para dar identidad de marketing a cada lote.
    
    * Se eliminaron variables sin varianza (Sweetness, Clean.Cup, Uniformity), ya que en café de especialidad casi siempre son 10/10 y no ayudan a predecir diferencias.
        
4. Hallazgos Clave del Análisis Visual (EDA)

A. El dataset presenta un fuerte sesgo negativo hacia la alta calidad. La media (82.16) y la mediana (82.50) están casi solapadas, con el 75% de los datos por encima de los 81 puntos. En este sentido, el modelo de ML debe ser extremadamente preciso, ya que la diferencia entre un café "Estándar" y uno "Premium" se juega en un rango muy estrecho de puntaje.

![Distribución Puntaje Total](images/Dist._PuntajeTotal.png)

B. Geografía y Calidad: Etiopía ostenta el puntaje promedio más alto (85.48) y el máximo histórico (90.58), consolidándose como el benchmark mundial. Países como El Salvador muestran una producción muy uniforme (baja desviación en altitud), mientras que Kenia y EE. UU. mantienen alta calidad bajo condiciones geográficas muy variadas.

![Puntaje Promedio por País](images/PuntajeProm_Pais.png)

C. El Mito de la Altitud (r = 0.20): Contrario a la creencia popular, la correlación entre altitud y calidad es débil-positiva. Conclusión: Si bien la altura ayuda a la densidad del grano, el procesamiento humano y la variedad botánica pueden compensar la falta de metros. Hay cafés a 1000m que superan técnicamente a lotes cultivados a 2000m.

![Altura vs Calidad](images/Altura_vs_Calidad.png)

D. Análisis Sensorial y de Procesamiento: El método Washed es el más común, pero los métodos artesanales como el Honey están logrando picos de calidad superiores, lo que sugiere un mercado creciente para procesos diferenciados.

![Método de Procesamiento vs Calidad](images/Metodo_Procesamiento_Calidad.png)

E. Atributos como Sweetness y Clean Cup no predicen el puntaje final, sino que actúan como "requisitos mínimos": si no son perfectos, el café sale del circuito de especialidad.

F. Variedades de Altura: Las variedades SL14 y Gesha se posicionan como la gama más alta, cultivándose consistentemente por encima de los 1500m.

G. Tendencia de Rigurosidad: Se observa una caída en los puntajes máximos desde 2015 a 2018. Esto no indica necesariamente granos de menor calidad, sino que podría darse por una mayor exigencia y rigurosidad técnica de los catadores certificados.

![Evolución Puntaje Máximo](images/Evolución_PuntajeMax.png)

Tras la consolidación del dataset, el proyecto avanzará hacia la implementación de inteligencia artificial para resolver dos necesidades de negocio: la predicción de calidad y la personalización de la oferta.