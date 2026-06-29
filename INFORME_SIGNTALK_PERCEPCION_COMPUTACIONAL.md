# UNIVERSIDAD PRIVADA ANTENOR ORREGO

**FACULTAD DE INGENIERIA**

**ESCUELA PROFESIONAL DE INGENIERIA DE SISTEMAS E INTELIGENCIA ARTIFICIAL**

---

# PROYECTO DE SISTEMA DE PERCEPCION COMPUTACIONAL PARA RECONOCIMIENTO DE SENAS MEDICAS "SIGNTALK"

**CURSO:** PERCEPCION COMPUTACIONAL

**SEMESTRE:** 2026

**PROFESOR:** Por completar

**INTEGRANTES:** Por completar

---

## INDICE

1. RESUMEN EJECUTIVO
2. INTRODUCCION
   - 2.1. Contexto del Problema
   - 2.2. Objetivos
   - 2.3. Hipotesis
3. ADQUISICION Y PREPROCESAMIENTO DE DATOS
   - 3.1. Fuente de Datos
   - 3.2. Preprocesamiento
   - 3.3. Herramientas Utilizadas
4. METODOLOGIA: DESARROLLO DEL MODELO DE PERCEPCION
   - 4.1. Modelo Base (Baseline Clasico)
   - 4.2. Arquitectura del Modelo Avanzado
   - 4.3. Proceso de Entrenamiento y Creacion del Modelo Final
5. RESULTADOS DEL MODELO
   - 5.1. Metricas de Rendimiento
   - 5.2. Visualizaciones
6. ARQUITECTURA DE DATOS Y PIPELINE A ESCALA
   - 6.1. Tipo de Arquitectura
   - 6.2. Diagrama de Arquitectura del Sistema
   - 6.3. Implementacion y Tecnologias
7. DESPLIEGUE Y OPERACIONALIZACION (MLOps)
   - 7.1. Contenerizacion y Servicio de Inferencia
   - 7.2. Seguimiento de Experimentos
   - 7.3. Gobernanza y Registro del Modelo
8. DISCUSION Y LIMITACIONES
   - 8.1. Desafios Tecnicos
   - 8.2. Etica, Sesgos y Sostenibilidad
9. CONCLUSIONES
   - 9.1. Logros Alcanzados
   - 9.2. Mejoras Futuras
10. ANEXOS TECNICOS
11. REFERENCIAS

---

## 1. RESUMEN EJECUTIVO

El presente proyecto desarrolla un sistema de percepcion computacional orientado al reconocimiento de senas medicas como apoyo a la comunicacion entre personas con discapacidad auditiva y entornos de atencion en salud. La solucion implementada, denominada **SignTalk**, utiliza **MediaPipe Holistic** para capturar keypoints corporales desde camara web, un pipeline de preprocesamiento para normalizar secuencias temporales y una arquitectura **Bidirectional LSTM (Bi-LSTM)** para clasificar senas aisladas. Adicionalmente, el repositorio ya incorpora una ruta metodologica clasica basada en imagen mediante **OpenCV**, con captura visual, escala de grises, deteccion de bordes con **Canny**, extraccion de contornos y un baseline visual con `LinearSVC`. Hasta el estado actual del proyecto se ha implementado un flujo funcional de captura, limpieza, preprocesamiento, entrenamiento, comparacion con baseline clasico, inferencia web local, formulacion inicial de oraciones y lectura en voz mediante `speechSynthesis`. La corrida validada mas reciente del modelo sobre el dataset propio de 14 clases alcanzo **99.36% de val_accuracy** y **100.00% de test_accuracy** en un entorno controlado, con secuencias de **60 frames** y **855 features** por muestra. Sin embargo, el proyecto aun no se encuentra completo respecto al alcance deseado por el silabo: todavia no incorpora un pipeline de procesamiento distribuido con Spark, no cuenta con MLOps formal y no realiza reconocimiento continuo de secuencias largas mas alla de senas aisladas confirmadas. Por ello, el estado actual debe entenderse como un prototipo academico funcional con salida comunicacional local, sobre el cual aun deben construirse los componentes de escalabilidad y madurez operacional.

---

## 2. INTRODUCCION

### 2.1. Contexto del Problema

En el Peru, segun reportes estadisticos del Instituto Nacional de Estadistica e Informatica (INEI), una parte importante de la poblacion presenta limitaciones auditivas permanentes, lo que impacta directamente en su acceso a servicios esenciales, incluyendo la atencion medica. En contextos clinicos, la comunicacion entre paciente y personal de salud suele depender del lenguaje oral o escrito, medios que no siempre son adecuados para personas usuarias de lengua de senas. Esta barrera dificulta que los pacientes expresen sintomas, comprendan indicaciones y participen plenamente en la consulta.

La lengua de senas constituye un sistema linguistico visoespacial con estructura propia, y en el ambito de la salud incorpora ademas un vocabulario especializado relacionado con dolor, malestares, sintomas y necesidades de atencion. En este escenario, los sistemas de percepcion computacional representan una alternativa tecnologica relevante para asistir la interpretacion automatica de senas, especialmente cuando se busca trabajar con datos visuales estructurados y preservando la privacidad del usuario.

El proyecto **SignTalk** aborda esta problematica mediante un enfoque dual. Por un lado, emplea reconocimiento de secuencias de keypoints corporales obtenidos con vision por computadora, lo que permite una representacion mas compacta y robusta frente a variaciones de iluminacion, fondo y vestimenta, y al mismo tiempo reduce la exposicion de informacion visual sensible. Por otro lado, incorpora una ruta clasica basada en imagen con **OpenCV**, orientada a trabajar directamente sobre pixeles mediante escala de grises, bordes y contornos, con fines metodologicos y comparativos. En su estado actual, el sistema se enfoca en un prototipo academico funcional para reconocimiento de senas medicas, formulacion inicial de oraciones y reproduccion local por voz.

### 2.2. Objetivos

En este proyecto se plantean objetivos claros, medibles y alineados con el desarrollo de un sistema de percepcion computacional aplicado al reconocimiento de senas medicas. Los objetivos se organizan en tres dimensiones: modelo, sistema y aplicacion practica.

- **Objetivo de Modelo:** Entrenar y evaluar un modelo de clasificacion de senas medicas basado en Bi-LSTM que alcance una precision superior al 95% en el conjunto de prueba del dataset propio que, ademas, supere el desempeno del baseline clasico implementado con tecnicas tradicionales de aprendizaje automatico.
- **Objetivo de Sistema:** Disenar e implementar un pipeline de datos para la captura, almacenamiento, preprocesamiento, entrenamiento y evaluacion de secuencias de keypoints, contemplando como extension necesaria la ingesta masiva por lotes con Apache Spark y la trazabilidad experimental mediante herramientas de MLOps.
- **Objetivo Practico:** Desarrollar un prototipo funcional de asistencia a la comunicacion clinica que permita capturar senas desde camara, reconocer palabras medicas en tiempo real, generar una primera formulacion de oraciones y reproducirlas localmente por sintesis de voz, dejando como siguiente etapa la mejora hacia reconocimiento continuo y oraciones mas naturales.

### 2.3. Hipotesis

La utilizacion de una arquitectura **Bi-LSTM** sobre secuencias temporales de keypoints corporales, previamente normalizadas espacial y temporalmente y enriquecidas con terminos cinematicos (velocidad y aceleracion), permite reconocer senas medicas aisladas de un dataset propio con un desempeno suficiente para construir la base de un sistema de asistencia comunicacional. Asimismo, se plantea que, al complementar este reconocimiento con formulacion de oraciones y sintesis de voz local, y posteriormente con reconocimiento continuo de secuencias mas largas, el proyecto podra evolucionar hacia una herramienta mas util para interacciones clinicas reales.

---

## 3. ADQUISICION Y PREPROCESAMIENTO DE DATOS

### 3.1. Fuente de Datos

Para el desarrollo de **SignTalk** se construyo un dataset propio mediante una herramienta web de captura implementada en el archivo `capturar.html`. Esta interfaz emplea **MediaPipe Holistic** para detectar landmarks corporales en tiempo real desde la camara del navegador y registrar secuencias asociadas a cada sena.

Cada frame capturado se representa mediante **95 keypoints tridimensionales** distribuidos de la siguiente forma:

- **Pose corporal:** 33 keypoints = 99 caracteristicas
- **Mano izquierda:** 21 keypoints = 63 caracteristicas
- **Mano derecha:** 21 keypoints = 63 caracteristicas
- **Rostro:** 20 keypoints seleccionados = 60 caracteristicas

En total, cada frame queda representado por un vector de **285 features**. Las secuencias completas se almacenan inicialmente en formato JSON en `data_set/signtalk_dataset_unificado.json`.

En la version actual del proyecto, el entrenamiento mas reciente utiliza **14 clases**:

- ARDOR
- BIEN
- DOLOR
- DORMIR
- ESPALDA
- ESTAR-BIEN
- ESTOMAGO
- GARGANTA
- GRIPE
- LO SIENTO
- MUCHO
- NAUSEAS
- PREGUNTAR
- SUDAR

La herramienta de captura contempla dos modos de trabajo:

- **Modo Batch:** captura automatica de un conjunto predefinido de senas.
- **Modo Manual:** registro flexible de clases y repeticiones definidas por el usuario.

Como respuesta a la exigencia metodologica de trabajar directamente sobre imagen, tambien se implemento una segunda herramienta de captura en Python llamada `capturar_cv.py`. Esta utiliza **OpenCV** para registrar imagenes RGB desde webcam y construir un dataset visual alternativo en carpetas por clase (`data_set_cv/`). Este dataset no reemplaza al de keypoints, sino que sirve como base para un baseline clasico construido desde pixeles.

### 3.2. Preprocesamiento

El pipeline de preprocesamiento esta implementado en `preprocessing.py` y transforma el JSON bruto en tensores listos para entrenamiento. Hasta el estado actual del proyecto, los pasos implementados son los siguientes:

1. **Carga de datos:** lectura del dataset fuente `data_set/signtalk_dataset_unificado.json`.
2. **Tratamiento de valores faltantes:** interpolacion lineal por secuencia si el porcentaje de frames invalidos no supera el 30%; en caso contrario, la secuencia se descarta.
3. **Normalizacion espacial:** centrado respecto al keypoint de referencia de la nariz y escalado por la distancia maxima de la secuencia.
4. **Normalizacion temporal:** remuestreo de todas las secuencias a una longitud fija de **60 frames**.
5. **Data augmentation:** generacion de variantes sinteticas por secuencia, incluyendo volteo horizontal, ruido gaussiano, escalado aleatorio y desplazamiento temporal.
6. **Ingenieria de caracteristicas temporales:** concatenacion de posicion, velocidad y aceleracion para pasar de **285** a **855 features** por frame.
7. **Division del dataset:** particion estratificada en entrenamiento, validacion y prueba.

En la corrida validada mas reciente, este flujo produjo un dataset final de **1562 muestras** con la siguiente distribucion:

- **Train:** 1249 muestras
- **Validation:** 156 muestras
- **Test:** 157 muestras

Los artefactos generados por esta etapa son:

- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`
- `label_map.json`

Adicionalmente, para responder a la observacion metodologica sobre percepcion computacional clasica, el repositorio incorpora un pipeline visual complementario implementado en `baseline_cv.py`. Este flujo trabaja directamente sobre imagen y sigue las etapas:

1. conversion a **escala de grises**
2. suavizado con **GaussianBlur**
3. deteccion de bordes con **Canny**
4. extraccion de **contornos** con `cv2.findContours`
5. calculo de descriptores visuales clasicos:
   - densidad de bordes
   - estadisticas de intensidad
   - area, perimetro, bounding box, extent y solidity del contorno principal
   - **Hu Moments**
   - **HOG**
6. entrenamiento de un clasificador lineal sobre dichas features

Este segundo pipeline ya se encuentra implementado como codigo, aunque su ejecucion cuantitativa completa depende de capturar previamente un dataset visual suficientemente balanceado en `data_set_cv/`.

### 3.3. Herramientas Utilizadas

Las herramientas principales empleadas en esta etapa son:

- **MediaPipe Holistic** para extraccion de landmarks.
- **Python** para preprocesamiento y entrenamiento.
- **NumPy** para manejo de arreglos y tensores.
- **SciPy** para interpolacion temporal.
- **scikit-learn** para division estratificada del dataset.
- **OpenCV** para captura visual, escala de grises, filtrado, Canny y contornos.
- **HTML/CSS/JavaScript** para la interfaz de captura.

---

## 4. METODOLOGIA: DESARROLLO DEL MODELO DE PERCEPCION

### 4.1. Modelo Base (Baseline Clasico)

El proyecto dispone actualmente de **dos lineas base clasicas** complementarias:

1. un baseline estadistico sobre secuencias de keypoints (`baseline_clasico.py`)
2. un baseline visual construido directamente desde imagen (`baseline_cv.py`)

La primera linea base sirve para comparar el modelo Bi-LSTM contra un enfoque clasico dentro del mismo espacio de representacion por keypoints. La segunda responde de forma mas estricta a la exigencia metodologica del curso, ya que trabaja sobre imagenes mediante tecnicas clasicas de percepcion computacional.

**Baseline 1: k-NN sobre secuencias de keypoints**

El procedimiento seguido por el baseline fue el siguiente:

1. Se cargaron los tensores `X_train.npy`, `X_val.npy` y `X_test.npy`.
2. Cada secuencia de forma `(60, 855)` se transformo en un vector fijo de **3420 features** mediante la concatenacion de:
   - media por feature,
   - desviacion estandar por feature,
   - minimo por feature,
   - maximo por feature.
3. Sobre esos vectores se entreno un clasificador **k-NN** con estandarizacion previa (`StandardScaler`).
4. Se evaluaron varias configuraciones de hiperparametros en validacion:
   - `k` en `{1, 3, 5, 7, 9, 11}`
   - `weights` en `uniform` y `distance`
5. La mejor configuracion encontrada fue:
   - `k = 3`
   - `weights = distance`

Este baseline se almacena como artefacto reproducible en:

- `DATOS/baseline/baseline_knn.joblib`
- `DATOS/baseline/baseline_clasico_info.json`
- `DATOS/baseline/baseline_clasico_reporte.txt`
- `DATOS/baseline/baseline_clasico_matriz_confusion.json`

Los resultados obtenidos con este enfoque fueron:

- **Validation accuracy:** 92.95%
- **Validation macro F1:** 0.9295
- **Test accuracy:** 92.36%
- **Test macro F1:** 0.9240

**Baseline 2: pipeline visual clasico con OpenCV**

Con el fin de incorporar procesamiento directo sobre pixeles, se implemento tambien `baseline_cv.py`, cuyo flujo metodologico es:

1. lectura de imagenes RGB capturadas con `capturar_cv.py`
2. conversion a **escala de grises**
3. suavizado con `GaussianBlur`
4. deteccion de bordes con `cv2.Canny`
5. extraccion de contornos externos con `cv2.findContours`
6. construccion de un vector clasico de caracteristicas a partir de:
   - densidad de bordes
   - media y desviacion estandar de intensidad
   - area y perimetro del contorno dominante
   - bounding box, aspect ratio, extent y solidity
   - **Hu Moments**
   - **HOG**
7. entrenamiento de un clasificador **LinearSVC**

Este baseline visual ya se encuentra **implementado y documentado** dentro del repositorio. Sin embargo, su evaluacion cuantitativa completa queda condicionada a la captura de un dataset visual por clase en `data_set_cv/`, por lo que actualmente debe considerarse **implementado pero pendiente de corrida experimental final**.

Con ello, el proyecto deja de depender exclusivamente de una representacion esquelética generada por software de terceros y pasa a incorporar tambien una ruta clasica de percepcion computacional basada en pixeles, mas alineada con las competencias del curso.

### 4.2. Arquitectura del Modelo Avanzado

El modelo avanzado del proyecto esta implementado en `entrenar_bilstm.py` y corresponde a una red secuencial con capas recurrentes profundas. La arquitectura utiliza como entrada tensores de forma `(60, 855)`, donde cada muestra representa una secuencia temporal ya normalizada y enriquecida con informacion cinematica.

La estructura principal es la siguiente:

```text
Input (60, 855)
  -> Masking(mask_value=0.0)
  -> Bidirectional(LSTM(128, return_sequences=True))
  -> BatchNormalization
  -> Dropout(0.3)
  -> LSTM(128, return_sequences=True)
  -> BatchNormalization
  -> Dropout(0.3)
  -> LSTM(64, return_sequences=False)
  -> BatchNormalization
  -> Dropout(0.3)
  -> Dense(256, relu)
  -> Dropout(0.3)
  -> Dense(128, relu)
  -> Dropout(0.2)
  -> Dense(14, softmax)
```

La corrida validada mas reciente reporta **1,307,278 parametros**. El uso de una capa bidireccional en el primer bloque permite capturar contexto temporal futuro y pasado, algo especialmente relevante en senas donde el significado depende de la dinamica completa del gesto.

### 4.3. Proceso de Entrenamiento y Creacion del Modelo Final

El entrenamiento actual se ejecuta en un entorno separado `.venv-train` con **Python 3.11**, debido a restricciones de compatibilidad de TensorFlow en el entorno principal del proyecto. La configuracion validada hasta ahora funciona en **CPU** con:

- **TensorFlow 2.21.0**
- **Keras 3.15.0**

Los hiperparametros y estrategias utilizadas son:

- **Optimizador:** Adam (`learning_rate=0.001`, `clipvalue=1.0`)
- **Loss:** sparse categorical crossentropy
- **Batch size:** 16
- **Maximo de epocas:** configurable mediante `SIGNTALK_EPOCHS`
- **EarlyStopping:** configurable mediante `SIGNTALK_EARLY_STOP_PATIENCE`
- **Objetivo de parada por metrica:** configurable mediante `SIGNTALK_TARGET_VAL_ACC`
- **ReduceLROnPlateau:** factor 0.5, patience 10

Adicionalmente, el script genera y guarda:

- `modelo_bilstm.keras`
- `DATOS/propios/historial_bilstm.json`
- `DATOS/propios/modelo_bilstm_info.json`

Este punto del proyecto puede considerarse **implementado y funcional** en entorno local.

---

## 5. RESULTADOS DEL MODELO

### 5.1. Metricas de Rendimiento

La corrida de entrenamiento mas reciente almacenada en `DATOS/propios/historial_bilstm.json` reporta los siguientes resultados globales:

| Metrica | Valor |
|---|---:|
| Numero de clases | 14 |
| Frames por muestra | 60 |
| Features por frame | 855 |
| Epocas entrenadas | 33 |
| Mejor epoca | 33 |
| Mejor val_accuracy | 99.36% |
| Test accuracy | 100.00% |

Adicionalmente, el baseline clasico implementado en `baseline_clasico.py` permite una comparacion directa con el modelo profundo:

| Modelo | Validation Accuracy | Validation Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| Baseline k-NN sobre keypoints | 92.95% | 0.9295 | 92.36% | 0.9240 |
| Baseline visual OpenCV + Canny + contornos + HOG | Implementado | Pendiente de corrida con `data_set_cv/` | Implementado | Pendiente |
| Bi-LSTM | 99.36% | No consolidado en archivo separado | 100.00% | 1.0000 |

Estos valores muestran que el modelo logra un rendimiento muy alto sobre el conjunto de validacion y prueba disponible en el entorno actual. Sin embargo, es importante interpretar estos resultados con cautela por las siguientes razones:

- el dataset es propio y relativamente pequeno,
- la captura fue realizada en condiciones controladas,
- no existe aun una evaluacion cruzada con usuarios externos,
- y el rendimiento del baseline sobre keypoints, aunque alto, sigue estando por debajo del modelo Bi-LSTM, lo que sugiere que la modelacion temporal profunda aporta valor adicional en este problema.

### 5.2. Visualizaciones

El proyecto incluye el script `metricas.py`, pensado para mostrar en consola:

- curvas de loss y accuracy por epoca,
- resumen del modelo,
- resultados por clase,
- matriz de confusion,
- y analisis textual de errores.

No obstante, en la revision del estado actual del repositorio se identifica una **desincronizacion funcional**:

- el entrenamiento mas reciente opera sobre **14 clases**,
- mientras que algunas partes del script `metricas.py` y de la app web de inferencia continúan reflejando una configuracion previa de **7 clases**.

Por ello, aunque las metricas globales del historial actual son validas, las visualizaciones por clase y la matriz de confusion del script deben considerarse **pendientes de actualizacion** para reflejar fielmente el experimento vigente de 14 clases.

---

## 6. ARQUITECTURA DE DATOS Y PIPELINE A ESCALA

### 6.1. Tipo de Arquitectura

Hasta el momento, el proyecto implementa una **arquitectura batch local**. Es decir, el flujo de datos se ejecuta por etapas discretas:

1. captura,
2. almacenamiento del dataset,
3. preprocesamiento offline,
4. entrenamiento offline,
5. inferencia web local.

No se ha implementado aun una arquitectura distribuida con Spark, una capa de ingesta masiva de datos ni un pipeline de streaming. Estas capacidades forman parte del alcance academico previsto, pero todavia no existen como codigo funcional dentro del repositorio.

### 6.2. Diagrama de Arquitectura del Sistema

**Arquitectura implementada actualmente**

```text
Camara Web
   |
   +--> capturar.html + MediaPipe Holistic
   |       |
   |       v
   |   data_set/signtalk_dataset_unificado.json
   |       |
   |       v
   |   preprocessing.py
   |       |
   |       +--> X_train.npy / X_val.npy / X_test.npy
   |       +--> y_train.npy / y_val.npy / y_test.npy
   |       +--> label_map.json
   |       |
   |       v
   |   entrenar_bilstm.py
   |       |
   |       +--> modelo_bilstm.keras
   |       +--> DATOS/propios/historial_bilstm.json
   |       +--> DATOS/propios/modelo_bilstm_info.json
   |       |
   |       v
   |   index.html + app.js + modelo_bilstm_web/
   |       |
   |       +--> reconocimiento en tiempo real
   |       +--> formulacion de oraciones
   |       +--> lectura local con speechSynthesis
   |
   +--> capturar_cv.py
           |
           v
       data_set_cv/
           |
           v
       baseline_cv.py
           |
           +--> DATOS/baseline_cv/baseline_cv_svm.joblib
           +--> DATOS/baseline_cv/baseline_cv_info.json
```

**Arquitectura objetivo propuesta segun el silabo**

```text
Captura de nuevas secuencias / ingesta masiva por lotes
   |
   v
Data Lake o almacenamiento estructurado
   |
   v
Pipeline batch distribuido con Spark
   |
   +--> preprocesamiento distribuido
   +--> generacion de datasets de entrenamiento
   +--> evaluacion y trazabilidad experimental
   |
   v
Modelo de reconocimiento de palabras medicas
   |
   v
Modulo de composicion de oraciones
   |
   v
Modulo de sintesis de voz
   |
   v
Asistencia comunicacional multimodal
```

### 6.3. Implementacion y Tecnologias

La implementacion actual del pipeline a escala se limita a un flujo reproducible en una sola maquina:

- **Captura en navegador** mediante JavaScript y MediaPipe.
- **Captura visual clasica** mediante OpenCV en `capturar_cv.py`.
- **Preprocesamiento offline** en Python.
- **Entrenamiento local** con TensorFlow/Keras.
- **Baseline visual clasico** con OpenCV + Canny + contornos + HOG.
- **Inferencia web local** con TensorFlow.js.
- **Formulacion de oraciones** mediante reglas semanticas en la app web.
- **Lectura local por voz** mediante `speechSynthesis`.

Desde el punto de vista academico, esta arquitectura demuestra una version funcional del sistema de percepcion computacional tanto en su ruta avanzada por keypoints como en una ruta clasica basada en imagen. **Todavia no constituye una arquitectura Big Data distribuida** y no incluye ingesta masiva real con Spark, UDFs ni ejecucion distribuida. Ademas, aunque ya existen formulacion de oraciones y lectura local por voz, el reconocimiento continuo de secuencias largas y una capa de generacion linguistica mas avanzada siguen siendo aspectos **pendientes de implementacion**.

---

## 7. DESPLIEGUE Y OPERACIONALIZACION (MLOps)

### 7.1. Contenerizacion y Servicio de Inferencia

Actualmente, el proyecto cuenta con una aplicacion web local que se sirve con `python -m http.server` y consume un modelo TensorFlow.js desde `modelo_bilstm_web/`. Este despliegue permite demostracion funcional en navegador, acceso a camara e inferencia local de senas aisladas.

Sin embargo, **todavia no existe** en el repositorio:

- un `Dockerfile`,
- una API REST de inferencia,
- ni una contenerizacion formal del modelo.

Por lo tanto, esta seccion debe considerarse **parcialmente desarrollada**: existe despliegue local de prototipo con reconocimiento, formulacion de oraciones y lectura por voz dentro del navegador, pero no operacionalizacion formal tipo servicio. Aun no existe una API desacoplada para inferencia, ni un servicio dedicado para TTS, ni contenerizacion formal del sistema.

### 7.2. Seguimiento de Experimentos

El proyecto ya almacena historial de entrenamiento y metadatos en archivos JSON (`historial_bilstm.json` y `modelo_bilstm_info.json`), lo cual aporta trazabilidad minima de:

- numero de epocas,
- accuracy,
- val_accuracy,
- numero de clases,
- parametros del input.

No obstante, **no se ha integrado aun MLflow** ni otra plataforma especializada de seguimiento de experimentos. La reproducibilidad existe a nivel de scripts y artefactos, pero no a nivel de dashboard, comparacion historica de corridas o versionado formal de ejecuciones.

### 7.3. Gobernanza y Registro del Modelo

No se ha implementado todavia un **Model Registry** ni un mecanismo de versionado formal del modelo mas alla del archivo `modelo_bilstm.keras`. El control de versiones del codigo se apoya en Git, pero el ciclo de vida del modelo aun no cuenta con:

- versionado semantico del modelo,
- estados tipo staging/production,
- despliegue automatizado,
- ni trazabilidad centralizada de artefactos.

Esta seccion queda por desarrollar en etapas futuras.

---

## 8. DISCUSION Y LIMITACIONES

### 8.1. Desafios Tecnicos

Durante el desarrollo se identifican los siguientes desafios tecnicos relevantes:

- **Compatibilidad de entorno:** fue necesario separar el entorno de preprocesamiento (`.venv`) del entorno de entrenamiento (`.venv-train`) por incompatibilidades de TensorFlow con Python 3.14.
- **Desincronizacion entre entrenamiento e inferencia web:** el entrenamiento mas reciente utiliza 14 clases, mientras la app web en `app.js` sigue acoplada a un conjunto anterior de 7 clases.
- **Visualizaciones parcialmente desactualizadas:** `metricas.py` conserva componentes hardcodeados de una corrida previa, por lo que no refleja completamente el experimento actual.
- **Escalabilidad limitada:** el flujo funciona localmente, pero no ha sido migrado a un entorno distribuido o de produccion.
- **Formulacion linguistica aun simple:** el sistema ya compone oraciones, pero actualmente lo hace mediante reglas semanticas fijas y no mediante un modelo linguistico entrenado.
- **Lectura por voz dependiente del navegador:** la sintesis de voz ya funciona localmente, pero depende de `speechSynthesis` y de las voces instaladas en el sistema operativo, lo que puede generar variabilidad entre entornos.

### 8.2. Etica, Sesgos y Sostenibilidad

El proyecto presenta tambien limitaciones desde una perspectiva etica y de generalizacion:

- **Sesgo de captura:** al tratarse de un dataset propio, el modelo puede estar sobreajustado a pocos usuarios, iluminaciones, posiciones de camara y estilos gestuales.
- **Cobertura limitada del vocabulario:** aunque el proyecto aborda senas medicas relevantes, no representa la totalidad del vocabulario clinico ni la variabilidad real de la lengua de senas en escenarios hospitalarios.
- **Riesgo de interpretacion automatica:** un sistema como este no debe reemplazar el juicio medico ni la labor de interpretes profesionales; debe entenderse como herramienta de apoyo.
- **Privacidad:** el uso de keypoints en lugar de imagenes completas reduce la exposicion de rasgos visuales, lo cual es una ventaja importante para el tratamiento responsable de datos humanos.
- **Impacto social positivo:** aun con sus limitaciones, el proyecto evidencia el potencial de la percepcion computacional para reducir barreras comunicacionales en atencion medica.

---

## 9. CONCLUSIONES

### 9.1. Logros Alcanzados

Hasta el punto actual del proyecto se han logrado los siguientes avances concretos:

1. Se implemento una herramienta web funcional para captura de secuencias de senas mediante MediaPipe Holistic.
2. Se construyo un dataset propio estructurado en formato JSON y un pipeline de preprocesamiento reproducible.
3. Se desarrollo un clasificador Bi-LSTM funcional para secuencias de keypoints con 855 features por frame.
4. Se valido un entorno de entrenamiento estable en macOS usando Python 3.11 y TensorFlow/Keras.
5. Se genero un modelo entrenado y artefactos de historial y metadatos.
6. Se dispuso una aplicacion web local para inferencia en tiempo real.
7. Se implemento un baseline visual clasico sobre imagen con OpenCV, Canny, contornos, Hu Moments y HOG.
8. Se incorporo una formulacion inicial de oraciones dentro de la app web.
9. Se agrego lectura local por voz mediante `speechSynthesis`.

En consecuencia, el proyecto ya demuestra un flujo funcional desde la adquisicion de datos hasta la inferencia local de senas aisladas, incluyendo una salida comunicacional basica en texto y voz. Sin embargo, todavia no cubre todo el alcance previsto por el silabo en materia de arquitectura a escala, MLOps y reconocimiento continuo de secuencias mas complejas.

### 9.2. Mejoras Futuras

Como siguientes pasos recomendados para fortalecer tecnica y academicamente el proyecto, se propone:

1. Sincronizar la **app web** y `metricas.py` con el modelo actual de 14 clases.
2. Aumentar el volumen y diversidad del dataset con mas usuarios y condiciones de captura.
3. Agregar evaluaciones mas robustas, como validacion cruzada o pruebas con sujetos no vistos.
4. Implementar la etapa de **reconocimiento continuo** que permita encadenar palabras reconocidas en secuencias mas largas.
5. Mejorar el modulo de **composicion de oraciones** para pasar de reglas fijas a una formulacion mas natural y robusta.
6. Mejorar la **sintesis de voz** actual o reemplazarla por un motor TTS mas controlable y consistente entre plataformas.
7. Incorporar una API de inferencia y contenerizacion con Docker.
8. Integrar seguimiento de experimentos con MLflow.
9. Extender el pipeline a una arquitectura batch distribuida con Spark e ingesta masiva de datos si el curso lo exige.

---

## 10. ANEXOS TECNICOS

Archivos tecnicos principales del proyecto:

- `capturar.html`: interfaz de captura de senas.
- `preprocessing.py`: pipeline de preprocesamiento.
- `entrenar_bilstm.py`: entrenamiento del modelo.
- `metricas.py`: visualizacion de metricas en consola.
- `index.html` y `app.js`: interfaz de inferencia en tiempo real.
- `modelo_bilstm.keras`: modelo entrenado.
- `modelo_bilstm_web/model.json`: modelo exportado para navegador.
- `data_set/signtalk_dataset_unificado.json`: dataset fuente.
- `DATOS/propios/historial_bilstm.json`: historial de entrenamiento.
- `DATOS/propios/modelo_bilstm_info.json`: metadatos del modelo.

---

## 11. REFERENCIAS

1. Instituto Nacional de Estadistica e Informatica (INEI). Reportes estadisticos sobre poblacion con discapacidad en el Peru.
2. Documentacion oficial de MediaPipe Holistic.
3. Documentacion oficial de TensorFlow.
4. Documentacion oficial de Keras.
5. Documentacion oficial de scikit-learn.
6. Goodfellow, I., Bengio, Y., Courville, A. *Deep Learning*. MIT Press.
7. Graves, A., Schmidhuber, J. *Framewise Phoneme Classification with Bidirectional LSTM and Other Neural Network Architectures*.
