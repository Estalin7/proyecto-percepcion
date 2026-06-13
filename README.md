# SignTalk ML — Reconocimiento de Señas Médicas

> Sistema de percepción computacional para reconocimiento de señas en Lengua de Señas mediante redes neuronales recurrentes (Bi-LSTM) y MediaPipe Holistic.

---

## Tabla de Contenidos

- [Descripción del proyecto](#descripcion)
- [Arquitectura del sistema](#arquitectura)
- [Señas reconocidas](#señas)
- [Estructura del repositorio](#estructura)
- [Requisitos e instalación](#instalacion)
- [Pipeline completo](#pipeline)
  - [Paso 1 — Captura de datos](#paso-1)
  - [Paso 2 — Preprocesamiento](#paso-2)
  - [Paso 3 — Entrenamiento](#paso-3)
  - [Paso 4 — Métricas](#paso-4)
  - [Paso 5 — Inferencia web](#paso-5)
- [Resultados del modelo](#resultados)
- [Decisiones de diseño](#decisiones)
- [Comandos de referencia](#comandos)

---

## Descripción del proyecto

SignTalk ML es un pipeline end-to-end de **percepción computacional** para reconocimiento de señas médicas en tiempo real. El sistema captura landmarks corporales con **MediaPipe Holistic**, los preprocesa aplicando normalización espacial y cinemáticas, y los clasifica usando un modelo **Bidirectional LSTM** entrenado con datos propios.

**Objetivo académico:** Demostrar el flujo completo de un sistema de reconocimiento de gestos: captura → preprocesamiento → entrenamiento → inferencia en tiempo real.

---

## Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE SIGNTALK ML                        │
│                                                                 │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │ capturar.html│───▶│ preprocessing.py│───▶│entrenar_      │  │
│  │  MediaPipe   │    │  normalización  │    │bilstm.py      │  │
│  │  285 feat.   │    │  cinemáticas    │    │  Bi-LSTM      │  │
│  │  JSON export │    │  855 feat. npy  │    │  96.1% acc.   │  │
│  └──────────────┘    └─────────────────┘    └───────┬───────┘  │
│                                                      │          │
│                      ┌───────────────────────────────▼───────┐  │
│                      │         inferencia en tiempo real      │  │
│                      │  index.html + app.js + modelo_web/     │  │
│                      └───────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Extracción de features (MediaPipe Holistic)

| Componente       | Keypoints | Features (×3) |
|------------------|-----------|----------------|
| Pose             | 33        | 99             |
| Mano izquierda   | 21        | 63             |
| Mano derecha     | 21        | 63             |
| Cara (top 20)    | 20        | 60             |
| **Subtotal pos** | **95**    | **285**        |
| + Velocidad      | —         | +285           |
| + Aceleración    | —         | +285           |
| **TOTAL**        | —         | **855**        |

### Modelo Bi-LSTM

```
Input  (1, 60, 855)
  ↓
Masking (mask_value=0.0)
  ↓
Bidirectional(LSTM(128))  + BatchNorm + Dropout(0.3)
  ↓
LSTM(128)                 + BatchNorm + Dropout(0.3)
  ↓
LSTM(64)                  + BatchNorm + Dropout(0.3)
  ↓
Dense(256, relu) + Dropout(0.3)
Dense(128, relu) + Dropout(0.2)
Dense(7, softmax)
  ↓
Output: probabilidades por clase
```

---

## Señas reconocidas

| # | Seña      | Emoji | Descripción                  |
|---|-----------|-------|------------------------------|
| 0 | ARDOR     | 🔥    | Sensación de ardor/quemazón  |
| 1 | DOLOR     | 🤕    | Dolor general                |
| 2 | ESPALDA   | 🪑    | Dolor de espalda             |
| 3 | ESTOMAGO  | 🤢    | Dolor o malestar estomacal   |
| 4 | GARGANTA  | 🗣️   | Dolor de garganta            |
| 5 | GRIPE     | 🤧    | Síntomas de gripe            |
| 6 | NAUSEAS   | 🌀    | Náuseas / mareos             |

---

## Estructura del repositorio

```
SignTalk_ML/
│
├── capturar.html              # Herramienta de captura (MediaPipe en browser)
├── preprocessing.py           # Preprocesador JSON → NumPy con cinemáticas
├── entrenar_bilstm.py         # Entrenamiento Bi-LSTM
├── metricas.py                # Visualización de métricas en consola
├── index.html                 # App de inferencia en tiempo real
├── app.js                     # Lógica de inferencia JS (mismo pipeline)
│
├── modelo_bilstm.keras        # Modelo entrenado (Keras)
├── modelo_bilstm_web/         # Modelo convertido a TensorFlow.js
│   ├── model.json
│   ├── group1-shard1of2.bin
│   └── group1-shard2of2.bin
│
├── label_map.json             # Mapeo clase → índice
├── X_train.npy                # Dataset de entrenamiento (616, 60, 855)
├── X_val.npy                  # Dataset de validación   (77, 60, 855)
├── X_test.npy                 # Dataset de test          (77, 60, 855)
├── y_train.npy / y_val.npy / y_test.npy
│
├── DATOS/
│   └── propios/
│       ├── historial_bilstm.json      # Curvas de entrenamiento
│       ├── modelo_bilstm_info.json    # Metadatos del modelo
│       └── reporte.txt               # Reporte de clasificación
│
├── venv_gpu/                  # Entorno Python con TensorFlow + GPU
└── venv_signtalk/             # Entorno Python con MediaPipe + TF.js tools
```

---

## Requisitos e instalación

### Entorno GPU (entrenamiento)
```powershell
# Ya configurado en venv_gpu/
venv_gpu\Scripts\activate
# Contiene: tensorflow, keras, scikit-learn, scipy, numpy
```

### Entorno SignTalk (herramientas)
```powershell
# Ya configurado en venv_signtalk/
venv_signtalk\Scripts\activate
# Contiene: mediapipe, tensorflowjs, scipy
```

### Para inferencia web
- Cualquier navegador moderno (Chrome / Edge recomendado)
- Servidor HTTP local (incluido en Python)

---

## Pipeline completo

### Paso 1 — Captura de datos

Abrir `capturar.html` en el navegador a través de un servidor local:

```powershell
python -m http.server 5500
# Abrir: http://localhost:5500/capturar.html
```

**Modo Batch:** Presiona "⚡ INICIAR BATCH COMPLETO" para capturar las 7 señas automáticamente (15 repeticiones × 3 segundos cada una, ~7 min total).

**Modo Manual:** Escribe el nombre de la seña y presiona "▶ INICIAR CAPTURA".

Al finalizar → presiona **"⬇ DESCARGAR JSON"** → guarda el archivo en la carpeta del proyecto.

**Formato de salida:**
```json
{
  "metadata": { "version": "2.0", "n_features": 285 },
  "data": {
    "ARDOR": [ [[frame1_f1, f2, ...f285], [frame2...], ...], ... ],
    "DOLOR": [ ... ]
  }
}
```

---

### Paso 2 — Preprocesamiento

```powershell
# Activar cualquier entorno con scipy + sklearn
venv_gpu\Scripts\activate

python preprocessing.py
```

**El script aplica en orden:**
1. **Manejo de NaN** — interpolación lineal si hay frames inválidos (descarta si >30%)
2. **Normalización espacial** — centra respecto a la nariz (kp 0), escala por distancia máxima
3. **Normalización temporal** — resamplea a **60 frames** fijos por interpolación lineal
4. **Data augmentation** (×10 por muestra):
   - Volteo horizontal (espejo en X)
   - Ruido gaussiano (σ=0.01)
   - Escala aleatoria (0.9–1.1)
   - Desplazamiento temporal (±5 frames)
5. **Cinemáticas** — concatena posición + velocidad + aceleración → **855 features**
6. **Split** — 80% train / 10% val / 10% test (estratificado)

**Salida:**
```
X_train.npy  (616, 60, 855)
X_val.npy    (77,  60, 855)
X_test.npy   (77,  60, 855)
y_*.npy
label_map.json
```

---

### Paso 3 — Entrenamiento

```powershell
venv_gpu\Scripts\activate
python entrenar_bilstm.py
```

**Hiperparámetros:**

| Parámetro         | Valor                         |
|-------------------|-------------------------------|
| Optimizador       | Adam (lr=0.001, clipvalue=1.0)|
| Loss              | Sparse Categorical Crossentropy|
| Batch size        | 16                            |
| Épocas máx.       | 200                           |
| EarlyStopping     | patience=30 (val_accuracy)    |
| ReduceLROnPlateau | factor=0.5, patience=10       |

**Salida:**
```
modelo_bilstm.keras
DATOS/propios/historial_bilstm.json
DATOS/propios/reporte.txt
DATOS/propios/modelo_bilstm_info.json
```

---

### Paso 4 — Métricas

```powershell
python metricas.py
```

Muestra en consola con visualización ASCII:
- Curvas de accuracy y loss por época (con barras de progreso)
- Reporte de clasificación por seña (precision / recall / F1)
- Matriz de confusión (verde = correcto, rojo = error)
- Análisis de errores con causa probable
- Resumen del dataset y pipeline de features

---

### Paso 5 — Inferencia web

#### Convertir modelo a TensorFlow.js
```powershell
venv_signtalk\Scripts\activate
tensorflowjs_converter --input_format keras modelo_bilstm.keras modelo_bilstm_web/
```

#### Iniciar servidor y abrir la app
```powershell
python -m http.server 5500
# Abrir: http://localhost:5500/index.html
```

**El pipeline de inferencia en JavaScript replica exactamente el de Python:**
1. Extraer 285 features por frame (pose + manos + cara)
2. Normalización espacial (centrar en nariz, escalar)
3. Resamplear a 60 frames por interpolación lineal
4. Agregar cinemáticas → 855 features
5. Predecir con el modelo Bi-LSTM
6. Confirmar predicción en 10 frames consecutivos → agregar a la frase

---

## Resultados del modelo

### Métricas globales

| Métrica              | Valor      |
|----------------------|------------|
| Val Accuracy (mejor) | **100.0%** |
| Test Accuracy        | **96.10%** |
| Épocas entrenadas    | 50         |
| Mejor época          | 20         |
| Parámetros totales   | ~1.2M      |

### Por clase (conjunto de test)

| Seña      | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| ARDOR     | 1.000     | 1.000  | **1.000**| 11      |
| DOLOR     | 1.000     | 0.909  | 0.952    | 11      |
| ESPALDA   | 1.000     | 1.000  | **1.000**| 11      |
| ESTOMAGO  | 1.000     | 1.000  | **1.000**| 11      |
| GARGANTA  | 1.000     | 0.818  | 0.900    | 11      |
| GRIPE     | 0.786     | 1.000  | 0.880    | 11      |
| NAUSEAS   | 1.000     | 1.000  | **1.000**| 11      |
| **Promedio** | **0.969** | **0.961** | **0.962** | 77 |

### Matriz de confusión

```
              ARDOR  DOLOR  ESPALDA  ESTOMAGO  GARGANTA  GRIPE  NAUSEAS
ARDOR          [11]     0       0        0         0       0       0
DOLOR            0    [10]      0        0         0       1       0   ← 1 error
ESPALDA          0      0     [11]       0         0       0       0
ESTOMAGO         0      0       0       [11]        0       0       0
GARGANTA         0      0       0        0         [9]     2       0   ← 2 errores
GRIPE            0      0       0        0          0     [11]      0
NAUSEAS          0      0       0        0          0       0      [11]
```

**Errores identificados:**
- `DOLOR → GRIPE` (1): Señas con movimiento de mano al pecho/cuello similares
- `GARGANTA → GRIPE` (2): Zona anatómica relacionada (ambas señas del área garganta-cuello)

---

## Decisiones de diseño

### ¿Por qué Bi-LSTM?
Las señas son **secuencias temporales** donde el contexto futuro ayuda a entender el pasado (por ejemplo, el comienzo de una seña puede depender de cómo termina). La capa `Bidirectional` procesa la secuencia en ambas direcciones, capturando este contexto completo.

### ¿Por qué cinemáticas (velocidad + aceleración)?
La posición sola no captura la **dinámica del movimiento**. Dos señas pueden tener keypoints similares en frames individuales pero diferente velocidad de movimiento. Agregar velocidad y aceleración triplica la información (285→855) sin aumentar el número de frames.

### ¿Por qué 60 frames fijos?
El modelo LSTM requiere input de longitud fija. 60 frames a ~30fps = 2 segundos, suficiente para capturar señas dinámicas. La interpolación temporal permite que secuencias de diferente duración sean comparables.

### ¿Por qué augmentation ×10?
Con solo 10 grabaciones por seña, el dataset es muy pequeño (70 muestras). El augmentation (volteo, ruido, escala, desplazamiento temporal) expande a 770 muestras y mejora la generalización sin necesidad de grabar más.

### ¿Por qué normalización espacial?
Sin normalizar, el modelo dependería de la posición de la persona en el encuadre. Al centrar respecto a la nariz y escalar por la distancia máxima, las señas son invariantes a la distancia a la cámara y posición en el frame.

---

## Comandos de referencia

```powershell
# ── VER MÉTRICAS EN CONSOLA ──────────────────────────────────
python metricas.py

# ── PIPELINE COMPLETO ────────────────────────────────────────

# 1. Servidor para captura y app
python -m http.server 5500

# 2. Preprocesar dataset capturado
venv_gpu\Scripts\activate
python preprocessing.py

# 3. Entrenar modelo
python entrenar_bilstm.py

# 4. Ver métricas
python metricas.py

# 5. Convertir a TF.js
venv_signtalk\Scripts\activate
tensorflowjs_converter --input_format keras modelo_bilstm.keras modelo_bilstm_web/

# ── ARCHIVOS CLAVE ───────────────────────────────────────────
# Captura:         capturar.html
# Preprocessing:   preprocessing.py
# Entrenamiento:   entrenar_bilstm.py
# Métricas:        metricas.py
# App web:         index.html + app.js
# Modelo Keras:    modelo_bilstm.keras
# Modelo Web:      modelo_bilstm_web/model.json
# Historial:       DATOS/propios/historial_bilstm.json
```

---

## Tecnologías utilizadas

| Tecnología       | Versión   | Uso                              |
|------------------|-----------|----------------------------------|
| Python           | 3.10      | Scripts de ML                    |
| TensorFlow/Keras | 2.x       | Modelo Bi-LSTM                   |
| TensorFlow.js    | 4.10      | Inferencia en browser            |
| MediaPipe        | Holistic  | Extracción de landmarks          |
| NumPy            | 1.x       | Manipulación de arrays           |
| SciPy            | 1.15      | Interpolación temporal           |
| scikit-learn     | 1.7       | Train/val/test split, métricas   |
| HTML/CSS/JS      | —         | Interfaz web                     |
