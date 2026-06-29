# SignTalk ML

Sistema de reconocimiento de señas médicas con `MediaPipe`, `Bi-LSTM` y demo web local con formulación de oraciones y lectura por voz.

## Lo mínimo para probarlo

### 1. Instalar dependencias base

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Levantar la demo web

```bash
python3 -m http.server 5500
```

Abre:

- [http://localhost:5500/index.html](http://localhost:5500/index.html)

La demo usa:

- cámara en tiempo real
- reconocimiento local en navegador
- construcción de frase
- generación de oración
- lectura por voz con `speechSynthesis`

## Requisitos para la demo

- Navegador moderno: `Chrome` o `Edge`
- Cámara habilitada
- Servidor local HTTP
- El modelo web debe existir en `modelo_bilstm_web/`

## Despliegue local

La app es estática. Para desplegarla localmente basta con servir el proyecto por HTTP:

```bash
cd /Users/dmedinasix/Desktop/proyecto-percepcion
python3 -m http.server 5500
```

Luego abre:

- [http://localhost:5500/index.html](http://localhost:5500/index.html)

## Flujo rápido completo

Si quieres regenerar datos y modelo:

### Dataset

Los dos archivos del dataset deben ir aquí:

- `data_set/signtalk_dataset_unificado.json`
- `data_set/signtalk_dataset_unificado.csv`

Si no existen, créalos o copia tus archivos en esa carpeta con esos nombres.

### Preprocesamiento

```bash
source .venv/bin/activate
python3 preprocessing.py
```

### Entrenamiento

`TensorFlow` no corre en la `.venv` normal con `Python 3.14`, por eso el entrenamiento usa `Python 3.11`.

```bash
python3.11 -m venv .venv-train
source .venv-train/bin/activate
pip install -r requirements-train.txt
python entrenar_bilstm.py
```

Prueba corta:

```bash
source .venv-train/bin/activate
SIGNTALK_EPOCHS=1 python entrenar_bilstm.py
```

### Métricas

```bash
source .venv/bin/activate
python3 metricas.py
```

## Archivos clave

- `index.html`: demo principal
- `app.js`: lógica de inferencia y voz
- `preprocessing.py`: genera `.npy`
- `entrenar_bilstm.py`: entrena el modelo
- `metricas.py`: muestra métricas
- `modelo_bilstm_web/model.json`: modelo para navegador
- `data_set/signtalk_dataset_unificado.json`: dataset fuente
- `data_set/signtalk_dataset_unificado.csv`: export auxiliar del dataset

## Notas

- La demo actual está orientada a prueba local.
- La lectura en voz depende de las voces disponibles en el navegador y sistema operativo.
- Si quieres desplegar fuera de `localhost`, conviene usar `HTTPS`.
- `.gitignore` ya está preparado para conservar `data_set/signtalk_dataset_unificado.json` y `data_set/signtalk_dataset_unificado.csv`.
