"""
entrenar_bilstm.py — Bi-LSTM para SignTalk (dataset médico propio)
===================================================================
Lee:  X_train.npy, X_val.npy, X_test.npy, y_*.npy, label_map.json
Guarda: modelo_bilstm.keras  +  DATOS/propios/historial_bilstm.json
"""

import os
import json
import numpy as np

# ──────────────────────────────────────────────────
# GPU
# ──────────────────────────────────────────────────
try:
    import nvidia.cudnn
    cudnn_bin = os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'bin')
    os.add_dll_directory(cudnn_bin)
    print("cuDNN path OK")
except Exception:
    pass

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"GPU: {gpus[0].name}")
else:
    print("Sin GPU, usando CPU")

import keras
print(f"Keras {keras.__version__} | TF {tf.__version__}")

from keras.models import Sequential
from keras.layers import (
    Input, Masking, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────
# CARGAR DATOS
# ──────────────────────────────────────────────────
print("\n[1] Cargando datos...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val   = np.load("X_val.npy")
y_val   = np.load("y_val.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

with open("label_map.json", encoding='utf-8') as f:
    label_map = json.load(f)

# Invertir: {0: "ARDOR", 1: "DOLOR", ...}
id2label = {v: k for k, v in label_map.items()}
nombres = [id2label[i] for i in range(len(id2label))]

N_FRAMES, N_FEAT = X_train.shape[1], X_train.shape[2]
NUM_CLASES = len(label_map)

print(f"   Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
print(f"   Frames={N_FRAMES} | Features={N_FEAT} | Clases={NUM_CLASES}")
print(f"   Clases: {nombres}")

# ──────────────────────────────────────────────────
# MODELO Bi-LSTM
# ──────────────────────────────────────────────────
print("\n[2] Construyendo modelo Bi-LSTM...")

model = Sequential([
    Input(shape=(N_FRAMES, N_FEAT)),
    Masking(mask_value=0.0),

    # Bloque 1: Bi-LSTM con proyección temporal
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),

    # Bloque 2: LSTM unidireccional
    LSTM(128, return_sequences=True),
    BatchNormalization(),
    Dropout(0.3),

    # Bloque 3: LSTM de salida (sin secuencia)
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),

    # Clasificador
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(NUM_CLASES, activation='softmax'),
], name="SignTalk_BiLSTM")

model.summary()
print(f"   Parametros totales: {model.count_params():,}")

model.compile(
    optimizer=Adam(learning_rate=0.001, clipvalue=1.0),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ──────────────────────────────────────────────────
# CALLBACKS
# ──────────────────────────────────────────────────
os.makedirs(os.path.join("DATOS", "propios"), exist_ok=True)

MODEL_OUT = "modelo_bilstm.keras"

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=30,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        MODEL_OUT,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
]

# ──────────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────────
print(f"\n[3] Entrenando en {'GPU' if gpus else 'CPU'}...")
print(f"   Epocas max=200 | Batch=16 | EarlyStopping patience=30\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

# ──────────────────────────────────────────────────
# EVALUACIÓN
# ──────────────────────────────────────────────────
print("\n[4] Evaluacion en conjunto de TEST...")
loss_t, acc_t = model.evaluate(X_test, y_test, verbose=0)
loss_v, acc_v = model.evaluate(X_val,  y_val,  verbose=0)
print(f"   Test  -> loss={loss_t:.4f} | accuracy={acc_t*100:.2f}%")
print(f"   Val   -> loss={loss_v:.4f} | accuracy={acc_v*100:.2f}%")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

print("\n[5] Reporte de clasificacion:\n")
reporte = classification_report(y_test, y_pred, target_names=nombres, digits=3)
print(reporte)

print("Matriz de confusion (filas=real, cols=predicho):")
cm = confusion_matrix(y_test, y_pred)
header = "        " + "  ".join(f"{n[:5]:>5}" for n in nombres)
print(header)
for i, row in enumerate(cm):
    print(f"{nombres[i][:7]:>7} " + "  ".join(f"{v:>5}" for v in row))

# ──────────────────────────────────────────────────
# GUARDAR HISTORIAL
# ──────────────────────────────────────────────────
hist_data = {
    'loss':              [float(v) for v in history.history['loss']],
    'val_loss':          [float(v) for v in history.history['val_loss']],
    'accuracy':          [float(v) for v in history.history['accuracy']],
    'val_accuracy':      [float(v) for v in history.history['val_accuracy']],
    'epochs_totales':    len(history.history['loss']),
    'mejor_val_acc':     float(max(history.history['val_accuracy'])),
    'mejor_epoch':       int(np.argmax(history.history['val_accuracy'])) + 1,
    'test_accuracy':     float(acc_t),
    'clases':            nombres,
    'n_frames':          int(N_FRAMES),
    'n_features':        int(N_FEAT),
}

hist_path = os.path.join("DATOS", "propios", "historial_bilstm.json")
with open(hist_path, 'w', encoding='utf-8') as f:
    json.dump(hist_data, f, ensure_ascii=False, indent=2)

# Info para inferencia web
model_info = {
    'n_features': int(N_FEAT),
    'n_frames':   int(N_FRAMES),
    'num_clases': NUM_CLASES,
    'label_map':  label_map,
    'nombres':    nombres,
    'test_accuracy': float(acc_t),
    'modelo':     MODEL_OUT,
}
with open(os.path.join("DATOS", "propios", "modelo_bilstm_info.json"), 'w', encoding='utf-8') as f:
    json.dump(model_info, f, ensure_ascii=False, indent=2)

print(f"\n=== RESULTADOS FINALES ===")
print(f"   Mejor val_accuracy : {hist_data['mejor_val_acc']*100:.1f}%  (epoch {hist_data['mejor_epoch']})")
print(f"   Test accuracy       : {acc_t*100:.2f}%")
print(f"   Modelo guardado en  : {MODEL_OUT}")
print(f"   Historial en        : {hist_path}")
print(f"\n>>> Siguiente paso:")
print(f"    tensorflowjs_converter --input_format keras {MODEL_OUT} modelo_bilstm_web/")
