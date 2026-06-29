"""
SignTalk - Pipeline de Preprocesamiento
========================================
Entrada : JSON con { "SEÑA": [seq1, seq2, ...] }
          Cada seq = lista de frames, cada frame = lista de 285 floats (95 kp × 3)
          Soporta formato v2.0 (con clave "data" + "metadata") y v1.0 (dict plano)
Salida  : X_train, X_val, X_test (N, T, F) y y_* (N,) en .npy
          label_map.json
"""

import json
import os
import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────
JSON_PATH   = os.path.join("data_set", "signtalk_dataset_unificado.json")
T_FRAMES    = 60          # longitud fija de salida
N_KP        = 95          # keypoints totales (285 / 3)
COORDS      = 3           # x, y, z
BASE_FEAT   = N_KP * COORDS        # 285
FULL_FEAT   = BASE_FEAT * 3        # 855  (pos + vel + acel)

# Índice del keypoint de referencia para normalización espacial
# MediaPipe Pose: 0=nariz — primer keypoint del vector de 95
REF_KP_IDX = 0   # nariz

# ──────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ──────────────────────────────────────────────────
def load_json(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    # Soportar formato v2.0 con wrapper {metadata, data}
    if isinstance(raw, dict) and 'data' in raw and 'metadata' in raw:
        print(f"  Formato v2.0 detectado | n_features={raw['metadata'].get('n_features','?')}")
        return raw['data']
    return raw


# ──────────────────────────────────────────────────
# 2. MANEJO DE NaN – interpolación lineal por frame
# ──────────────────────────────────────────────────
def fix_nans(seq: np.ndarray):
    """
    seq: (T, F)  float array, puede tener NaN.
    Devuelve seq interpolada o None si >30% frames son inválidos.
    """
    nan_frames = np.any(np.isnan(seq), axis=1)   # (T,)
    nan_ratio  = nan_frames.mean()

    if nan_ratio > 0.30:
        return None   # descartar

    if nan_ratio == 0:
        return seq    # sin NaN, rápido

    # Interpolación lineal columna a columna
    t = np.arange(len(seq))
    valid = ~nan_frames
    seq_fixed = seq.copy()
    for col in range(seq.shape[1]):
        if np.any(np.isnan(seq[:, col])):
            f = interp1d(t[valid], seq[valid, col],
                         kind="linear", fill_value="extrapolate")
            seq_fixed[:, col] = f(t)
    return seq_fixed


# ──────────────────────────────────────────────────
# 3. NORMALIZACIÓN ESPACIAL
# ──────────────────────────────────────────────────
def normalize_spatial(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 285)  → reshape a (T, 95, 3)
    Centra cada frame respecto al keypoint de referencia (nariz).
    Escala por la distancia máxima al punto de referencia.
    """
    T = seq.shape[0]
    kps = seq.reshape(T, N_KP, COORDS)          # (T, 95, 3)

    # Punto de referencia por frame
    ref = kps[:, REF_KP_IDX, :]                 # (T, 3)

    # Centrar
    kps_centered = kps - ref[:, np.newaxis, :]  # broadcast (T, 95, 3)

    # Escalar por distancia máxima en la secuencia
    dists = np.linalg.norm(kps_centered.reshape(T, -1, 3), axis=2)  # (T, 95)
    max_dist = dists.max()
    if max_dist > 1e-6:
        kps_centered /= max_dist

    return kps_centered.reshape(T, BASE_FEAT)   # vuelve a (T, 285)


# ──────────────────────────────────────────────────
# 4. NORMALIZACIÓN TEMPORAL → T_FRAMES fijos
# ──────────────────────────────────────────────────
def normalize_temporal(seq: np.ndarray, t_out: int = T_FRAMES) -> np.ndarray:
    """
    seq: (T_orig, F) → (t_out, F) por interpolación lineal.
    """
    T_orig = seq.shape[0]
    if T_orig == t_out:
        return seq

    t_orig = np.linspace(0, 1, T_orig)
    t_new  = np.linspace(0, 1, t_out)
    out = np.zeros((t_out, seq.shape[1]), dtype=np.float32)
    for col in range(seq.shape[1]):
        f = interp1d(t_orig, seq[:, col], kind="linear")
        out[:, col] = f(t_new)
    return out


# ──────────────────────────────────────────────────
# 5. DERIVADAS TEMPORALES (velocidad + aceleración)
# ──────────────────────────────────────────────────
def add_kinematics(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, 285)
    Añade velocidad (Δp) y aceleración (Δ²p) → (T, 855)
    El primer y segundo frame se rellenan con ceros para mantener T.
    """
    vel  = np.zeros_like(seq)
    acel = np.zeros_like(seq)

    vel[1:]    = seq[1:] - seq[:-1]             # Δp(t)
    acel[2:]   = vel[2:] - vel[1:-1]            # Δ²p(t)

    return np.concatenate([seq, vel, acel], axis=1)  # (T, 855)


# ──────────────────────────────────────────────────
# 6. DATA AUGMENTATION
# ──────────────────────────────────────────────────
def augment(seq: np.ndarray, n: int = 5) -> list:
    """
    Genera `n` versiones aumentadas de una secuencia.
    seq: (T, 285) — ANTES de añadir cinemáticas.
    """
    augmented = []
    kps = seq.reshape(T_FRAMES, N_KP, COORDS)   # (T, 95, 3)

    for _ in range(n):
        aug = kps.copy()

        # a) Volteo horizontal (espejo en X)
        if np.random.rand() < 0.5:
            aug[:, :, 0] *= -1

        # b) Ruido gaussiano
        aug += np.random.normal(0, 0.01, aug.shape)

        # c) Escala aleatoria
        scale = np.random.uniform(0.9, 1.1)
        aug  *= scale

        # d) Desplazamiento temporal (±5 frames con wrap)
        shift = np.random.randint(-5, 6)
        aug   = np.roll(aug, shift, axis=0)

        augmented.append(aug.reshape(T_FRAMES, BASE_FEAT).astype(np.float32))

    return augmented


# ──────────────────────────────────────────────────
# 7. PIPELINE COMPLETO
# ──────────────────────────────────────────────────
def preprocess_dataset(data: dict,
                       augment_per_seq: int = 10,
                       use_kinematics: bool = True):
    """
    data: { "SEÑA": [[frame,...], ...], ... }
    Devuelve X (N, T, F), y (N,), label_map {str: int}
    """
    label_map = {label: i for i, label in enumerate(sorted(data.keys()))}
    X_list, y_list = [], []

    for label, sequences in data.items():
        cls_idx = label_map[label]
        print(f"  [{cls_idx:02d}] {label}: {len(sequences)} seqs", end="")

        accepted = 0
        for raw_seq in sequences:
            seq = np.array(raw_seq, dtype=np.float32)  # (T_orig, 285)

            # 2. Manejo NaN
            seq = fix_nans(seq)
            if seq is None:
                continue

            # 3. Normalización espacial
            seq = normalize_spatial(seq)

            # 4. Normalización temporal → T_FRAMES
            seq = normalize_temporal(seq, T_FRAMES)

            # Guardar original + augmentaciones
            seqs_to_add = [seq] + augment(seq, n=augment_per_seq)

            # 5. Cinemáticas opcionales (pos + vel + acel)
            for s in seqs_to_add:
                feat = add_kinematics(s) if use_kinematics else s
                X_list.append(feat)
                y_list.append(cls_idx)

            accepted += 1

        total = accepted * (1 + augment_per_seq)
        print(f"  -> {total} muestras")

    X = np.array(X_list, dtype=np.float32)   # (N, T, F)
    y = np.array(y_list, dtype=np.int32)     # (N,)
    print(f"\nDataset final: {X.shape}  |  clases: {len(label_map)}")
    return X, y, label_map


# ──────────────────────────────────────────────────
# 8. DIVISIÓN TRAIN / VAL / TEST
# ──────────────────────────────────────────────────
def split_dataset(X, y, val_size=0.10, test_size=0.10, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size),
        stratify=y, random_state=random_state
    )
    ratio_val = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - ratio_val),
        stratify=y_temp, random_state=random_state
    )
    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ──────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  SignTalk — Preprocesamiento")
    print("=" * 50)

    print(f"\n1. Cargando: {JSON_PATH}")
    data = load_json(JSON_PATH)
    print(f"   Señas encontradas: {sorted(data.keys())}")

    print("\n2. Preprocesando...")
    X, y, label_map = preprocess_dataset(
        data,
        augment_per_seq=10,   # 10 versiones por secuencia
        use_kinematics=True   # pos + vel + acel → 855 features
    )

    print("\n3. Dividiendo dataset (80/10/10)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y)

    print("\n4. Guardando archivos .npy ...")
    np.save("X_train.npy", X_train); np.save("y_train.npy", y_train)
    np.save("X_val.npy",   X_val);   np.save("y_val.npy",   y_val)
    np.save("X_test.npy",  X_test);  np.save("y_test.npy",  y_test)

    with open("label_map.json", "w", encoding='utf-8') as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    print("\n✅ Preprocesamiento completo.")
    print(f"   Shape modelo: ({X_train.shape[1]} frames, {X_train.shape[2]} features)")
    print(f"   Label map: {label_map}")
    print(f"\n▶  Siguiente: python entrenar_bilstm.py")
