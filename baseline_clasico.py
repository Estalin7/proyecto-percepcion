"""
baseline_clasico.py — Baseline clasico para SignTalk
====================================================
Lee:  X_train.npy, X_val.npy, X_test.npy, y_*.npy, label_map.json
Guarda:
  - DATOS/baseline/baseline_knn.joblib
  - DATOS/baseline/baseline_clasico_info.json
  - DATOS/baseline/baseline_clasico_reporte.txt
  - DATOS/baseline/baseline_clasico_matriz_confusion.json

Idea:
  1. Resume cada secuencia temporal con descriptores estadisticos.
  2. Entrena un clasificador clasico k-NN.
  3. Selecciona hiperparametros por macro F1 en validacion.
  4. Reporta metricas comparables con el Bi-LSTM.
"""

import json
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


K_GRID = [1, 3, 5, 7, 9, 11]
WEIGHTS_GRID = ["uniform", "distance"]
ALGORITHM = "auto"
MODEL_DIR = os.path.join("DATOS", "baseline")
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_knn.joblib")
INFO_PATH = os.path.join(MODEL_DIR, "baseline_clasico_info.json")
REPORT_PATH = os.path.join(MODEL_DIR, "baseline_clasico_reporte.txt")
CM_PATH = os.path.join(MODEL_DIR, "baseline_clasico_matriz_confusion.json")


def load_data():
    print("\n[1] Cargando datos del baseline...")
    x_train = np.load("X_train.npy")
    y_train = np.load("y_train.npy")
    x_val = np.load("X_val.npy")
    y_val = np.load("y_val.npy")
    x_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    with open("label_map.json", encoding="utf-8") as f:
        label_map = json.load(f)

    id2label = {v: k for k, v in label_map.items()}
    nombres = [id2label[i] for i in range(len(id2label))]

    print(f"   Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")
    print(f"   Clases: {len(nombres)} | Labels: {nombres}")
    return x_train, y_train, x_val, y_val, x_test, y_test, label_map, nombres


def build_statistical_features(x: np.ndarray) -> np.ndarray:
    """
    Convierte una secuencia (T, F) en un vector fijo:
    media + desviacion estandar + minimo + maximo por feature.
    """
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    min_v = x.min(axis=1)
    max_v = x.max(axis=1)
    return np.concatenate([mean, std, min_v, max_v], axis=1)


def evaluate_config(x_train_f, y_train, x_val_f, y_val, k, weights):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, algorithm=ALGORITHM)),
    ])
    model.fit(x_train_f, y_train)
    y_val_pred = model.predict(x_val_f)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    return model, val_acc, val_f1


def main():
    x_train, y_train, x_val, y_val, x_test, y_test, label_map, nombres = load_data()

    print("\n[2] Extrayendo descriptores estadisticos...")
    x_train_f = build_statistical_features(x_train)
    x_val_f = build_statistical_features(x_val)
    x_test_f = build_statistical_features(x_test)
    print(f"   Features baseline: {x_train_f.shape[1]} por muestra")

    print("\n[3] Buscando mejor configuracion k-NN...")
    resultados_val = []
    mejor = None

    for k in K_GRID:
        for weights in WEIGHTS_GRID:
            model, val_acc, val_f1 = evaluate_config(x_train_f, y_train, x_val_f, y_val, k, weights)
            resultados_val.append({
                "k": k,
                "weights": weights,
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_f1),
            })
            print(
                f"   k={k:>2} | weights={weights:<8} | "
                f"val_acc={val_acc*100:>6.2f}% | val_macro_f1={val_f1:.4f}"
            )

            if mejor is None or val_f1 > mejor["val_macro_f1"] or (
                val_f1 == mejor["val_macro_f1"] and val_acc > mejor["val_accuracy"]
            ):
                mejor = {
                    "k": k,
                    "weights": weights,
                    "val_accuracy": float(val_acc),
                    "val_macro_f1": float(val_f1),
                    "model": model,
                }

    print(
        "\n[4] Mejor baseline encontrado: "
        f"k={mejor['k']} | weights={mejor['weights']} | "
        f"val_acc={mejor['val_accuracy']*100:.2f}% | "
        f"val_macro_f1={mejor['val_macro_f1']:.4f}"
    )

    print("\n[5] Evaluando en TEST...")
    model = mejor["model"]
    y_test_pred = model.predict(x_test_f)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    reporte = classification_report(
        y_test,
        y_test_pred,
        target_names=nombres,
        digits=3,
        zero_division=0,
    )
    matriz = confusion_matrix(y_test, y_test_pred)

    print(f"   Test accuracy : {test_acc*100:.2f}%")
    print(f"   Test macro F1 : {test_macro_f1:.4f}")
    print("\n[6] Reporte de clasificacion:\n")
    print(reporte)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(reporte)

    with open(CM_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": nombres,
                "confusion_matrix": matriz.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    info = {
        "modelo": "k-NN",
        "descripcion": "Baseline clasico sobre descriptores estadisticos por secuencia",
        "features_por_muestra": int(x_train_f.shape[1]),
        "feature_extractor": [
            "mean(axis=1)",
            "std(axis=1)",
            "min(axis=1)",
            "max(axis=1)",
        ],
        "input_original": {
            "frames": int(x_train.shape[1]),
            "features_por_frame": int(x_train.shape[2]),
            "clases": len(nombres),
        },
        "mejor_configuracion": {
            "k": mejor["k"],
            "weights": mejor["weights"],
        },
        "validacion": {
            "accuracy": float(mejor["val_accuracy"]),
            "macro_f1": float(mejor["val_macro_f1"]),
        },
        "test": {
            "accuracy": float(test_acc),
            "macro_f1": float(test_macro_f1),
        },
        "label_map": label_map,
        "resultados_validacion": resultados_val,
        "artefactos": {
            "modelo": MODEL_PATH,
            "reporte": REPORT_PATH,
            "matriz_confusion": CM_PATH,
        },
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n[7] Artefactos generados:")
    print(f"   Modelo : {MODEL_PATH}")
    print(f"   Info   : {INFO_PATH}")
    print(f"   Reporte: {REPORT_PATH}")
    print(f"   Matriz : {CM_PATH}")


if __name__ == "__main__":
    main()
