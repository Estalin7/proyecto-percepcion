"""
baseline_cv.py
==============

Baseline clasico de vision por computadora sobre imagenes.

Pipeline metodologico:
    RGB -> escala de grises -> GaussianBlur -> Canny -> contornos -> HOG + Hu moments

Estructura esperada del dataset:
    data_set_cv/
        ARDOR/
            img_001.png
            ...
        DOLOR/
            ...

Artefactos generados:
    DATOS/baseline_cv/baseline_cv_svm.joblib
    DATOS/baseline_cv/baseline_cv_info.json
    DATOS/baseline_cv/baseline_cv_reporte.txt
    DATOS/baseline_cv/baseline_cv_matriz_confusion.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
MODEL_DIR = Path("DATOS") / "baseline_cv"
MODEL_PATH = MODEL_DIR / "baseline_cv_svm.joblib"
INFO_PATH = MODEL_DIR / "baseline_cv_info.json"
REPORT_PATH = MODEL_DIR / "baseline_cv_reporte.txt"
CM_PATH = MODEL_DIR / "baseline_cv_matriz_confusion.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena un baseline clasico sobre imagenes con OpenCV + Canny + contornos."
    )
    parser.add_argument(
        "--dataset-dir",
        default="data_set_cv",
        help="Directorio con subcarpetas por clase y archivos PNG/JPG.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Tamano cuadrado de normalizacion para cada imagen.",
    )
    return parser.parse_args()


def iter_image_files(dataset_dir: Path) -> list[tuple[Path, str]]:
    allowed = {".png", ".jpg", ".jpeg", ".webp"}
    samples: list[tuple[Path, str]] = []
    if not dataset_dir.exists():
        return samples

    for class_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        for file_path in sorted(class_dir.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in allowed:
                samples.append((file_path, label))
    return samples


def build_hog_descriptor() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize=(64, 64),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )


def safe_log_hu(hu: np.ndarray) -> np.ndarray:
    hu = hu.flatten()
    out = np.zeros_like(hu, dtype=np.float32)
    for i, value in enumerate(hu):
        if abs(value) < 1e-12:
            out[i] = 0.0
        else:
            out[i] = -np.sign(value) * np.log10(abs(value))
    return out.astype(np.float32)


def preprocess_image(image_path: Path, image_size: int, hog: cv2.HOGDescriptor) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (image_size, image_size), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_features = extract_contour_features(gray, edges, contours)
    hog_input = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    hog_features = hog.compute(hog_input).flatten().astype(np.float32)
    return np.concatenate([contour_features, hog_features], axis=0)


def extract_contour_features(
    gray: np.ndarray,
    edges: np.ndarray,
    contours: list[np.ndarray],
) -> np.ndarray:
    h, w = gray.shape[:2]
    edge_density = float(np.count_nonzero(edges)) / float(h * w)
    mean_intensity = float(np.mean(gray)) / 255.0
    std_intensity = float(np.std(gray)) / 255.0
    contour_count = float(len(contours))

    if not contours:
        base = np.array(
            [
                edge_density,
                mean_intensity,
                std_intensity,
                contour_count,
                0.0,  # area_norm
                0.0,  # perimeter_norm
                0.0,  # bbox_w
                0.0,  # bbox_h
                0.0,  # aspect_ratio
                0.0,  # extent
                0.0,  # solidity
            ],
            dtype=np.float32,
        )
        hu = np.zeros(7, dtype=np.float32)
        return np.concatenate([base, hu], axis=0)

    largest = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(largest))
    perimeter = float(cv2.arcLength(largest, True))
    _, _, bw, bh = cv2.boundingRect(largest)
    rect_area = float(max(bw * bh, 1))
    hull = cv2.convexHull(largest)
    hull_area = float(max(cv2.contourArea(hull), 1e-6))
    aspect_ratio = float(bw) / float(max(bh, 1))
    extent = area / rect_area
    solidity = area / hull_area
    hu = safe_log_hu(cv2.HuMoments(cv2.moments(largest)))

    base = np.array(
        [
            edge_density,
            mean_intensity,
            std_intensity,
            contour_count,
            area / float(h * w),
            perimeter / float(2 * (h + w)),
            float(bw) / float(w),
            float(bh) / float(h),
            aspect_ratio,
            extent,
            solidity,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, hu], axis=0)


def load_dataset(dataset_dir: Path, image_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    samples = iter_image_files(dataset_dir)
    if not samples:
        raise FileNotFoundError(
            "No se encontraron imagenes en data_set_cv/. "
            "Primero captura datos con capturar_cv.py."
        )

    hog = build_hog_descriptor()
    features = []
    labels = []
    for image_path, label in samples:
        features.append(preprocess_image(image_path, image_size, hog))
        labels.append(label)

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Se necesitan al menos 2 clases visuales para entrenar el baseline CV.")

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels)
    return x, y, unique_labels


def split_dataset(x: np.ndarray, y: np.ndarray):
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def evaluate_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    c_value: float,
):
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", LinearSVC(C=c_value, class_weight="balanced", max_iter=10000)),
        ]
    )
    model.fit(x_train, y_train)
    y_val_pred = model.predict(x_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    return model, val_acc, val_f1


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)

    print("\n[1] Cargando dataset visual...")
    x, labels_text, _ = load_dataset(dataset_dir, args.image_size)
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels_text)
    label_names = list(encoder.classes_)
    print(f"   Muestras: {x.shape[0]} | Features por imagen: {x.shape[1]}")
    print(f"   Clases  : {label_names}")

    print("\n[2] Dividiendo train/val/test...")
    x_train, x_val, x_test, y_train, y_val, y_test = split_dataset(x, y)
    print(f"   Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")

    c_grid = [0.1, 1.0, 5.0]
    mejor = None
    resultados = []

    print("\n[3] Buscando mejor SVM lineal...")
    for c_value in c_grid:
        model, val_acc, val_f1 = evaluate_model(x_train, y_train, x_val, y_val, c_value)
        resultados.append(
            {
                "C": c_value,
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_f1),
            }
        )
        print(f"   C={c_value:<4} | val_acc={val_acc*100:6.2f}% | val_macro_f1={val_f1:.4f}")

        if mejor is None or val_f1 > mejor["val_macro_f1"] or (
            val_f1 == mejor["val_macro_f1"] and val_acc > mejor["val_accuracy"]
        ):
            mejor = {
                "C": c_value,
                "val_accuracy": float(val_acc),
                "val_macro_f1": float(val_f1),
                "model": model,
            }

    assert mejor is not None
    print(
        f"\n[4] Mejor baseline CV: C={mejor['C']} | "
        f"val_acc={mejor['val_accuracy']*100:.2f}% | "
        f"val_macro_f1={mejor['val_macro_f1']:.4f}"
    )

    print("\n[5] Evaluando en TEST...")
    model = mejor["model"]
    y_test_pred = model.predict(x_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)
    reporte = classification_report(
        y_test,
        y_test_pred,
        target_names=label_names,
        digits=3,
        zero_division=0,
    )
    matriz = confusion_matrix(y_test, y_test_pred)

    print(f"   Test accuracy : {test_acc*100:.2f}%")
    print(f"   Test macro F1 : {test_macro_f1:.4f}")
    print("\n[6] Reporte de clasificacion:\n")
    print(reporte)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": encoder,
            "image_size": args.image_size,
            "feature_pipeline": "gray->gaussian->canny->contours+hog",
        },
        MODEL_PATH,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(reporte)

    with open(CM_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": label_names,
                "confusion_matrix": matriz.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    info = {
        "modelo": "LinearSVC",
        "descripcion": "Baseline visual clasico sobre imagenes con OpenCV, Canny, contornos y HOG",
        "dataset_dir": str(dataset_dir),
        "image_size": int(args.image_size),
        "n_muestras": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "pipeline_visual": [
            "cv2.cvtColor(..., COLOR_BGR2GRAY)",
            "cv2.GaussianBlur((5,5), 0)",
            "cv2.Canny(50, 150)",
            "cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)",
            "Hu moments",
            "HOG",
        ],
        "split": {
            "train": int(x_train.shape[0]),
            "val": int(x_val.shape[0]),
            "test": int(x_test.shape[0]),
        },
        "mejor_configuracion": {"C": float(mejor["C"])},
        "validacion": {
            "accuracy": float(mejor["val_accuracy"]),
            "macro_f1": float(mejor["val_macro_f1"]),
        },
        "test": {
            "accuracy": float(test_acc),
            "macro_f1": float(test_macro_f1),
        },
        "labels": label_names,
        "resultados_validacion": resultados,
        "artefactos": {
            "modelo": str(MODEL_PATH),
            "reporte": str(REPORT_PATH),
            "matriz_confusion": str(CM_PATH),
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
