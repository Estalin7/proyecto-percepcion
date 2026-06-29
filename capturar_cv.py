"""
capturar_cv.py
==============

Captura imagenes desde webcam para un baseline clasico de vision por computadora.
El objetivo es construir un dataset basado en pixeles para luego aplicar:

- escala de grises
- suavizado gaussiano
- deteccion de bordes con Canny
- extraccion de contornos

Uso:
    python capturar_cv.py --label DOLOR --output-dir data_set_cv --max-images 80

Controles:
    ESPACIO  -> guardar frame actual
    q        -> salir
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "SignTalk CV Capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Captura imagenes de webcam para el baseline clasico con OpenCV."
    )
    parser.add_argument("--label", required=True, help="Nombre de la clase a capturar.")
    parser.add_argument(
        "--output-dir",
        default="data_set_cv",
        help="Directorio raiz del dataset visual.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=60,
        help="Numero maximo de imagenes a capturar.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Indice de la camara a usar.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Tamano cuadrado al que se recorta y guarda cada imagen.",
    )
    return parser.parse_args()


def center_crop_square(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def build_preview(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame, gray_bgr, edges_bgr


def overlay_text(panel: np.ndarray, lines: list[str]) -> np.ndarray:
    out = panel.copy()
    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (10, 24 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    args = parse_args()
    label_dir = Path(args.output_dir) / args.label.upper()
    label_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la camara para la captura con OpenCV.")

    print("\n=== SignTalk CV Capture ===")
    print(f"Clase          : {args.label.upper()}")
    print(f"Directorio     : {label_dir}")
    print(f"Max imagenes   : {args.max_images}")
    print("Controles      : ESPACIO=guardar | q=salir\n")

    saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No se pudo leer un frame desde la camara.")
                break

            crop = center_crop_square(frame, args.size)
            rgb, gray, edges = build_preview(crop)
            preview = np.hstack([rgb, gray, edges])
            preview = overlay_text(
                preview,
                [
                    f"Clase: {args.label.upper()}",
                    f"Guardadas: {saved}/{args.max_images}",
                    "ESPACIO: guardar",
                    "q: salir",
                ],
            )

            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32:
                timestamp = int(time.time() * 1000)
                out_path = label_dir / f"{args.label.upper()}_{timestamp}_{saved:04d}.png"
                cv2.imwrite(str(out_path), crop)
                saved += 1
                print(f"[{saved:03d}] Guardada: {out_path}")

                if saved >= args.max_images:
                    print("\nCaptura completada.")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
