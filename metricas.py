"""
metricas.py — Visualización completa de métricas del modelo SignTalk Bi-LSTM
=============================================================================
USO:
    python metricas.py

Muestra en consola:
  - Curvas de entrenamiento (loss y accuracy por época)
  - Reporte de clasificación por seña
  - Matriz de confusión
  - Resumen del dataset y modelo
"""

import json
import os
import numpy as np

# ──────────────────────────────────────────────────
# COLORES ANSI para consola
# ──────────────────────────────────────────────────
class C:
    RESET  = '\033[0m'
    BOLD   = '\033[1m'
    CYAN   = '\033[96m'
    GREEN  = '\033[92m'
    YELLOW = '\033[93m'
    RED    = '\033[91m'
    PURPLE = '\033[95m'
    MUTED  = '\033[90m'
    WHITE  = '\033[97m'
    BG_DARK = '\033[40m'

def titulo(texto):
    w = 60
    print(f"\n{C.PURPLE}{C.BOLD}{'═'*w}{C.RESET}")
    print(f"{C.PURPLE}{C.BOLD}  {texto}{C.RESET}")
    print(f"{C.PURPLE}{C.BOLD}{'═'*w}{C.RESET}")

def subtitulo(texto):
    print(f"\n{C.CYAN}{C.BOLD}  ▶ {texto}{C.RESET}")
    print(f"{C.MUTED}  {'─'*50}{C.RESET}")

def barra_horizontal(valor, maximo=1.0, ancho=30, color=C.GREEN):
    lleno = int(valor / maximo * ancho)
    vacio = ancho - lleno
    barra = '█' * lleno + '░' * vacio
    return f"{color}{barra}{C.RESET}"

# ──────────────────────────────────────────────────
# CARGAR DATOS
# ──────────────────────────────────────────────────
HIST_PATH = os.path.join("DATOS", "propios", "historial_bilstm.json")
INFO_PATH = os.path.join("DATOS", "propios", "modelo_bilstm_info.json")

if not os.path.exists(HIST_PATH):
    print(f"❌ No se encontró {HIST_PATH}")
    print("   Primero ejecuta: python entrenar_bilstm.py")
    exit(1)

with open(HIST_PATH, encoding='utf-8') as f:
    hist = json.load(f)

with open(INFO_PATH, encoding='utf-8') as f:
    info = json.load(f)

# ──────────────────────────────────────────────────
# 1. RESUMEN GENERAL
# ──────────────────────────────────────────────────
titulo("SIGNTALK — REPORTE DE METRICAS")

subtitulo("RESUMEN DEL MODELO")
print(f"  {'Arquitectura':<22} {C.WHITE}Bidirectional LSTM (Bi-LSTM){C.RESET}")
print(f"  {'Clases':<22} {C.WHITE}{info['num_clases']} señas medicas{C.RESET}")
print(f"  {'Features por frame':<22} {C.WHITE}{info['n_features']} (pos + vel + acel){C.RESET}")
print(f"  {'Frames por muestra':<22} {C.WHITE}{info['n_frames']} frames{C.RESET}")
print(f"  {'Modelo guardado':<22} {C.WHITE}{info['modelo']}{C.RESET}")

# ──────────────────────────────────────────────────
# 2. MÉTRICAS FINALES
# ──────────────────────────────────────────────────
subtitulo("RESULTADOS FINALES")

mejor_val  = hist['mejor_val_acc']
test_acc   = hist['test_accuracy']
mejor_ep   = hist['mejor_epoch']
total_ep   = hist['epochs_totales']

print(f"\n  {'Epocas entrenadas':<25} {C.YELLOW}{total_ep}{C.RESET}")
print(f"  {'Mejor epoca':<25} {C.YELLOW}{mejor_ep}{C.RESET}")
print()
print(f"  {'Val Accuracy (mejor)':<25} {barra_horizontal(mejor_val)} {C.GREEN}{C.BOLD}{mejor_val*100:.2f}%{C.RESET}")
print(f"  {'Test Accuracy':<25} {barra_horizontal(test_acc)} {C.CYAN}{C.BOLD}{test_acc*100:.2f}%{C.RESET}")
print()

# ──────────────────────────────────────────────────
# 3. CURVA DE ACCURACY POR ÉPOCA (visual en consola)
# ──────────────────────────────────────────────────
subtitulo("CURVA DE ACCURACY POR EPOCA")

train_acc = hist['accuracy']
val_acc   = hist['val_accuracy']
n = len(train_acc)

print(f"  {'Ep':>3}  {'Train Acc':>10}  {'Val Acc':>10}  {'Train':>30}  {'Val':>30}")
print(f"  {'─'*3}  {'─'*10}  {'─'*10}  {'─'*30}  {'─'*30}")

# Mostrar épocas relevantes (primeras 5, mejor época, últimas 3)
mostrar = set(range(0, min(5, n)))
mostrar.add(mejor_ep - 1)
mostrar.update(range(max(0, n-3), n))
mostrar = sorted(mostrar)

for i in mostrar:
    ta = train_acc[i]
    va = val_acc[i]
    barra_t = barra_horizontal(ta, color=C.PURPLE)
    barra_v = barra_horizontal(va, color=C.CYAN)
    marca = f" {C.YELLOW}← MEJOR{C.RESET}" if i == mejor_ep - 1 else ""
    print(f"  {i+1:>3}  {ta*100:>9.2f}%  {va*100:>9.2f}%  {barra_t}  {barra_v}{marca}")

if len(mostrar) < n:
    print(f"  {C.MUTED}  ... ({n - len(mostrar)} epocas omitidas){C.RESET}")

# ──────────────────────────────────────────────────
# 4. CURVA DE LOSS POR ÉPOCA
# ──────────────────────────────────────────────────
subtitulo("CURVA DE LOSS POR EPOCA")

train_loss = hist['loss']
val_loss   = hist['val_loss']
max_loss   = max(max(train_loss), max(val_loss))

print(f"  {'Ep':>3}  {'Train Loss':>11}  {'Val Loss':>11}  {'Train':>30}  {'Val':>30}")
print(f"  {'─'*3}  {'─'*11}  {'─'*11}  {'─'*30}  {'─'*30}")

for i in mostrar:
    tl = train_loss[i]
    vl = val_loss[i]
    # Invertir barra: pérdida baja = barra larga
    barra_t = barra_horizontal(max_loss - tl, maximo=max_loss, color=C.RED)
    barra_v = barra_horizontal(max_loss - vl, maximo=max_loss, color=C.YELLOW)
    marca = f" {C.YELLOW}← MEJOR{C.RESET}" if i == mejor_ep - 1 else ""
    print(f"  {i+1:>3}  {tl:>11.6f}  {vl:>11.6f}  {barra_t}  {barra_v}{marca}")

# ──────────────────────────────────────────────────
# 5. PRECISIÓN POR CLASE (desde reporte guardado)
# ──────────────────────────────────────────────────
subtitulo("RESULTADOS POR CLASE")

# Datos hardcodeados del reporte de clasificación (corridos en entrenar_bilstm.py)
clases_reporte = {
    'ARDOR':    {'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 11},
    'DOLOR':    {'precision': 1.000, 'recall': 0.909, 'f1': 0.952, 'support': 11},
    'ESPALDA':  {'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 11},
    'ESTOMAGO': {'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 11},
    'GARGANTA': {'precision': 1.000, 'recall': 0.818, 'f1': 0.900, 'support': 11},
    'GRIPE':    {'precision': 0.786, 'recall': 1.000, 'f1': 0.880, 'support': 11},
    'NAUSEAS':  {'precision': 1.000, 'recall': 1.000, 'f1': 1.000, 'support': 11},
}

print(f"\n  {'Seña':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}  {'F1 Score':>30}")
print(f"  {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*9}  {'─'*30}")

for seña, m in clases_reporte.items():
    f1_color = C.GREEN if m['f1'] >= 0.95 else (C.YELLOW if m['f1'] >= 0.85 else C.RED)
    barra_f1 = barra_horizontal(m['f1'], color=f1_color)
    print(f"  {seña:<12} {m['precision']:>10.3f} {m['recall']:>8.3f} {m['f1']:>8.3f} {m['support']:>9}  {barra_f1}")

# Promedio
avg_prec = np.mean([m['precision'] for m in clases_reporte.values()])
avg_rec  = np.mean([m['recall']    for m in clases_reporte.values()])
avg_f1   = np.mean([m['f1']        for m in clases_reporte.values()])
print(f"  {'─'*12} {'─'*10} {'─'*8} {'─'*8} {'─'*9}")
print(f"  {C.BOLD}{'PROMEDIO':<12} {avg_prec:>10.3f} {avg_rec:>8.3f} {avg_f1:>8.3f}{C.RESET}")

# ──────────────────────────────────────────────────
# 6. MATRIZ DE CONFUSIÓN
# ──────────────────────────────────────────────────
subtitulo("MATRIZ DE CONFUSION (Test Set)")

clases = hist['clases']
# Datos de la corrida de entrenamiento
cm = [
    [11,  0,  0,  0,  0,  0,  0],  # ARDOR
    [ 0, 10,  0,  0,  0,  1,  0],  # DOLOR
    [ 0,  0, 11,  0,  0,  0,  0],  # ESPALDA
    [ 0,  0,  0, 11,  0,  0,  0],  # ESTOMAGO
    [ 0,  0,  0,  0,  9,  2,  0],  # GARGANTA
    [ 0,  0,  0,  0,  0, 11,  0],  # GRIPE
    [ 0,  0,  0,  0,  0,  0, 11],  # NAUSEAS
]

# Header
abrevs = [c[:5] for c in clases]
header = f"  {'':>10} " + "  ".join(f"{a:>7}" for a in abrevs)
print(f"\n{C.MUTED}  (filas=real, columnas=predicho){C.RESET}\n")
print(header)
print(f"  {'':>10} " + "  ".join('─'*7 for _ in abrevs))

for i, fila in enumerate(cm):
    row_str = f"  {clases[i]:>10} "
    for j, v in enumerate(fila):
        if i == j:
            row_str += f"  {C.GREEN}{C.BOLD}{v:>5}{C.RESET}  "
        elif v > 0:
            row_str += f"  {C.RED}{v:>5}{C.RESET}  "
        else:
            row_str += f"  {C.MUTED}{v:>5}{C.RESET}  "
    print(row_str)

print(f"\n  {C.MUTED}  Verde = correctos | Rojo = errores{C.RESET}")

# ──────────────────────────────────────────────────
# 7. ANÁLISIS DE ERRORES
# ──────────────────────────────────────────────────
subtitulo("ANALISIS DE ERRORES")

print(f"  {C.YELLOW}Confusiones detectadas:{C.RESET}")
errores = [
    ("DOLOR",    "GRIPE",    1, "Señas visualmente similares (movimiento de mano al pecho/cuello)"),
    ("GARGANTA", "GRIPE",    2, "Zona anatómica cercana (garganta y síntomas de gripe)"),
]
for real, pred, n, motivo in errores:
    print(f"  • {C.RED}{real}{C.RESET} → clasificado como {C.YELLOW}{pred}{C.RESET} ({n} muestra{'s' if n>1 else ''})")
    print(f"    {C.MUTED}Posible causa: {motivo}{C.RESET}")

total_errores = sum(e[2] for e in errores)
total_test = 77
print(f"\n  Total errores: {C.RED}{total_errores}{C.RESET} / {total_test} muestras de test")
print(f"  Test Accuracy: {C.GREEN}{C.BOLD}{(total_test-total_errores)/total_test*100:.2f}%{C.RESET}")

# ──────────────────────────────────────────────────
# 8. RESUMEN DEL DATASET
# ──────────────────────────────────────────────────
subtitulo("RESUMEN DEL DATASET")

print(f"  {'Dataset':<25} {'Muestras':>10}  {'Shape':>20}")
print(f"  {'─'*25} {'─'*10}  {'─'*20}")
print(f"  {'X_train (entrenamiento)':<25} {'616':>10}  {'(616, 60, 855)':>20}")
print(f"  {'X_val (validacion)':<25} {'77':>10}  {'(77, 60, 855)':>20}")
print(f"  {'X_test (prueba)':<25} {'77':>10}  {'(77, 60, 855)':>20}")
print(f"  {'─'*25} {'─'*10}")
print(f"  {C.BOLD}{'TOTAL':<25} {'770':>10}{C.RESET}  {C.MUTED}(10 orig × 11 aug × 7 clases){C.RESET}")

print(f"\n  {'Pipeline de features:'}")
print(f"  • Pose:        {C.CYAN}33 kp × 3 = 99{C.RESET}")
print(f"  • Mano izq:    {C.CYAN}21 kp × 3 = 63{C.RESET}")
print(f"  • Mano der:    {C.CYAN}21 kp × 3 = 63{C.RESET}")
print(f"  • Cara top-20: {C.CYAN}20 kp × 3 = 60{C.RESET}")
print(f"  • Subtotal:    {C.YELLOW}285 features/frame (posición){C.RESET}")
print(f"  • + velocidad: {C.YELLOW}+285{C.RESET}")
print(f"  • + acelerac.: {C.YELLOW}+285{C.RESET}")
print(f"  • TOTAL:       {C.GREEN}{C.BOLD}855 features/frame{C.RESET}")

# ──────────────────────────────────────────────────
# CIERRE
# ──────────────────────────────────────────────────
print(f"\n{C.PURPLE}{'═'*60}{C.RESET}")
print(f"{C.GREEN}{C.BOLD}  ✓ Mejor modelo: época {mejor_ep} | Val: {mejor_val*100:.1f}% | Test: {test_acc*100:.2f}%{C.RESET}")
print(f"{C.PURPLE}{'═'*60}{C.RESET}\n")
