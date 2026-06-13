/**
 * SignTalk — Reconocedor en tiempo real
 * Modelo: Bi-LSTM propio (96.1% test accuracy)
 * Dataset: 7 señas médicas propias
 * Pipeline: MediaPipe → 285 features → normalización + cinemáticas → 855 → Bi-LSTM → clase
 */

// ──────────────────────────────────────────────────────────
// CONSTANTES DEL PIPELINE (deben coincidir con preprocessing.py)
// ──────────────────────────────────────────────────────────
const N_KP       = 95;          // keypoints totales (285 / 3)
const COORDS     = 3;
const BASE_FEAT  = N_KP * COORDS;   // 285  (pose + manoIzq + manoDer + cara_top20)
const FULL_FEAT  = BASE_FEAT * 3;   // 855  (pos + vel + acel)
const T_FRAMES   = 60;              // ventana fija
const REF_KP     = 0;              // keypoint de referencia (nariz, idx 0 del vector aplanado)

// Clases (label_map invertido — coincide con sorted(data.keys()))
const CLASES = {
  0: 'ARDOR',
  1: 'DOLOR',
  2: 'ESPALDA',
  3: 'ESTOMAGO',
  4: 'GARGANTA',
  5: 'GRIPE',
  6: 'NAUSEAS'
};

const EMOJIS = {
  'ARDOR':     '🔥',
  'DOLOR':     '🤕',
  'ESPALDA':   '🪑',
  'ESTOMAGO':  '🤢',
  'GARGANTA':  '🗣️',
  'GRIPE':     '🤧',
  'NAUSEAS':   '🌀',
};

// ──────────────────────────────────────────────────────────
// ESTADO
// ──────────────────────────────────────────────────────────
let modelo = null;
let frameBuffer = [];     // buffer circular de frames crudos (285 features)
let ultimaSeña = null;
let contadorConfirm = 0;
let palabrasFrase = [];
const CONFIRM_FRAMES = 10;  // frames consecutivos para confirmar una seña
const MAX_FRASE = 8;
const UMBRAL_CONF = 0.70;   // confianza mínima para mostrar predicción

// ──────────────────────────────────────────────────────────
// DOM
// ──────────────────────────────────────────────────────────
const videoEl      = document.getElementById('videoElement');
const estadoEl     = document.getElementById('estado');
const resultadoEl  = document.getElementById('resultado');
const fraseEl      = document.getElementById('frase');
const emocionEl    = document.getElementById('emocion');
const confianzaEl  = document.getElementById('confianza');
const bufferBarEl  = document.getElementById('buffer-bar');

// ──────────────────────────────────────────────────────────
// 1. EXTRAER 285 FEATURES (igual que capturar.html)
// pose(99) + manoIzq(63) + manoDer(63) + cara_top20(60) = 285
// ──────────────────────────────────────────────────────────
function extraerPuntos(results) {
  const pose = results.poseLandmarks
    ? results.poseLandmarks.map(lm => [lm.x, lm.y, lm.z]).flat()
    : new Array(99).fill(0);

  const manoIzq = results.leftHandLandmarks
    ? results.leftHandLandmarks.map(lm => [lm.x, lm.y, lm.z]).flat()
    : new Array(63).fill(0);

  const manoDer = results.rightHandLandmarks
    ? results.rightHandLandmarks.map(lm => [lm.x, lm.y, lm.z]).flat()
    : new Array(63).fill(0);

  const face = results.faceLandmarks
    ? results.faceLandmarks.slice(0, 20).map(lm => [lm.x, lm.y, lm.z]).flat()
    : new Array(60).fill(0);

  return [...pose, ...manoIzq, ...manoDer, ...face]; // 285
}

// ──────────────────────────────────────────────────────────
// 2. NORMALIZACIÓN ESPACIAL (igual que preprocessing.py)
// Centra respecto al keypoint 0 (nariz) y escala por dist. máx.
// ──────────────────────────────────────────────────────────
function normalizarEspacial(seq) {
  // seq: Float32Array o Array de forma (T, 285)
  const T = seq.length;
  // Reshape mental a (T, 95, 3)
  // Punto de referencia: índice REF_KP → offset = REF_KP * 3

  const refOffset = REF_KP * COORDS; // 0
  const centered = seq.map(frame => {
    const refX = frame[refOffset];
    const refY = frame[refOffset + 1];
    const refZ = frame[refOffset + 2];
    return frame.map((v, i) => {
      const coord = i % COORDS;
      if (coord === 0) return v - refX;
      if (coord === 1) return v - refY;
      return v - refZ;
    });
  });

  // Escalar por distancia máxima en toda la secuencia
  let maxDist = 0;
  for (const frame of centered) {
    for (let k = 0; k < N_KP; k++) {
      const x = frame[k * COORDS];
      const y = frame[k * COORDS + 1];
      const z = frame[k * COORDS + 2];
      const d = Math.sqrt(x*x + y*y + z*z);
      if (d > maxDist) maxDist = d;
    }
  }
  if (maxDist < 1e-6) maxDist = 1;

  return centered.map(frame => frame.map(v => v / maxDist));
}

// ──────────────────────────────────────────────────────────
// 3. NORMALIZACIÓN TEMPORAL → T_FRAMES fijos
// Interpolación lineal sobre el índice de frame
// ──────────────────────────────────────────────────────────
function normalizarTemporal(seq, tOut) {
  const T = seq.length;
  if (T === tOut) return seq;

  const out = [];
  for (let i = 0; i < tOut; i++) {
    const t = i / (tOut - 1) * (T - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, T - 1);
    const alpha = t - lo;
    const frame = seq[lo].map((v, j) => v * (1 - alpha) + seq[hi][j] * alpha);
    out.push(frame);
  }
  return out;
}

// ──────────────────────────────────────────────────────────
// 4. CINEMÁTICAS: pos + vel + acel → 855 features
// ──────────────────────────────────────────────────────────
function agregarCinematicas(seq) {
  // seq: (T, 285)  →  (T, 855)
  const T = seq.length;
  const F = seq[0].length; // 285

  const vel  = Array.from({length: T}, () => new Array(F).fill(0));
  const acel = Array.from({length: T}, () => new Array(F).fill(0));

  for (let t = 1; t < T; t++) {
    for (let f = 0; f < F; f++) {
      vel[t][f] = seq[t][f] - seq[t-1][f];
    }
  }
  for (let t = 2; t < T; t++) {
    for (let f = 0; f < F; f++) {
      acel[t][f] = vel[t][f] - vel[t-1][f];
    }
  }

  return seq.map((frame, t) => [...frame, ...vel[t], ...acel[t]]); // (T, 855)
}

// ──────────────────────────────────────────────────────────
// 5. PREDICCIÓN COMPLETA
// ──────────────────────────────────────────────────────────
async function predecir(seqCruda) {
  // seqCruda: Array (T_FRAMES, 285) — ya tiene longitud exacta T_FRAMES
  const norm = normalizarEspacial(seqCruda);
  const full = agregarCinematicas(norm);  // (60, 855)

  // Crear tensor (1, 60, 855)
  const flat = [];
  for (const frame of full) for (const v of frame) flat.push(v);
  const tensor = tf.tensor3d([full], [1, T_FRAMES, FULL_FEAT]);

  let probs;
  try {
    const out = modelo.predict(tensor);
    probs = await out.data();
    out.dispose();
  } finally {
    tensor.dispose();
  }

  const claseIdx = probs.indexOf(Math.max(...probs));
  const confianza = probs[claseIdx];
  return { claseIdx, confianza, probs: Array.from(probs) };
}

// ──────────────────────────────────────────────────────────
// 6. DETECCIÓN DE EMOCIÓN
// ──────────────────────────────────────────────────────────
function detectarEmocion(faceLandmarks) {
  if (!faceLandmarks || faceLandmarks.length < 300) return null;
  try {
    const bocaIzq = faceLandmarks[61];
    const bocaDer = faceLandmarks[291];
    const labioSup = faceLandmarks[13];
    const labioInf = faceLandmarks[14];
    const cejaIzq = faceLandmarks[70];
    const cejaDer = faceLandmarks[300];
    const ojoCentro = faceLandmarks[168];
    if (!bocaIzq || !bocaDer || !labioSup || !labioInf) return null;
    const aperturaBoca = Math.abs(labioInf.y - labioSup.y);
    const centroMouth = (bocaIzq.y + bocaDer.y) / 2;
    const curvaturaBoca = centroMouth - labioSup.y;
    const alturaCejas = ojoCentro ? (cejaIzq.y + cejaDer.y) / 2 - ojoCentro.y : 0;
    if (aperturaBoca > 0.04)    return { emoji: '😮', nombre: 'Sorpresa', color: '#ff6b35' };
    if (curvaturaBoca < -0.01)  return { emoji: '😊', nombre: 'Tranquilo', color: '#00e5c0' };
    if (alturaCejas < -0.03)    return { emoji: '😟', nombre: 'Preocupado', color: '#ff4d6d' };
    return { emoji: '😐', nombre: 'Neutro', color: '#5a5a72' };
  } catch(e) { return null; }
}

// ──────────────────────────────────────────────────────────
// 7. FRASE
// ──────────────────────────────────────────────────────────
function agregarPalabra(palabra) {
  if (palabrasFrase[palabrasFrase.length - 1] === palabra) return;
  palabrasFrase.push(palabra);
  if (palabrasFrase.length > MAX_FRASE) palabrasFrase.shift();
  actualizarFrase();
}

function actualizarFrase() {
  if (!fraseEl) return;
  fraseEl.innerHTML = palabrasFrase.map((p, i) =>
    `<span class="word ${i === palabrasFrase.length-1 ? 'word-last' : ''}">${EMOJIS[p] || ''} ${p}</span>`
  ).join(' <span class="arrow">→</span> ');
}

function limpiarFrase() {
  palabrasFrase = [];
  if (fraseEl) fraseEl.innerHTML = '';
}

// ──────────────────────────────────────────────────────────
// 8. MEDIAPIPE CALLBACK
// ──────────────────────────────────────────────────────────
async function onResultados(results) {
  if (!modelo) return;

  // Emoción
  if (results.faceLandmarks && emocionEl) {
    const em = detectarEmocion(results.faceLandmarks);
    if (em) {
      emocionEl.innerHTML = `<span style="font-size:1.4em">${em.emoji}</span> <span style="color:${em.color}">${em.nombre}</span>`;
    }
  }

  // Solo si hay al menos una mano
  const hayManos = results.leftHandLandmarks || results.rightHandLandmarks;
  if (!hayManos) {
    // Si se quita la mano y el buffer tiene datos, limpiar suavemente
    if (frameBuffer.length > 0) {
      frameBuffer = [];
      updateBufferBar(0);
    }
    return;
  }

  // Acumular frame
  const frame = extraerPuntos(results);
  frameBuffer.push(frame);
  if (frameBuffer.length > T_FRAMES) frameBuffer.shift();

  const pct = (frameBuffer.length / T_FRAMES) * 100;
  updateBufferBar(pct);

  // Solo predecir cuando el buffer está lleno
  if (frameBuffer.length < T_FRAMES) return;

  try {
    const { claseIdx, confianza } = await predecir([...frameBuffer]);
    const nombreSeña = CLASES[claseIdx] || `?`;
    const pctConf = (confianza * 100).toFixed(1);

    if (confianza >= UMBRAL_CONF) {
      resultadoEl.innerHTML = `
        <span class="emoji-seña">${EMOJIS[nombreSeña] || ''}</span>
        <span class="nombre-seña">${nombreSeña}</span>
      `;
      confianzaEl.textContent = `${pctConf}%`;
      confianzaEl.style.color = confianza > 0.9 ? '#00e5c0' : confianza > 0.75 ? '#f59e0b' : '#ff6b6b';

      // Sistema de confirmación
      if (nombreSeña === ultimaSeña) {
        contadorConfirm++;
        if (contadorConfirm === CONFIRM_FRAMES) {
          agregarPalabra(nombreSeña);
        }
      } else {
        ultimaSeña = nombreSeña;
        contadorConfirm = 1;
      }
    } else {
      confianzaEl.textContent = `${pctConf}% (bajo)`;
      confianzaEl.style.color = '#5a5a72';
    }
  } catch(err) {
    console.error('Error predicción:', err);
  }
}

function updateBufferBar(pct) {
  if (bufferBarEl) bufferBarEl.style.width = pct + '%';
}

// ──────────────────────────────────────────────────────────
// 9. CARGAR MODELO
// ──────────────────────────────────────────────────────────
async function cargarModelo() {
  try {
    estadoEl.innerHTML = '⏳ Cargando modelo Bi-LSTM...';
    modelo = await tf.loadLayersModel('./modelo_bilstm_web/model.json');
    // Warm-up: inference vacía para que no haya lag en la primera predicción
    const dummy = tf.zeros([1, T_FRAMES, FULL_FEAT]);
    modelo.predict(dummy).dispose();
    dummy.dispose();
    estadoEl.innerHTML = '✅ Modelo cargado (96.1% acc) — Encendiendo cámara...';
    iniciarCamara();
  } catch(err) {
    estadoEl.innerHTML = '❌ Error cargando modelo. Abre desde un servidor local.';
    console.error(err);
  }
}

// ──────────────────────────────────────────────────────────
// 10. MEDIAPIPE + CÁMARA
// ──────────────────────────────────────────────────────────
const holistic = new Holistic({
  locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${f}`
});

holistic.setOptions({
  modelComplexity: 1,
  smoothLandmarks: true,
  enableSegmentation: false,
  refineFaceLandmarks: false,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5
});

holistic.onResults(onResultados);

function iniciarCamara() {
  const camera = new Camera(videoEl, {
    onFrame: async () => { await holistic.send({ image: videoEl }); },
    width: 640, height: 480
  });
  camera.start()
    .then(() => { estadoEl.innerHTML = '✅ Sistema activo — Realiza una seña médica'; })
    .catch(() => { estadoEl.innerHTML = '❌ No se pudo acceder a la cámara'; });
}

// ──────────────────────────────────────────────────────────
// INIT
// ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const btnLimpiar = document.getElementById('btn-limpiar');
  if (btnLimpiar) btnLimpiar.addEventListener('click', limpiarFrase);
});

cargarModelo();