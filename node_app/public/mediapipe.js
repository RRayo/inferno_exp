import { FaceDetector, FilesetResolver } from "./tasks-vision@0.10.0.js";

// --- 1. CONFIGURACIÓN Y CONSTANTES ---
const CONFIG = {
  minScore: 0.90,
  requiredConsecutiveFrames: 60,
  frontalThreshold: 0.7,
  detectionFontSize: '28px', // Tamaño del texto "Validando rostro..."
  successFontSize: '46px'    // Tamaño del texto "¡Captura completada!"
};

const DOM = {
  video: document.getElementById("webcam"),
  liveView: document.getElementById("liveView"),
  canvas: document.getElementById("snapshotCanvas")
};

const STATUS = {
    VALIDATING: (progress) => ({
        text: `Validando su rostro... ${progress}%`,
        color: '#FFC300'
    }),
    VALIDATED: {
        text: 'Rostro validado',
        color: '#009933'
    },
    INVALID: {
        text: 'Enderece su rostro',
        color: '#CC3300'
    }
};

let faceDetector;
let lastVideoTime = -1;
let detectionChildren = [];
let consecutiveFramesCounter = 0;
let appState = 'DETECTING';

// --- 2. INICIALIZACIÓN ---
async function main() {
  try {
    const vision = await FilesetResolver.forVisionTasks("./task_vision");
    faceDetector = await FaceDetector.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "blaze_face_short_range.tflite", delegate: "GPU" },
      runningMode: "VIDEO"
    });
    startWebcam();
  } catch (error) {
    console.error("Error al inicializar el detector de rostros:", error);
  }
}

function startWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    console.error("getUserMedia no es soportado en este navegador.");
    return;
  }
  navigator.mediaDevices.getUserMedia({ video: true })
    .then((stream) => {
      DOM.video.srcObject = stream;
      DOM.video.addEventListener("loadeddata", predictWebcam);
    })
    .catch((err) => {
      console.error("No se pudo acceder a la webcam:", err);
    });
}

// --- 3. BUCLE DE DETECCIÓN ---
async function predictWebcam() {
  if (appState === 'SUCCESS') return;

  if (DOM.video.currentTime !== lastVideoTime) {
    lastVideoTime = DOM.video.currentTime;
    const detections = faceDetector.detectForVideo(DOM.video, performance.now())?.detections || [];
    handleDetections(detections);
  }
  
  requestAnimationFrame(predictWebcam);
}

// --- 4. LÓGICA DE DETECCIÓN Y UI ---
function handleDetections(detections) {
  clearDetections();
  
  if (detections.length === 0) {
    consecutiveFramesCounter = 0;
    return;
  }

  const detection = detections[0];
  const score = detection.categories[0].score;
  const isFrontal = isFacingForward(detection.keypoints);
  
  let currentStatus;

  if (score > CONFIG.minScore && isFrontal) {
    consecutiveFramesCounter++;
    if (consecutiveFramesCounter >= CONFIG.requiredConsecutiveFrames) {
        captureAndFinalize();
        return;
    }
    if (consecutiveFramesCounter > 0) {
        const progress = Math.round((consecutiveFramesCounter / CONFIG.requiredConsecutiveFrames) * 100);
        currentStatus = STATUS.VALIDATING(progress);
    } else {
        currentStatus = STATUS.VALIDATED;
    }
  } else {
    consecutiveFramesCounter = 0;
    currentStatus = STATUS.INVALID;
  }
  
  updateDetectionUI(detection, currentStatus);
}

function updateDetectionUI(det, status) {
    const scale = DOM.video.clientHeight / DOM.video.videoHeight;
    const offsetX = (DOM.video.clientWidth - DOM.video.videoWidth * scale) / 2;
    
    const scaledWidth = det.boundingBox.width * scale;
    const scaledHeight = det.boundingBox.height * scale;
    const scaledOriginX = det.boundingBox.originX * scale;
    const scaledOriginY = det.boundingBox.originY * scale;

    const box = document.createElement("div");
    box.className = "highlighter";
    box.style.cssText = `
      left:${DOM.video.clientWidth - scaledOriginX - scaledWidth - offsetX}px; 
      top:${scaledOriginY}px; 
      width:${scaledWidth}px; 
      height:${scaledHeight}px;
      border-color: ${status.color};
      border-width: 3px;
    `;
    DOM.liveView.appendChild(box);
    detectionChildren.push(box);

    const p = document.createElement("p");
    p.className = 'detection-status';
    p.innerText = status.text;
    p.style.backgroundColor = status.color;
    p.style.left = `${DOM.video.clientWidth - scaledOriginX - scaledWidth - offsetX}px`;
    p.style.top = `${scaledOriginY - 40}px`;
    p.style.width = `${scaledWidth}px`;
    p.style.textAlign = 'center';
    p.style.fontSize = CONFIG.detectionFontSize;
    DOM.liveView.appendChild(p);
    detectionChildren.push(p);
}

function clearDetections() {
  for (let child of detectionChildren) {
    DOM.liveView.removeChild(child);
  }
  detectionChildren = [];
}

// --- 5. FUNCIONES AUXILIARES Y FINALIZACIÓN ---
function isFacingForward(keypoints) {
  const leftEye = keypoints[0];
  const rightEye = keypoints[1];
  const noseTip = keypoints[2];
  
  const distNoseToLeftEye = Math.abs(noseTip.x - leftEye.x);
  const distNoseToRightEye = Math.abs(noseTip.x - rightEye.x);
  
  if (distNoseToLeftEye === 0 || distNoseToRightEye === 0) return false;
  
  const ratio = Math.min(distNoseToLeftEye, distNoseToRightEye) / Math.max(distNoseToLeftEye, distNoseToRightEye);
  return ratio > CONFIG.frontalThreshold;
}

function captureAndFinalize() {
    appState = 'SUCCESS';
    saveFrame();
    clearDetections();
    showSuccessMessage();
}

function showSuccessMessage() {
    const p = document.createElement("p");
    p.innerText = "¡Captura completada!";
    p.style.cssText = `
        background-color: #009933;
        color: white;
        position: absolute;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        padding: 20px;
        border-radius: 10px;
        font-size: ${CONFIG.successFontSize};
        z-index: 10;
    `;
    DOM.liveView.appendChild(p);
    console.log("Captura completada.");
}

/**
 * Captura y descarga solo el área visible del video, replicando el 'object-fit: cover'.
 */
function saveFrame() {
  const ctx = DOM.canvas.getContext("2d");
  const video = DOM.video;
  const canvas = DOM.canvas;

  // 1. Establecer el tamaño del canvas igual al tamaño visible del video en pantalla
  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;

  // 2. Calcular las proporciones para replicar 'object-fit: cover'
  const videoAspectRatio = video.videoWidth / video.videoHeight;
  const canvasAspectRatio = canvas.width / canvas.height;
  let renderableWidth, renderableHeight, xStart, yStart;

  if (videoAspectRatio < canvasAspectRatio) {
    // Si el video es más 'alto' que el canvas, el ancho se ajusta y la altura se recorta
    renderableWidth = canvas.width;
    renderableHeight = renderableWidth / videoAspectRatio;
    xStart = 0;
    yStart = (canvas.height - renderableHeight) / 2;
  } else if (videoAspectRatio > canvasAspectRatio) {
    // Si el video es más 'ancho' que el canvas, la altura se ajusta y el ancho se recorta
    renderableHeight = canvas.height;
    renderableWidth = renderableHeight * videoAspectRatio;
    yStart = 0;
    xStart = (canvas.width - renderableWidth) / 2;
  } else {
    // Si las proporciones son iguales, no hay recorte
    renderableHeight = canvas.height;
    renderableWidth = canvas.width;
    xStart = 0;
    yStart = 0;
  }

  // 3. Aplicar el efecto espejo antes de dibujar
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);

  // 4. Dibujar el video en el canvas con el recorte y escala calculados
  // Esto simula el 'object-fit: cover'
  ctx.drawImage(video, xStart, yStart, renderableWidth, renderableHeight);
  
  // 5. Descargar la imagen del canvas
  const link = document.createElement('a');
  link.download = `captura-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

// --- INICIAR LA APLICACIÓN ---
main();