import { FaceDetector, FilesetResolver } from "./tasks-vision@0.10.0.js";

// --- 1. CONFIGURACIÓN Y CONSTANTES ---
const CONFIG = {
  minScore: 0.90,
  requiredConsecutiveFrames: 120,
  frontalThreshold: 0.7,
  detectionFontSize: '46px',
  successFontSize: '84px',
  showScore: false,
  showKeypoints: true,
  keypointSize: 20,
  roi: {
    x: 0.20,
    y: 0.15,
    width: 0.60,
    height: 0.70
  }
};

const DOM = {
  video: document.getElementById("webcam"),
  liveView: document.getElementById("liveView"),
  canvas: document.getElementById("snapshotCanvas")
};

const STATUS = {
    VALIDATING: (progress) => ({ text: `Validando su rostro... ${progress}%`, color: '#FFC300' }),
    VALIDATED: { text: 'Rostro validado', color: '#009933' },
    INVALID: { text: 'Enderece su rostro', color: '#CC3300' },
    NO_FACE: { text: 'Acérquese a la cámara', color: '#CC3300' }
};

let faceDetector;
let lastVideoTime = -1;
let detectionChildren = [];
let consecutiveFramesCounter = 0;
let appState = 'DETECTING';

// --- 2. INICIALIZACIÓN ---
async function main() {
  drawROI(); 

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
    .catch((err) => { console.error("No se pudo acceder a la webcam:", err); });
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

/**
 * Gestiona la lógica de detección, incluyendo la validación dentro del ROI.
 */
function handleDetections(detections) {
  clearDetections();

  // Reinicia el contador si no hay detecciones o si el rostro está fuera del ROI
  if (detections.length === 0 || !isFaceInROI(detections[0])) {
    consecutiveFramesCounter = 0;
    updateUIMessage(STATUS.NO_FACE); // Muestra "Acérquese a la cámara"
    return;
  }

  // --- El rostro está detectado Y dentro del ROI ---
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
    const progress = Math.round((consecutiveFramesCounter / CONFIG.requiredConsecutiveFrames) * 100);
    currentStatus = STATUS.VALIDATING(progress);
  } else {
    consecutiveFramesCounter = 0;
    currentStatus = STATUS.INVALID;
  }

  if (CONFIG.showScore) {
    if (currentStatus.text.includes('Score:')) {
        currentStatus.text = currentStatus.text.split(' (Score:')[0];
    }
    currentStatus.text += ` (Score: ${Math.round(score * 100)}%)`;
  }
  
  updateDetectionUI(detection, currentStatus); // Dibuja la caja y puntos solo si está en el ROI
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
    p.style.top = `${scaledOriginY - 60}px`;
    p.style.width = `${scaledWidth}px`;
    p.style.textAlign = 'center';
    p.style.fontSize = CONFIG.detectionFontSize;
    DOM.liveView.appendChild(p);
    detectionChildren.push(p);

    if (CONFIG.showKeypoints) {
      for (const k of det.keypoints) {
          const kp = document.createElement("span");
          kp.className = "key-point";
          const keypointX = DOM.video.clientWidth - (k.x * DOM.video.videoWidth * scale) - offsetX;
          const keypointY = (k.y * DOM.video.videoHeight * scale);
          
          kp.style.cssText = `
            width: ${CONFIG.keypointSize}px; height: ${CONFIG.keypointSize}px;
            left: ${keypointX - (CONFIG.keypointSize / 2)}px;
            top: ${keypointY - (CONFIG.keypointSize / 2)}px;
          `;
          DOM.liveView.appendChild(kp);
          detectionChildren.push(kp);
      }
    }
}

// --- 5. FUNCIONES AUXILIARES Y FINALIZACIÓN ---

function drawROI() {
  const roiBox = document.createElement("div");
  roiBox.className = "roi-box";
  roiBox.style.left = `${CONFIG.roi.x * 100}%`;
  roiBox.style.top = `${CONFIG.roi.y * 100}%`;
  roiBox.style.width = `${CONFIG.roi.width * 100}%`;
  roiBox.style.height = `${CONFIG.roi.height * 100}%`;
  DOM.liveView.appendChild(roiBox);
}

function isFaceInROI(detection) {
  const scale = DOM.video.clientHeight / DOM.video.videoHeight;
  const offsetX = (DOM.video.clientWidth - DOM.video.videoWidth * scale) / 2;
  
  const boxCenterX = DOM.video.clientWidth - (detection.boundingBox.originX + detection.boundingBox.width / 2) * scale - offsetX;
  const boxCenterY = (detection.boundingBox.originY + detection.boundingBox.height / 2) * scale;

  const roiPx = {
    x: DOM.video.clientWidth * CONFIG.roi.x,
    y: DOM.video.clientHeight * CONFIG.roi.y,
    width: DOM.video.clientWidth * CONFIG.roi.width,
    height: DOM.video.clientHeight * CONFIG.roi.height,
  };

  return (
    boxCenterX > roiPx.x &&
    boxCenterX < roiPx.x + roiPx.width &&
    boxCenterY > roiPx.y &&
    boxCenterY < roiPx.y + roiPx.height
  );
}

function updateUIMessage(status) {
  const existingMessage = DOM.liveView.querySelector('.detection-status');
  if (existingMessage) {
      DOM.liveView.removeChild(existingMessage);
      detectionChildren = detectionChildren.filter(child => child !== existingMessage);
  }

  const p = document.createElement("p");
  p.className = 'detection-status';
  p.innerText = status.text;
  p.style.backgroundColor = status.color;
  p.style.fontSize = CONFIG.detectionFontSize;
  p.style.textAlign = 'center';
  p.style.position = 'absolute';
  p.style.left = `${(CONFIG.roi.x + CONFIG.roi.width / 2) * 100}%`;
  p.style.top = `${(CONFIG.roi.y + CONFIG.roi.height / 2) * 100}%`;
  p.style.transform = 'translate(-50%, -50%)';
  p.style.padding = '10px';
  DOM.liveView.appendChild(p);
  detectionChildren.push(p);
}


function clearDetections() {
  for (let child of detectionChildren) {
    DOM.liveView.removeChild(child);
  }
  detectionChildren = [];
}

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
        padding: 20px;
        border-radius: 10px;
        font-size: ${CONFIG.successFontSize};
        z-index: 10;
    `;
    DOM.liveView.appendChild(p);
    console.log("Captura completada.");
}

function saveFrame() {
  const ctx = DOM.canvas.getContext("2d");
  const video = DOM.video;
  const canvas = DOM.canvas;

  canvas.width = video.clientWidth;
  canvas.height = video.clientHeight;

  const videoAspectRatio = video.videoWidth / video.videoHeight;
  const canvasAspectRatio = canvas.width / canvas.height;
  let renderableWidth, renderableHeight, xStart, yStart;

  if (videoAspectRatio < canvasAspectRatio) {
    renderableWidth = canvas.width;
    renderableHeight = renderableWidth / videoAspectRatio;
    xStart = 0;
    yStart = (canvas.height - renderableHeight) / 2;
  } else if (videoAspectRatio > canvasAspectRatio) {
    renderableHeight = canvas.height;
    renderableWidth = renderableHeight * videoAspectRatio;
    yStart = 0;
    xStart = (canvas.width - renderableWidth) / 2;
  } else {
    renderableHeight = canvas.height;
    renderableWidth = canvas.width;
    xStart = 0;
    yStart = 0;
  }

  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, xStart, yStart, renderableWidth, renderableHeight);
  
  const link = document.createElement('a');
  link.download = `captura-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.href = canvas.toDataURL("image/png");
  link.click();
}

// --- INICIAR LA APLICACIÓN ---
main();