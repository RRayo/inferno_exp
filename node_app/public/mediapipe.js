import { FaceDetector, FilesetResolver } from "./tasks-vision@0.10.0.js";

// --- ESTADO DE LA APLICACIÓN ---
let faceDetector;
let lastVideoTime = -1;
let state = {
  canSave: true,
  consecutiveFrames: 0,
  isPhotoTaken: false,
  requiredFrames: 15 // Ajusta la estabilidad aquí
};

// --- REFERENCIAS A ELEMENTOS DEL DOM (se obtienen una sola vez) ---
const elements = {
  video: document.getElementById("webcam"),
  liveView: document.getElementById("liveView"),
  uiContainer: document.getElementById("uiContainer"),
  overlayContainer: document.getElementById("overlayContainer"),
  roiGuide: document.getElementById("roiGuide"),
  bbox: document.getElementById("bbox"),
  statusText: document.getElementById("statusText"),
  keypoints: Array.from({length: 6}, (_, i) => document.getElementById(`kp-${i}`))
};

// --- LÓGICA PRINCIPAL ---

async function initializeApp() {
  const vision = await FilesetResolver.forVisionTasks("./task_vision");
  faceDetector = await FaceDetector.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "blaze_face_short_range.tflite", delegate: "GPU" },
    runningMode: "VIDEO"
  });
  await startWebcam();
}

async function startWebcam() {
  if (!navigator.mediaDevices?.getUserMedia) {
    console.error("getUserMedia() is not supported by your browser");
    return;
  }
  const constraints = { video: { width: { ideal: 1280 }, height: { ideal: 720 } } };
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    elements.video.srcObject = stream;
    elements.video.addEventListener("loadeddata", predictWebcam);
  } catch (err) {
    console.error("Error al acceder a la webcam:", err);
  }
}

function predictWebcam() {
  if (state.isPhotoTaken) {
    showFinalMessage();
    return; // Detener el bucle si ya se tomó la foto
  }

  if (elements.video.currentTime !== lastVideoTime) {
    lastVideoTime = elements.video.currentTime;
    const detections = faceDetector.detectForVideo(elements.video, performance.now())?.detections;

    if (detections && detections.length > 0) {
      // Solo procesamos la detección más probable (la primera)
      displayDetections(detections[0]);
    } else {
      hideOverlays();
    }
  }
  requestAnimationFrame(predictWebcam);
}

function displayDetections(det) {
  const { liveView, video } = elements;
  
  // 1. Definir y posicionar la Región de Interés (ROI)
  const roi = {
    width: liveView.clientWidth * 0.5,
    height: liveView.clientHeight * 0.7,
    x: (liveView.clientWidth * 0.25),
    y: (liveView.clientHeight * 0.15)
  };
  const roiGuideStyle = elements.roiGuide.style;
  roiGuideStyle.width = `${roi.width}px`;
  roiGuideStyle.height = `${roi.height}px`;
  roiGuideStyle.left = `${roi.x}px`;
  roiGuideStyle.top = `${roi.y}px`;

  // 2. Calcular factores de escala para las superposiciones
  const videoRatio = video.videoWidth / video.videoHeight;
  const viewRatio = liveView.clientWidth / liveView.clientHeight;
  let scale = 1, offsetX = 0, offsetY = 0;
  if (videoRatio > viewRatio) {
    scale = liveView.clientHeight / video.videoHeight;
    offsetX = (liveView.clientWidth - video.videoWidth * scale) / 2;
  } else {
    scale = liveView.clientWidth / video.videoWidth;
    offsetY = (liveView.clientHeight - video.videoHeight * scale) / 2;
  }
  
  // 3. Calcular la posición en pantalla del rostro detectado
  const scaledWidth = det.boundingBox.width * scale;
  const scaledOriginX = det.boundingBox.originX * scale;
  const scaledOriginY = det.boundingBox.originY * scale;
  const boxCenterX = liveView.clientWidth - scaledOriginX - (scaledWidth / 2) - offsetX;
  const boxCenterY = scaledOriginY + (det.boundingBox.height * scale / 2) + offsetY;

  // 4. Comprobar si el rostro está dentro del ROI
  const isInRoi = (
    boxCenterX > roi.x && boxCenterX < roi.x + roi.width &&
    boxCenterY > roi.y && boxCenterY < roi.y + roi.height
  );

  const score = det.categories[0].score;
  const isFrontal = isFacingForward(det.keypoints);

  // 5. Activar contador solo si todas las condiciones se cumplen
  if (isInRoi && score > 0.90 && isFrontal) {
    state.consecutiveFrames++;
  } else {
    state.consecutiveFrames = 0;
  }

  // 6. Guardar la foto si se alcanza el umbral de estabilidad
  if (state.consecutiveFrames >= state.requiredFrames && state.canSave) {
    state.canSave = false;
    state.isPhotoTaken = true;
    saveFrame(det);
    return;
  }
  
  // 7. Actualizar la información visual en cada frame
  updateOverlays(det, { scale, offsetX, offsetY });
}

// --- FUNCIONES DE AYUDA (Helpers) ---

function updateOverlays(det, scaling) {
  elements.overlayContainer.style.display = "block";
  const { scale, offsetX, offsetY } = scaling;
  const { video, liveView, bbox, statusText, keypoints } = elements;

  // Actualizar Bounding Box
  const scaledWidth = det.boundingBox.width * scale;
  const scaledHeight = det.boundingBox.height * scale;
  const scaledOriginX = det.boundingBox.originX * scale;
  const scaledOriginY = det.boundingBox.originY * scale;
  const bboxStyle = bbox.style;
  bboxStyle.left = `${liveView.clientWidth - scaledOriginX - scaledWidth - offsetX}px`;
  bboxStyle.top = `${scaledOriginY + offsetY}px`;
  bboxStyle.width = `${scaledWidth}px`;
  bboxStyle.height = `${scaledHeight}px`;
  bboxStyle.borderColor = state.consecutiveFrames > 0 ? "#00FF00" : "#FFFFFF";
  
  // Actualizar Texto de Estado
  let text = "";
  if (state.consecutiveFrames > 0) {
    const progress = Math.round((state.consecutiveFrames / state.requiredFrames) * 100);
    text = `Mantén la posición... ${progress}%`;
    statusText.style.backgroundColor = "#E67E22";
  } else {
    text = "Ubica tu rostro en el óvalo";
    statusText.style.backgroundColor = "#CC3300";
  }
  statusText.innerText = text;
  statusText.style.left = bboxStyle.left;
  statusText.style.top = `${parseFloat(bboxStyle.top) - 45}px`;
  statusText.style.width = bboxStyle.width;
  
  // Actualizar Keypoints
  det.keypoints.forEach((k, i) => {
    const kpStyle = keypoints[i].style;
    kpStyle.left = `${liveView.clientWidth - (k.x * video.videoWidth * scale) - offsetX - 3}px`;
    kpStyle.top = `${(k.y * video.videoHeight * scale) + offsetY - 3}px`;
  });
}

function hideOverlays() {
  elements.overlayContainer.style.display = "none";
  state.consecutiveFrames = 0;
}

function showFinalMessage() {
    hideOverlays();
    elements.uiContainer.innerHTML = ""; // Limpiar
    const p = document.createElement("p");
    p.innerText = "¡Captura completada!";
    p.style.cssText = `
        background-color: #009933; color: white; position: absolute;
        top: 20px; left: 50%; transform: translateX(-50%);
        padding: 12px 20px; border-radius: 8px; font-size: 18px;
    `;
    elements.uiContainer.appendChild(p);
}

function isFacingForward(keypoints) {
  const [leftEye, rightEye, noseTip] = keypoints;
  const distNoseToLeftEye = Math.abs(noseTip.x - leftEye.x);
  const distNoseToRightEye = Math.abs(noseTip.x - rightEye.x);
  if (distNoseToLeftEye === 0 || distNoseToRightEye === 0) return false;
  const ratio = Math.min(distNoseToLeftEye, distNoseToRightEye) / Math.max(distNoseToLeftEye, distNoseToRightEye);
  return ratio > 0.7; // Umbral de pose frontal
}

function saveFrame(det) {
  const frameCaptureCanvas = document.getElementById("frameCaptureCanvas");
  const frameCtx = frameCaptureCanvas.getContext("2d");
  frameCaptureCanvas.width = elements.video.videoWidth;
  frameCaptureCanvas.height = elements.video.videoHeight;
  frameCtx.drawImage(elements.video, 0, 0, frameCaptureCanvas.width, frameCaptureCanvas.height);
  
  const snapshotCanvas = document.getElementById("snapshotCanvas");
  const ctx = snapshotCanvas.getContext("2d");
  
  // Versión de guardado limpio (sin dibujos) para máxima calidad de imagen
  snapshotCanvas.width = frameCaptureCanvas.width;
  snapshotCanvas.height = frameCaptureCanvas.height;
  ctx.drawImage(frameCaptureCanvas, 0, 0); // Dibuja el frame capturado
  const link = document.createElement('a');
  link.download = `captura-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
  link.href = snapshotCanvas.toDataURL("image/png");
  link.click();
  console.log("Frame guardado con éxito.");
}

// Iniciar la aplicación
initializeApp();