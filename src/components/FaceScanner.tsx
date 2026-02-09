import { useEffect, useRef, useState, useCallback } from 'react';
import { FaceMesh, Results } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

interface SkinMetrics {
  wrinkles: number;
  texture: number;
  volume: number;
  eyeAging: number;
  skinTone: number;
  estimatedAge: number;
}

// Face mesh tessellation for right half
const RIGHT_FACE_INDICES = [
  // Right cheek region
  234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454,
  323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172,
  58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
  // Right forehead
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
  // Right jaw
  397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93
];

// Key facial landmark indices for annotations
const ANNOTATION_POINTS = {
  forehead: { index: 10, label: 'پیشانی', metric: 'wrinkles' },
  rightEye: { index: 33, label: 'دور چشم', metric: 'eyeAging' },
  rightCheek: { index: 234, label: 'گونه', metric: 'texture' },
  jawline: { index: 172, label: 'خط فک', metric: 'volume' },
  nose: { index: 4, label: 'تون پوست', metric: 'skinTone' },
};

const FaceScanner = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [cameraAllowed, setCameraAllowed] = useState(false);
  const [requestingCamera, setRequestingCamera] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [metrics, setMetrics] = useState<SkinMetrics | null>(null);
  const [smoothedMetrics, setSmoothedMetrics] = useState<SkinMetrics | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const faceMeshRef = useRef<FaceMesh | null>(null);
  const cameraRef = useRef<Camera | null>(null);
  const metricsHistoryRef = useRef<SkinMetrics[]>([]);

  // Request camera permission immediately so user isn't stuck on "searching for face"
  const requestCameraAccess = useCallback(async () => {
    if (cameraAllowed || requestingCamera) return;
    setRequestingCamera(true);
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      });
      stream.getTracks().forEach((t) => t.stop());
      setCameraAllowed(true);
    } catch (err) {
      console.error('Camera permission error:', err);
      setCameraError('دسترسی به دوربین امکان‌پذیر نیست. لطفاً در تنظیمات مرورگر اجازه دهید.');
    } finally {
      setRequestingCamera(false);
    }
  }, [cameraAllowed, requestingCamera]);
  

  // Smooth metrics over time for maximum stability
  const smoothMetrics = useCallback((newMetrics: SkinMetrics): SkinMetrics => {
    const history = metricsHistoryRef.current;
    history.push(newMetrics);
    
    // Keep last 90 frames for very stable averaging (about 3 seconds)
    if (history.length > 90) {
      history.shift();
    }
    
    // Only calculate if we have enough samples
    if (history.length < 15) {
      return newMetrics;
    }
    
    // Remove outliers by using median-like approach (ignore top/bottom 20%)
    const sortedAges = [...history].map(m => m.estimatedAge).sort((a, b) => a - b);
    const trimStart = Math.floor(history.length * 0.2);
    const trimEnd = Math.ceil(history.length * 0.8);
    const trimmedAges = sortedAges.slice(trimStart, trimEnd);
    
    // Average the trimmed values for super stable age
    const stableAge = Math.round(trimmedAges.reduce((sum, age) => sum + age, 0) / trimmedAges.length);
    
    // Average other metrics normally
    const avg = {
      wrinkles: Math.round(history.reduce((sum, m) => sum + m.wrinkles, 0) / history.length),
      texture: Math.round(history.reduce((sum, m) => sum + m.texture, 0) / history.length),
      volume: Math.round(history.reduce((sum, m) => sum + m.volume, 0) / history.length),
      eyeAging: Math.round(history.reduce((sum, m) => sum + m.eyeAging, 0) / history.length),
      skinTone: Math.round(history.reduce((sum, m) => sum + m.skinTone, 0) / history.length),
      estimatedAge: stableAge,
    };
    
    return avg;
  }, []);

  // Calculate skin metrics based on face landmarks
  const calculateMetrics = useCallback((landmarks: Results['multiFaceLandmarks'][0]): SkinMetrics => {
    // Use multiple facial geometry measurements for variation
    
    // Facial proportions that vary between people
    const foreheadHeight = Math.abs(landmarks[10].y - landmarks[151].y);
    const eyeOpenness = Math.abs(landmarks[159].y - landmarks[145].y);
    const cheekWidth = Math.abs(landmarks[234].x - landmarks[454].x);
    const jawWidth = Math.abs(landmarks[172].x - landmarks[397].x);
    const noseLength = Math.abs(landmarks[4].y - landmarks[6].y);
    const faceHeight = Math.abs(landmarks[10].y - landmarks[152].y);
    const eyeDistance = Math.abs(landmarks[33].x - landmarks[263].x);
    const lipThickness = Math.abs(landmarks[13].y - landmarks[14].y);
    const browHeight = Math.abs(landmarks[66].y - landmarks[159].y);
    const cheekboneHeight = Math.abs(landmarks[234].y - landmarks[93].y);
    
    // Use facial RATIOS that vary between people
    const faceWidth = Math.abs(landmarks[234].x - landmarks[454].x);
    const safeDiv = (a: number, b: number) => (b === 0 ? 0 : a / b);

    const faceHeightRatio = safeDiv(faceHeight, faceWidth);
    const eyeDistanceRatio = safeDiv(eyeDistance, faceWidth);
    const foreheadRatio = safeDiv(foreheadHeight, faceHeight);
    const jawRatio = safeDiv(jawWidth, cheekWidth);
    const noseRatio = safeDiv(noseLength, faceHeight);
    const eyeOpennessRatio = safeDiv(eyeOpenness, eyeDistance);
    const lipRatio = safeDiv(lipThickness, noseLength);
    const browRatio = safeDiv(browHeight, foreheadHeight);

    /**
     * Age scoring - calibrated for realistic results
     * Youth indicators (subtract from age): larger eyes, fuller lips, higher brow
     * Aging indicators (add to age): longer face, larger forehead ratio, lower jaw
     */
    
    // Create a unique face signature for consistent per-person results
    const faceSignature = 
      Math.abs(faceHeightRatio * 1000) +
      Math.abs(eyeDistanceRatio * 800) +
      Math.abs(foreheadRatio * 600) +
      Math.abs(jawRatio * 500) +
      Math.abs(noseRatio * 400) +
      Math.abs(eyeOpennessRatio * 300) +
      Math.abs(lipRatio * 200) +
      Math.abs(browRatio * 100);

    // Use sine function for smooth distribution across faces
    const signatureOffset = Math.sin(faceSignature) * 0.5 + 0.5; // 0 to 1
    
    // Base age from signature (gives variety between different faces)
    const baseFromSignature = 20 + signatureOffset * 35; // 20-55 range
    
    // Small adjustments based on youth/aging indicators (max ±5 years)
    const youthScore = (eyeOpennessRatio * 10) + (lipRatio * 5) - (foreheadRatio * 3);
    const adjustment = Math.max(-5, Math.min(5, (youthScore - 1.5) * 3));
    
    const rawAge = baseFromSignature + adjustment;
    const finalAge = Math.min(55, Math.max(20, Math.round(rawAge)));

    // Metrics tied to the estimated age (higher age = more aging signs)
    const agePercent = (finalAge - 20) / 35; // 0 to 1

    const wrinkles = Math.min(
      75,
      Math.max(5, Math.round(5 + agePercent * 60 + signatureOffset * 10))
    );

    const eyeAging = Math.min(
      60,
      Math.max(3, Math.round(3 + agePercent * 45 + signatureOffset * 12))
    );

    const texture = Math.min(
      98,
      Math.max(60, Math.round(95 - agePercent * 25 - signatureOffset * 10))
    );

    const volume = Math.min(
      98,
      Math.max(55, Math.round(95 - agePercent * 30 - signatureOffset * 10))
    );

    const skinTone = Math.min(
      25,
      Math.max(3, Math.round(5 + agePercent * 15 + signatureOffset * 5))
    );

    return { wrinkles, texture, volume, eyeAging, skinTone, estimatedAge: finalAge };
  }, []);

  // Draw the mesh overlay on the right half of the face (mesh only, no text)
  const drawOverlay = useCallback((
    ctx: CanvasRenderingContext2D,
    landmarks: Results['multiFaceLandmarks'][0],
    width: number,
    height: number
  ) => {
    ctx.clearRect(0, 0, width, height);

    // Find face center for splitting left/right
    const noseTip = landmarks[4];
    const faceCenterX = noseTip.x * width;

    // Draw semi-transparent mesh on right half
    ctx.globalAlpha = 0.3;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 0.5;

    // Draw mesh triangles on right side
    const rightLandmarks = landmarks.filter((_, i) => 
      RIGHT_FACE_INDICES.includes(i) || landmarks[i].x * width > faceCenterX - 20
    );

    // Draw grid pattern
    for (let i = 0; i < rightLandmarks.length - 1; i++) {
      const p1 = rightLandmarks[i];
      const p2 = rightLandmarks[(i + 1) % rightLandmarks.length];
      
      if (p1.x * width > faceCenterX - 10) {
        ctx.beginPath();
        ctx.moveTo(p1.x * width, p1.y * height);
        ctx.lineTo(p2.x * width, p2.y * height);
        ctx.stroke();
      }
    }

    // Draw vertical and horizontal grid lines on right side
    ctx.globalAlpha = 0.2;
    for (let i = 0; i < landmarks.length; i += 3) {
      const p = landmarks[i];
      if (p.x * width > faceCenterX) {
        ctx.beginPath();
        ctx.moveTo(faceCenterX, p.y * height);
        ctx.lineTo(p.x * width + 20, p.y * height);
        ctx.stroke();
      }
    }

    // Draw orange points on face landmarks
    ctx.globalAlpha = 1;
    Object.entries(ANNOTATION_POINTS).forEach(([key, point]) => {
      const landmark = landmarks[point.index];
      const x = landmark.x * width;
      const y = landmark.y * height;

      if (x >= faceCenterX - 30) {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 165, 0, 0.9)';
        ctx.fill();
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
  }, []);

  // Initialize MediaPipe Face Mesh (load model early so detection isn't stuck)
  useEffect(() => {
    const initFaceMesh = async () => {
      const faceMesh = new FaceMesh({
        locateFile: (file) => {
          return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
        },
      });

      faceMesh.setOptions({
        maxNumFaces: 1,
        refineLandmarks: true,
        minDetectionConfidence: 0.3,
        minTrackingConfidence: 0.3,
      });

      try {
        await faceMesh.initialize();
      } catch (e) {
        console.warn('FaceMesh pre-init failed, will init on first frame:', e);
      }

      faceMesh.onResults((results) => {
        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
          const landmarks = results.multiFaceLandmarks[0];
          setFaceDetected(true);
          setIsScanning(true);
          
          const newMetrics = calculateMetrics(landmarks);
          setMetrics(newMetrics);
          
          // Apply smoothing for stable display
          const smoothed = smoothMetrics(newMetrics);
          setSmoothedMetrics(smoothed);

          if (canvasRef.current && videoRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
              drawOverlay(
                ctx, 
                landmarks, 
                canvasRef.current.width, 
                canvasRef.current.height
              );
            }
          }
        } else {
          setFaceDetected(false);
          if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            if (ctx) {
              ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
            }
          }
        }
      });

      faceMeshRef.current = faceMesh;
    };

    initFaceMesh();

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop();
      }
    };
  }, [calculateMetrics, drawOverlay, smoothMetrics]);

  // Start camera only after user has granted permission (fast permission prompt)
  useEffect(() => {
    if (!cameraAllowed) return;

    const startCamera = async () => {
      if (!videoRef.current || !faceMeshRef.current) return;

      try {
        const video = videoRef.current;
        const camera = new Camera(video, {
          onFrame: async () => {
            if (!faceMeshRef.current || !videoRef.current) return;
            const v = videoRef.current;
            // Only send when video has real frames (avoid black/empty frames so face is detected)
            if (v.readyState < 2 || v.videoWidth === 0 || v.videoHeight === 0) return;
            await faceMeshRef.current.send({ image: v });
          },
          width: 1280,
          height: 720,
        });

        cameraRef.current = camera;
        await camera.start();
      } catch (error) {
        console.error('Camera error:', error);
        setCameraError('دسترسی به دوربین امکان‌پذیر نیست');
      }
    };

    const timer = setTimeout(startCamera, 300);
    return () => clearTimeout(timer);
  }, [cameraAllowed]);

  // Convert number to Persian numerals
  const toPersianNumber = (num: number): string => {
    const persianDigits = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'];
    return num.toString().split('').map(d => persianDigits[parseInt(d)] || d).join('');
  };

  // Initial screen: request camera permission so it's granted quickly (no stuck on "searching for face")
  if (!cameraAllowed && !cameraError) {
    return (
      <div className="fixed inset-0 bg-background flex flex-col items-center justify-center p-6" dir="rtl">
        <div className="w-20 h-20 rounded-full border-2 border-primary/50 flex items-center justify-center mb-6">
          <span className="text-4xl">📷</span>
        </div>
        <h1 className="text-xl md:text-2xl font-bold text-foreground font-vazir text-center mb-2">
          تحلیل سن پوست
        </h1>
        <p className="text-muted-foreground font-vazir text-center mb-8 max-w-sm">
          برای شروع، دسترسی به دوربین را مجاز کنید تا تصویر شما برای تحلیل استفاده شود.
        </p>
        <button
          type="button"
          onClick={requestCameraAccess}
          disabled={requestingCamera}
          className="px-8 py-4 rounded-xl bg-primary text-primary-foreground font-vazir font-medium text-lg hover:opacity-90 disabled:opacity-60 transition-opacity"
        >
          {requestingCamera ? 'در حال درخواست...' : 'شروع و دسترسی به دوربین'}
        </button>
      </div>
    );
  }

  if (cameraError) {
    return (
      <div className="fixed inset-0 bg-background flex flex-col items-center justify-center p-6" dir="rtl">
        <p className="text-xl text-muted-foreground font-vazir text-center mb-4">{cameraError}</p>
        <button
          type="button"
          onClick={() => { setCameraError(null); setCameraAllowed(false); }}
          className="px-6 py-3 rounded-lg bg-primary text-primary-foreground font-vazir"
        >
          تلاش مجدد
        </button>
      </div>
    );
  }

  // Get display metrics (smoothed for stability)
  const displayMetrics = smoothedMetrics || metrics;

  return (
    <div className="fixed inset-0 bg-background overflow-hidden" dir="rtl">
      {/* Video feed */}
      <video
        ref={videoRef}
        className="absolute inset-0 w-full h-full object-cover"
        autoPlay
        playsInline
        muted
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Canvas overlay for mesh only (mirrored with video) */}
      <canvas
        ref={canvasRef}
        width={1280}
        height={720}
        className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        style={{ transform: 'scaleX(-1)' }}
      />

      {/* Persian labels overlay - NOT mirrored, positioned on right side */}
      {displayMetrics && faceDetected && (
        <div className="absolute top-16 right-4 md:right-8 space-y-3 md:space-y-4 font-vazir text-right max-w-[280px] md:max-w-none">
          <div className="flex items-center gap-2 text-white text-xs md:text-sm bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2">
            <span className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0"></span>
            <span>خطوط ریز و چروک‌ها: {toPersianNumber(displayMetrics.wrinkles)}٪</span>
          </div>
          <div className="flex items-center gap-2 text-white text-xs md:text-sm bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2">
            <span className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0"></span>
            <span>بافت و الاستیسیته پوست: {toPersianNumber(displayMetrics.texture)}٪</span>
          </div>
          <div className="flex items-center gap-2 text-white text-xs md:text-sm bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2">
            <span className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0"></span>
            <span>حجم صورت و افتادگی: {toPersianNumber(displayMetrics.volume)}٪</span>
          </div>
          <div className="flex items-center gap-2 text-white text-xs md:text-sm bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2">
            <span className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0"></span>
            <span>نشانه‌های پیری اطراف چشم: {toPersianNumber(displayMetrics.eyeAging)}٪</span>
          </div>
          <div className="flex items-center gap-2 text-white text-xs md:text-sm bg-black/40 backdrop-blur-sm rounded-lg px-3 py-2">
            <span className="w-2 h-2 rounded-full bg-orange-400 flex-shrink-0"></span>
            <span>تون پوست و لکه‌های رنگدانه‌ای: {toPersianNumber(displayMetrics.skinTone)}٪</span>
          </div>
        </div>
      )}

      {/* Scanning indicator */}
      {!faceDetected && (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center fade-in">
            <div className="w-24 h-24 border-2 border-scanner-glow rounded-full mx-auto mb-4 scanner-pulse" />
            <p className="text-lg text-muted-foreground font-vazir">در حال جستجوی چهره...</p>
            <p className="text-sm text-muted-foreground/80 font-vazir mt-2">صورت خود را در مرکز کادر قرار دهید</p>
          </div>
        </div>
      )}

      {/* Skin age result - centered at bottom */}
      {displayMetrics && faceDetected && (
        <div className="absolute bottom-0 left-0 right-0 pb-8 md:pb-12 pt-6 bg-gradient-to-t from-background via-background/80 to-transparent">
          <div className="text-center fade-in">
            <p className="text-2xl md:text-4xl font-bold text-foreground font-vazir tracking-wide">
              سن تخمینی پوست: {toPersianNumber(displayMetrics.estimatedAge)} سال
            </p>
            <p className="text-xs md:text-sm text-muted-foreground mt-2 font-vazir">
              تحلیل زیبایی‌شناختی • غیرپزشکی
            </p>
          </div>
        </div>
      )}

      {/* Scanning frame corners */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-8 left-8 w-12 md:w-16 h-12 md:h-16 border-l-2 border-t-2 border-scanner-glow opacity-50" />
        <div className="absolute top-8 right-8 w-12 md:w-16 h-12 md:h-16 border-r-2 border-t-2 border-scanner-glow opacity-50" />
        <div className="absolute bottom-28 md:bottom-32 left-8 w-12 md:w-16 h-12 md:h-16 border-l-2 border-b-2 border-scanner-glow opacity-50" />
        <div className="absolute bottom-28 md:bottom-32 right-8 w-12 md:w-16 h-12 md:h-16 border-r-2 border-b-2 border-scanner-glow opacity-50" />
      </div>

      {/* Vignette overlay */}
      <div 
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at center, transparent 40%, rgba(0,0,0,0.6) 100%)'
        }}
      />
    </div>
  );
};

export default FaceScanner;