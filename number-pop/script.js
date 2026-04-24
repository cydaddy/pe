/**
 * Number Pop Game Logic
 * Uses MediaPipe/TensorFlow.js to track body keypoints.
 * Synthesizes sounds via Web Audio API. 
 */

// Basic State
let state = 'LOADING'; // LOADING, MENU, CALIBRATION, PLAYING, RESULT
let targetNumber = 10;
let lastFrameTime = 0;
let colorHintEnabled = true;
let timerEnabled = true;

// Game State Tracking
class PlayerState {
    constructor(id) {
        this.id = id;
        this.progress = 0; // Current target number (started from 1 to targetNumber)
        this.startTime = 0;
        this.elapsedTime = 0;
        this.finished = false;
        this.finishTime = 0;
        
        // Calibration
        this.isCalibrated = false;
        this.baselineY = 0;
        this.circles = []; // Currently assigned circles to pop
        
        // Warnings
        this.jumpWarningFrames = 0;
        this.isReady = false; // When hands are clapped during calibration
        
        // UI Bindings
        this.ui = {
            container: document.getElementById(`player${id}-ui`),
            time: document.getElementById(`p${id}-time`),
            progress: document.getElementById(`p${id}-progress`),
            warning: document.getElementById(`center-warning`),
            status: document.getElementById(`p${id}-status`)
        };
    }
    
    reset() {
        this.progress = 0;
        this.elapsedTime = 0;
        this.finished = false;
        this.finishTime = 0;
        this.circles = [];
        this.jumpWarningFrames = 0;
        this.isReady = false;
        if(this.ui.status) this.ui.status.innerText = '';
        this.updateUI();
    }
    
    updateUI() {
        if(!this.ui.container) return;
        this.ui.progress.innerText = `${this.progress} / ${targetNumber}`;
        
        // Update Time (MM:SS.ms)
        if (timerEnabled) {
            this.ui.time.style.display = 'block';
            let timeToDisplay = this.finished ? this.finishTime : this.elapsedTime;
            let ms = timeToDisplay % 1000;
            let s = Math.floor((timeToDisplay / 1000) % 60);
            let m = Math.floor(timeToDisplay / 60000);
            this.ui.time.innerText = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${Math.floor(ms/10).toString().padStart(2, '0')}`;
        } else {
            this.ui.time.style.display = 'none';
        }
        
        if (this.jumpWarningFrames > 0) {
            this.ui.warning.classList.remove('hidden');
            this.jumpWarningFrames--;
        } else {
            this.ui.warning.classList.add('hidden');
        }
    }
}

let p1 = new PlayerState(1);

// TFJS & Camera Variables
let video, canvas, ctx;
let detector;
let currentModel = null; // tracks which pose model is active (MoveNet or BlazePose)

// Audio Context
let audioCtx;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function playTone(freq, type, duration, vol=0.5) {
    if (!audioCtx) return;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    
    gain.gain.setValueAtTime(vol, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);
    
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
}

const sounds = {
    pop: () => {
        playTone(600, 'sine', 0.1, 0.7);
        setTimeout(() => playTone(800, 'sine', 0.1, 0.7), 50);
    },
    error: () => playTone(150, 'sawtooth', 0.3, 0.8),
    clap: () => {
        playTone(1000, 'square', 0.1, 0.2);
        playTone(1200, 'sine', 0.1, 0.3);
    },
    win: () => {
        playTone(400, 'sine', 0.2, 0.5);
        setTimeout(() => playTone(500, 'sine', 0.2, 0.5), 150);
        setTimeout(() => playTone(600, 'sine', 0.4, 0.5), 300);
    }
};

// UI Elements
const els = {
    overlayLoading: document.getElementById('loading-overlay'),
    overlayStart: document.getElementById('start-menu'),
    overlayResult: document.getElementById('result-overlay'),
    inGameUI: document.getElementById('in-game-ui'),
    centerMsg: document.getElementById('center-message'),
    btnSolo: document.getElementById('btn-hint-on'),
    btnPvp: document.getElementById('btn-hint-off'),
    btnTimerOn: document.getElementById('btn-timer-on'),
    btnTimerOff: document.getElementById('btn-timer-off'),
    inpTarget: document.getElementById('target-number'),
    btnStart: document.getElementById('start-btn'),
    btnRestart: document.getElementById('restart-btn'),
    resTitle: document.getElementById('result-title'),
    resDetails: document.getElementById('result-details'),
    retryBtn: document.getElementById('retry-init-btn'),
    loader: document.querySelector('.loader')
};

let cameraReady = false;
let modelReady = false;
let renderLoopStarted = false;
let resizeListenerAdded = false;

// 성능 최적화: 추론과 렌더링 분리
// lastPoses: 추론 루프가 채워넣는 최신 AI 결과
// displayPoses: 렌더 루프에서 lerp 연산된 표시용 포즈
let lastPoses = [];
let displayPoses = [];
let inferLoopStarted = false;

// 키포인트 보간: 새 포즈 생길 때마다 60fps로 부드럽게 이동
// alpha=1: 즉시반영 (laggy 하지만 정확), alpha=0.3: 여운 후행 (smooth)
const LERP_ALPHA = 0.2; // 크롬북의 지터(Jitter)를 잡기 위해 0.4에서 0.2로 낮춤

function lerpPoses(current, target) {
    if (!current || current.length !== target.length) return target;
    return target.map((pose, pi) => {
        const prev = current[pi];
        if (!prev) return pose;
        return {
            ...pose,
            keypoints: pose.keypoints.map((kp, ki) => {
                const pkp = prev.keypoints[ki];
                if (!pkp || kp.score < 0.2) return kp;
                return {
                    ...kp,
                    x: pkp.x + (kp.x - pkp.x) * LERP_ALPHA,
                    y: pkp.y + (kp.y - pkp.y) * LERP_ALPHA,
                };
            })
        };
    });
}

async function setupCamera() {
    video = document.getElementById('video');
    
    // Check if we are on a secure origin or localhost
    const isSecureOrigin = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!isSecureOrigin) {
        throw new Error('INSECURE_ORIGIN');
    }

    // If camera stream already exists, stop and release it first
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': { 
                width: { ideal: 320 },   // 크롬북 성능 최적화를 위해 해상도를 절반(320x240)으로 더 낮춤
                height: { ideal: 240 }, 
                facingMode: 'user' 
            }
        });
        video.srcObject = stream;
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                video.play();
                resolve(video);
            };
        });
    } catch (err) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
            throw new Error('PERMISSION_DENIED');
        } else {
            throw err;
        }
    }
}

async function loadModel() {
    // Pre-check: make sure CDN scripts actually loaded
    if (typeof tf === 'undefined') {
        throw new Error('SCRIPT_LOAD_ERROR: @tensorflow/tfjs 스크립트를 불러오지 못했습니다.');
    }
    if (typeof poseDetection === 'undefined') {
        throw new Error('SCRIPT_LOAD_ERROR: pose-detection 스크립트를 불러오지 못했습니다.');
    }

    // Initialize TF backend (WebGL → CPU fallback)
    try {
        await tf.setBackend('webgl');
        await tf.ready();
        console.log('TF 백엔드: webgl');
    } catch (backendErr) {
        console.warn('WebGL 실패, CPU로 전환:', backendErr);
        await tf.setBackend('cpu');
        await tf.ready();
        console.log('TF 백엔드: cpu');
    }

    // 1차 시도: MoveNet (tfhub.dev에서 모델 로딩)
    try {
        const config = { modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING };
        detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, config);
        currentModel = poseDetection.SupportedModels.MoveNet;
        console.log('모델: MoveNet MULTIPOSE');
        return; // 성공
    } catch (moveNetErr) {
        console.warn('MoveNet 로딩 실패 (tfhub.dev 차단 가능성):', moveNetErr.message);
    }

    // 2차 시도: BlazePose via MediaPipe (jsDelivr CDN — 학교 방화벽 우회)
    // jsDelivr는 npm 패키지 CDN으로 방화벽에서 허용되는 경우가 많습니다.
    try {
        detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.BlazePose,
            {
                runtime: 'mediapipe',
                solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404',
                modelType: 'lite'
            }
        );
        currentModel = poseDetection.SupportedModels.BlazePose;
        console.log('모델: BlazePose (MediaPipe 폴백)');
    } catch (blazeErr) {
        console.error('BlazePose도 실패:', blazeErr);
        throw new Error('MODEL_LOAD_ERROR: MoveNet와 BlazePose 모두 실패. ' + blazeErr.message);
    }
}

async function init() {
    const loadingH2 = els.overlayLoading.querySelector('h2');
    const loadingP = els.overlayLoading.querySelector('p');

    // Reset UI for loading
    els.overlayLoading.classList.remove('hidden');
    els.loader.classList.remove('hidden');
    els.retryBtn.classList.add('hidden');
    loadingH2.innerText = "카메라와 AI 모델을 불러오는 중...";
    loadingP.innerText = "잠시만 기다려주세요.";

    try {
        // Step 1: Camera (skip if already connected)
        if (!cameraReady) {
            await setupCamera();
            cameraReady = true;
        }
        
        // Canvas setup (only once)
        if (!canvas) {
            canvas = document.getElementById('game-canvas');
            ctx = canvas.getContext('2d');
            resizeCanvas();
        }
        if (!resizeListenerAdded) {
            window.addEventListener('resize', resizeCanvas);
            resizeListenerAdded = true;
        }
        
        // Step 2: Model (skip if already loaded)
        if (!modelReady) {
            loadingH2.innerText = "AI 모델 데이터 로딩 중...";
            await loadModel();
            modelReady = true;
        }
        
        // Hide loading, show menu
        els.overlayLoading.classList.add('hidden');
        els.overlayStart.classList.remove('hidden');
        state = 'MENU';
        
        // Start render & inference loops only once
        if (!renderLoopStarted) {
            renderLoopStarted = true;
            requestAnimationFrame(renderLoop); // 렌더링: 60fps, 추론 미포함
            inferLoop();                       // 추론: 모델 속도로 독립 실행
        }
    } catch (e) {
        console.error(e);
        loadingH2.innerText = "설정이 필요합니다";
        els.loader.classList.add('hidden');
        els.retryBtn.classList.remove('hidden');
        
        if (e.message === 'INSECURE_ORIGIN') {
            loadingP.innerHTML = "보안 연결(HTTPS)이 필요합니다.<br>로컬 주소(localhost)나 보안 서버에서 실행해주세요.";
        } else if (e.message === 'PERMISSION_DENIED') {
            loadingP.innerText = "카메라 사용 권한이 거부되었습니다. 브라우저 주소창 왼쪽의 자물쇠 아이콘을 클릭해 권한을 허용해주세요.";
        } else if (e.message.startsWith('SCRIPT_LOAD_ERROR')) {
            loadingP.innerHTML = "스크립트를 불러오지 못했습니다. 인터넷 연결을 확인해주세요.<br><small style='opacity:0.7'>" + e.message + "</small>";
        } else if (e.message.startsWith('MODEL_LOAD_ERROR')) {
            // Show the real error detail to help diagnose
            const detail = e.message.replace('MODEL_LOAD_ERROR: ', '');
            loadingP.innerHTML = "AI 모델 초기화 실패.<br><small style='word-break:break-all;opacity:0.7'>" + detail + "</small>";
        } else {
            loadingP.innerHTML = "오류: <small style='word-break:break-all;opacity:0.7'>" + e.message + "</small>";
        }
    }
}

els.retryBtn.onclick = init;

function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

// Menu Bindings
els.btnSolo.onclick = () => {
    colorHintEnabled = true;
    els.btnSolo.classList.add('active');
    els.btnPvp.classList.remove('active');
};
els.btnPvp.onclick = () => {
    colorHintEnabled = false;
    els.btnSolo.classList.remove('active');
    els.btnPvp.classList.add('active');
};
els.btnTimerOn.onclick = () => {
    timerEnabled = true;
    els.btnTimerOn.classList.add('active');
    els.btnTimerOff.classList.remove('active');
};
els.btnTimerOff.onclick = () => {
    timerEnabled = false;
    els.btnTimerOn.classList.remove('active');
    els.btnTimerOff.classList.add('active');
};
els.btnStart.onclick = () => {
    initAudio();
    targetNumber = parseInt(els.inpTarget.value) || 10;
    startGameCalibration();
};
els.btnRestart.onclick = () => {
    els.overlayResult.classList.add('hidden');
    els.overlayStart.classList.remove('hidden');
    state = 'MENU';
};

function startGameCalibration() {
    state = 'CALIBRATION';
    els.overlayStart.classList.add('hidden');
    els.inGameUI.classList.remove('hidden');
    els.centerMsg.classList.remove('hidden');
    els.centerMsg.innerText = "바닥 선에 발을 맞추고\n박수를 쳐서 시작하세요!";
    
    p1.reset();
    p1.ui.container.classList.remove('hidden');
}

function generateCircles(player, poseKeypoints, regionX, regionWidth, regionHeight) {
    player.circles = [];

    // 머리(코) 위치 기준으로 팔을 뻗을 수 있는 최대 높이 계산
    let nose        = poseKeypoints.find(k => k.name === 'nose');
    let leftWrist   = poseKeypoints.find(k => k.name === 'left_wrist');
    let rightWrist  = poseKeypoints.find(k => k.name === 'right_wrist');
    let leftShoulder = poseKeypoints.find(k => k.name === 'left_shoulder');

    // 어깨~손목 거리를 팔 길이로 추정 (없으면 200px 기본값)
    let armLength = 200;
    if (leftShoulder && leftWrist && leftShoulder.score > 0.3 && leftWrist.score > 0.3) {
        armLength = Math.abs(leftShoulder.y - leftWrist.y) * 1.1;
    }

    // 위: 머리 위로 팔 하나 길이만큼 + 여유 (단, 화면 밖으로 나가지 않게 최소 30px)
    let headY  = (nose?.score > 0.3) ? nose.y : (leftShoulder?.y || regionHeight * 0.2) - 100;
    let topY   = Math.max(30, headY - armLength);

    // 아래: 바닥선(발 위치)까지 생성 (원이 반쯤 잘리지 않게 반지름만큼 여유)
    let bottomY = player.baselineY;

    
    for(let i=1; i<=targetNumber; i++) {
        let radius = 30 + Math.random() * 20;
        let maxRetries = 10;
        let x, y;
        let overlap = true;
        
        while(overlap && maxRetries > 0) {
            x = regionX + radius + Math.random() * (regionWidth - radius*2);
            y = topY + radius + Math.random() * (bottomY - topY - radius*2);
            overlap = false;
            for(let j=0; j<player.circles.length; j++) {
                let dx = x - player.circles[j].x;
                let dy = y - player.circles[j].y;
                if(Math.sqrt(dx*dx + dy*dy) < radius + player.circles[j].r + 10) {
                    overlap = true; break;
                }
            }
            maxRetries--;
        }
        
        player.circles.push({ num: i, x, y, r: radius, active: false });
    }
    
    if(player.circles.length > 0) player.circles[0].active = true;
}

function startActualGame() {
    state = 'PLAYING';
    els.centerMsg.classList.add('hidden');
    let now = performance.now();
    p1.startTime = now;
}

function renderBaseline(ctx, x, w, y) {
    ctx.strokeStyle = '#20D68A';
    ctx.lineWidth = 10;
    ctx.setLineDash([20, 10]);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.stroke();
    ctx.setLineDash([]);
}

function drawSkeleton(ctx, keypoints, minConfidence, offsetX, scaleX, scaleY) {
    const adjacentKeyPoints = poseDetection.util.getAdjacentPairs(currentModel || poseDetection.SupportedModels.MoveNet);
    
    ctx.save();
    ctx.lineWidth = 8;
    ctx.strokeStyle = '#FFFFFF';
    ctx.fillStyle = '#1EA7FD';
    
    adjacentKeyPoints.forEach((pair) => {
        const i = pair[0];
        const j = pair[1];
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];
        if (kp1.score >= minConfidence && kp2.score >= minConfidence) {
            ctx.beginPath();
            ctx.moveTo(offsetX + kp1.x * scaleX, kp1.y * scaleY);
            ctx.lineTo(offsetX + kp2.x * scaleX, kp2.y * scaleY);
            ctx.stroke();
        }
    });
    
    keypoints.forEach(kp => {
        if(kp.score >= minConfidence) {
            ctx.beginPath();
            ctx.arc(offsetX + kp.x * scaleX, kp.y * scaleY, 8, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }
    });

    // Draw Head
    let nose = keypoints.find(k=>k.name==='nose');
    if(nose && nose.score > minConfidence) {
        ctx.beginPath();
        ctx.arc(offsetX + nose.x * scaleX, nose.y * scaleY, 25, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFAE00';
        ctx.fill();
        ctx.stroke();
    }
    
    ctx.restore();
}

function checkClap(keypoints, minConf) {
    let lw = keypoints.find(k=>k.name==='left_wrist');
    let rw = keypoints.find(k=>k.name==='right_wrist');
    if(lw && rw && lw.score > minConf && rw.score > minConf) {
        let dist = Math.sqrt(Math.pow(lw.x - rw.x, 2) + Math.pow(lw.y - rw.y, 2));
        return dist < 50; // threshold
    }
    return false;
}

function checkJump(keypoints, player, minConf) {
    let lA = keypoints.find(k=>k.name==='left_ankle');
    let rA = keypoints.find(k=>k.name==='right_ankle');
    
    // Check y distance to baseline
    let jumping = false;
    if(lA && rA && lA.score > minConf && rA.score > minConf) {
        let lowestFootY = Math.max(lA.y, rA.y);
        if (player.baselineY - lowestFootY > 80) { // arbitrary jump threshold
            jumping = true;
        }
    }
    return jumping;
}

function processPlayerLogic(player, pose, regionX, regionWidth, regionHeight, scaleX, scaleY) {
    if(!pose) return;
    let keypoints = pose.keypoints;
    let minConf = 0.3;
    
    // Scale keypoints
    let scaledKps = keypoints.map(kp => ({...kp, x: kp.x * scaleX, y: kp.y * scaleY}));

    if (state === 'CALIBRATION') {
        // Set baseline based on canvas height initially, wait for clap
        player.baselineY = regionHeight * 0.85; 
        
        let isClapping = checkClap(scaledKps, minConf);
        if (isClapping && !player.isReady) {
            player.isReady = true;
            sounds.clap();
            
            // Record initial shoulder distance
            let ls = scaledKps.find(k=>k.name==='left_shoulder');
            let rs = scaledKps.find(k=>k.name==='right_shoulder');
            if (ls && rs && ls.score > minConf && rs.score > minConf) {
                player.initialShoulderDist = Math.sqrt(Math.pow(ls.x - rs.x, 2) + Math.pow(ls.y - rs.y, 2));
            }
            
            generateCircles(player, scaledKps, regionX, regionWidth, regionHeight);
            player.ui.status.innerText = "준비 완료!";
        }
    } 
    else if (state === 'PLAYING') {
        if (player.finished) return;
        
        player.elapsedTime = performance.now() - player.startTime;
        
        // Distance detection (too close)
        let tooClose = false;
        if (player.initialShoulderDist) {
            let ls = scaledKps.find(k=>k.name==='left_shoulder');
            let rs = scaledKps.find(k=>k.name==='right_shoulder');
            if (ls && rs && ls.score > minConf && rs.score > minConf) {
                let dist = Math.sqrt(Math.pow(ls.x - rs.x, 2) + Math.pow(ls.y - rs.y, 2));
                if (dist > player.initialShoulderDist * 1.3) {
                    tooClose = true;
                }
            }
        }

        // Jump detection
        if (checkJump(scaledKps, player, minConf) || tooClose) {
            if(player.jumpWarningFrames === 0) {
                sounds.error();
            }
            player.jumpWarningFrames = 60; // show warning for 1 sec at 60fps
        }

        // Only process collisions if there is no warning active
        if (player.jumpWarningFrames > 0) return;

        // Collision detection for hands vs active circle
        let nextTargetNum = player.progress + 1;
        let c = player.circles.find(circ => circ.num === nextTargetNum);
        if (c) {
            let lw = scaledKps.find(k=>k.name==='left_wrist');
            let rw = scaledKps.find(k=>k.name==='right_wrist');
            
            let hit = false;
            if (lw && lw.score > minConf) {
                let dx = (regionX + lw.x) - c.x;
                let dy = lw.y - c.y;
                if (Math.sqrt(dx*dx + dy*dy) < c.r + 20) hit = true;
            }
            if (!hit && rw && rw.score > minConf) {
                let dx = (regionX + rw.x) - c.x;
                let dy = rw.y - c.y;
                if (Math.sqrt(dx*dx + dy*dy) < c.r + 20) hit = true;
            }
            
            if (hit) {
                player.progress++;
                sounds.pop();
                if(player.progress === targetNumber) {
                    player.finished = true;
                    player.finishTime = player.elapsedTime;
                    sounds.win();
                    player.ui.status.innerText = "성공!";
                } else {
                    // Activate next
                    let nextC = player.circles.find(circ => circ.num === player.progress + 1);
                    if(nextC) nextC.active = true;
                }
            }
        }
    }
}

function drawPlayerCircles(ctx, player) {
    player.circles.forEach(c => {
        if (c.num <= player.progress) return; // already popped
        if (c.num > player.progress + 10) return; // max 10 visible numbers
        
        ctx.save();
        ctx.beginPath();
        ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
        
        if (c.num === player.progress + 1 && colorHintEnabled) {
            // Next target - animate and color hint
            ctx.fillStyle = `hsl(${(performance.now()/10) % 360}, 100%, 60%)`;
            ctx.shadowBlur = 20;
            ctx.shadowColor = ctx.fillStyle;
            ctx.lineWidth = 4;
            ctx.strokeStyle = '#fff';
        } else {
            // Upcoming targets (or next target when hints are disabled)
            ctx.fillStyle = 'rgba(255,255,255,0.2)';
            ctx.shadowBlur = 0;
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgba(255,255,255,0.5)';
        }
        
        ctx.fill();
        ctx.stroke();
        
        // Text
        ctx.fillStyle = '#fff';
        ctx.font = 'bold ' + (c.r) + 'px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(c.num, c.x, c.y);
        ctx.restore();
    });
}

function checkGameState() {
    if (state === 'CALIBRATION') {
        if (p1.isReady) {
            startActualGame();
        }
    } else if (state === 'PLAYING') {
        if (p1.finished) {
            state = 'RESULT';
            showResult();
        }
    }
}

function formatTime(ms) {
    let t = Math.floor(ms);
    let s = Math.floor((t / 1000) % 60);
    let m = Math.floor(t / 60000);
    let ms_part = t % 1000;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${Math.floor(ms_part/10).toString().padStart(2, '0')}`;
}

function showResult() {
    els.inGameUI.classList.add('hidden');
    els.overlayResult.classList.remove('hidden');
    
    els.resTitle.innerText = "임무 완수!";
    els.resDetails.innerHTML = `걸린 시간: <span style="color:var(--accent);font-weight:900;">${formatTime(p1.finishTime)}</span>`;
}

// 추론 루프: 렌더링과 독립적으로 AI 추론만 담당.
// 추론에 시간이 걸려도 requestAnimationFrame을 막지 않아 화면이 부드러움.
async function inferLoop() {
    if (inferLoopStarted) return;
    inferLoopStarted = true;

    const isBlazePose = currentModel === poseDetection.SupportedModels.BlazePose;
    const maxPoses = 1;

    while (renderLoopStarted) {
        if ((state === 'CALIBRATION' || state === 'PLAYING') && video && video.readyState >= 2) {
            try {
                // flipHorizontal은 모델마다 동작이 달라 신뢰할 수 없음.
                // false로 원본 좌표를 받은 뒤, 아래에서 직접 뒤집는다.
                const raw = await detector.estimatePoses(video, { maxPoses, flipHorizontal: false });

                // 미러 모드: x 좌표를 직접 반전 (x = videoWidth - x)
                // 이렇게 해야 오른팔을 들면 화면 오른쪽 팔이 올라가는 거울 효과가 확실히 작동
                const vw = video.videoWidth;
                lastPoses = raw.map(pose => ({
                    ...pose,
                    keypoints: pose.keypoints.map(kp => ({
                        ...kp,
                        x: vw - kp.x
                    }))
                }));
            } catch (e) { /* 추론 오류 무시 */ }
        }
        // 크롬북 CPU 부하 및 발열 방지: 추론 빈도를 초당 약 20회(50ms 대기)로 제한
        await new Promise(r => setTimeout(r, 50));
    }

    inferLoopStarted = false;
}


function renderLoop() {
    // 렌더링만 담당 — AI 추론은 inferLoop()에서 비동기 처리
    if (video.readyState < 2) {
        requestAnimationFrame(renderLoop);
        return;
    }

    // 표시용 포즈: lastPoses를 향해 매 프레임 부드럽게 lerp 이동
    if (lastPoses.length > 0) {
        displayPoses = lerpPoses(displayPoses, lastPoses);
    }
    const poses = displayPoses;

    // Render Logic
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (state === 'CALIBRATION' || state === 'PLAYING') {
        let cw = canvas.width;
        let ch = canvas.height;
        let vw = video.videoWidth;
        let vh = video.videoHeight;
        
        let scaleX = cw / vw;
        let scaleY = ch / vh;
        
        renderBaseline(ctx, 0, cw, p1.baselineY || ch*0.85);
        processPlayerLogic(p1, poses[0], 0, cw, ch, scaleX, scaleY);
        if(poses[0]) drawSkeleton(ctx, poses[0].keypoints, 0.3, 0, scaleX, scaleY);
        drawPlayerCircles(ctx, p1);
        p1.updateUI();
        
        checkGameState();
    }

    requestAnimationFrame(renderLoop);
}

// Start sequence when page loads
window.onload = init;

