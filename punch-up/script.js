/**
 * Punch Up Game Logic
 * Uses MediaPipe/TensorFlow.js to track body keypoints.
 */

let state = 'LOADING'; // LOADING, MENU, CALIBRATION, PLAYING, RESULT
let timeLimitSec = 30;
let lastFrameTime = 0;

class GameState {
    constructor() {
        this.score = 0;
        this.startTime = 0;
        this.timeRemaining = 0;
        this.finished = false;
        
        // Calibration
        this.baselineY = 0;
        this.isReady = false; 
        
        // Game Logic
        this.leftNext = 'TOP'; // 'TOP' or 'BOTTOM'
        this.rightNext = 'TOP';
        this.targets = {
            tl: { x: 0, y: 0, r: 50, scale: 1 },
            bl: { x: 0, y: 0, r: 50, scale: 1 },
            tr: { x: 0, y: 0, r: 50, scale: 1 },
            br: { x: 0, y: 0, r: 50, scale: 1 }
        };
        this.startTarget = { x: 0, y: 0, r: 70, scale: 1 };
        this.leftHandWasIn = { tl: false, bl: false };
        this.rightHandWasIn = { tr: false, br: false };
        this.startHandWasIn = false;
        
        this.ui = {
            time: document.getElementById(`p1-time`),
            score: document.getElementById(`p1-score`),
            status: document.getElementById(`p1-status`)
        };
    }
    
    reset() {
        this.score = 0;
        this.timeRemaining = timeLimitSec * 1000;
        this.finished = false;
        this.isReady = false;
        this.leftNext = 'TOP';
        this.rightNext = 'TOP';
        this.leftHandWasIn = { tl: false, bl: false };
        this.rightHandWasIn = { tr: false, br: false };
        this.startHandWasIn = false;
        if(this.ui.status) this.ui.status.innerText = '';
        this.updateUI();
    }
    
    updateUI() {
        if(!this.ui.score) return;
        this.ui.score.innerText = `${this.score}`;
        
        let t = Math.max(0, this.timeRemaining);
        let ms = t % 1000;
        let s = Math.floor(t / 1000);
        this.ui.time.innerText = `${s.toString().padStart(2, '0')}.${Math.floor(ms/10).toString().padStart(2, '0')}`;
    }
}

let game = new GameState();

// TFJS & Camera Variables
let video, canvas, ctx;
let detector;
let currentModel = null;

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
    inpTime: document.getElementById('time-limit'),
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

let lastPoses = [];
let displayPoses = [];
let inferLoopStarted = false;

const LERP_ALPHA = 0.2;

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
    
    const isSecureOrigin = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!isSecureOrigin) {
        throw new Error('INSECURE_ORIGIN');
    }

    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            'audio': false,
            'video': { 
                width: { ideal: 320 },
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
    if (typeof tf === 'undefined') {
        throw new Error('SCRIPT_LOAD_ERROR: @tensorflow/tfjs 스크립트를 불러오지 못했습니다.');
    }
    if (typeof poseDetection === 'undefined') {
        throw new Error('SCRIPT_LOAD_ERROR: pose-detection 스크립트를 불러오지 못했습니다.');
    }

    try {
        await tf.setBackend('webgl');
        await tf.ready();
    } catch (backendErr) {
        await tf.setBackend('cpu');
        await tf.ready();
    }

    try {
        const config = { modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING };
        detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, config);
        currentModel = poseDetection.SupportedModels.MoveNet;
        return;
    } catch (moveNetErr) {}

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
    } catch (blazeErr) {
        throw new Error('MODEL_LOAD_ERROR: MoveNet와 BlazePose 모두 실패. ' + blazeErr.message);
    }
}

async function init() {
    const loadingH2 = els.overlayLoading.querySelector('h2');
    const loadingP = els.overlayLoading.querySelector('p');

    els.overlayLoading.classList.remove('hidden');
    els.loader.classList.remove('hidden');
    els.retryBtn.classList.add('hidden');
    loadingH2.innerText = "카메라와 AI 모델을 불러오는 중...";
    loadingP.innerText = "잠시만 기다려주세요.";

    try {
        if (!cameraReady) {
            await setupCamera();
            cameraReady = true;
        }
        
        if (!canvas) {
            canvas = document.getElementById('game-canvas');
            ctx = canvas.getContext('2d');
            resizeCanvas();
        }
        if (!resizeListenerAdded) {
            window.addEventListener('resize', resizeCanvas);
            resizeListenerAdded = true;
        }
        
        if (!modelReady) {
            loadingH2.innerText = "AI 모델 데이터 로딩 중...";
            await loadModel();
            modelReady = true;
        }
        
        els.overlayLoading.classList.add('hidden');
        els.overlayStart.classList.remove('hidden');
        state = 'MENU';
        
        if (!renderLoopStarted) {
            renderLoopStarted = true;
            requestAnimationFrame(renderLoop);
            inferLoop();
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
    
    // Update target positions based on canvas size
    game.targets.tl.x = canvas.width * 0.25;
    game.targets.tl.y = canvas.height * 0.2;
    game.targets.tl.r = canvas.height * 0.1;

    game.targets.bl.x = canvas.width * 0.25;
    game.targets.bl.y = canvas.height * 0.8;
    game.targets.bl.r = canvas.height * 0.1;

    game.targets.tr.x = canvas.width * 0.75;
    game.targets.tr.y = canvas.height * 0.2;
    game.targets.tr.r = canvas.height * 0.1;

    game.targets.br.x = canvas.width * 0.75;
    game.targets.br.y = canvas.height * 0.8;
    game.targets.br.r = canvas.height * 0.1;

    game.startTarget.x = canvas.width * 0.5;
    game.startTarget.y = canvas.height * 0.3;
    game.startTarget.r = canvas.height * 0.12;
}

els.btnStart.onclick = () => {
    initAudio();
    timeLimitSec = parseInt(els.inpTime.value) || 30;
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
    els.centerMsg.innerText = "가운데 시작 버튼을\n터치하여 시작하세요!";
    
    game.reset();
}

function startActualGame() {
    state = 'PLAYING';
    els.centerMsg.classList.add('hidden');
    game.startTime = performance.now();
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

function processGameLogic(pose, regionX, scaleX, scaleY) {
    if(!pose) return;
    let keypoints = pose.keypoints;
    let minConf = 0.3;
    
    let scaledKps = keypoints.map(kp => ({...kp, x: kp.x * scaleX, y: kp.y * scaleY}));

    let leftHand = null;
    let rightHand = null;

    let w1 = scaledKps.find(k=>k.name==='left_wrist');
    let w2 = scaledKps.find(k=>k.name==='right_wrist');
    
    let hands = [];
    if (w1 && w1.score > minConf) hands.push(w1);
    if (w2 && w2.score > minConf) hands.push(w2);

    hands.forEach(h => {
        if (h.x < canvas.width / 2) leftHand = h;
        else rightHand = h;
    });

    if (state === 'CALIBRATION') {
        game.startTarget.scale += (1 - game.startTarget.scale) * 0.1;
        
        let startHit = false;
        if (leftHand) {
            let dx = leftHand.x - game.startTarget.x;
            let dy = leftHand.y - game.startTarget.y;
            if (Math.sqrt(dx*dx + dy*dy) < game.startTarget.r * 1.5) startHit = true;
        }
        if (rightHand) {
            let dx = rightHand.x - game.startTarget.x;
            let dy = rightHand.y - game.startTarget.y;
            if (Math.sqrt(dx*dx + dy*dy) < game.startTarget.r * 1.5) startHit = true;
        }

        if (startHit && !game.startHandWasIn && !game.isReady) {
            game.isReady = true;
            game.startTarget.scale = 1.5;
            sounds.clap();
            game.ui.status.innerText = "준비 완료!";
        }
        game.startHandWasIn = startHit;
    } 
    else if (state === 'PLAYING') {
        if (game.finished) return;
        
        let now = performance.now();
        let elapsed = now - game.startTime;
        game.timeRemaining = (timeLimitSec * 1000) - elapsed;
        
        if (game.timeRemaining <= 0) {
            game.timeRemaining = 0;
            game.finished = true;
            sounds.win();
            return;
        }

        // Animate targets
        game.targets.tl.scale += (1 - game.targets.tl.scale) * 0.1;
        game.targets.bl.scale += (1 - game.targets.bl.scale) * 0.1;
        game.targets.tr.scale += (1 - game.targets.tr.scale) * 0.1;
        game.targets.br.scale += (1 - game.targets.br.scale) * 0.1;

        // Collision detection
        let leftHand = null;
        let rightHand = null;

        let w1 = scaledKps.find(k=>k.name==='left_wrist');
        let w2 = scaledKps.find(k=>k.name==='right_wrist');
        
        let hands = [];
        if (w1 && w1.score > minConf) hands.push(w1);
        if (w2 && w2.score > minConf) hands.push(w2);

        hands.forEach(h => {
            if (h.x < canvas.width / 2) leftHand = h;
            else rightHand = h;
        });

        let tlIn = false;
        let blIn = false;
        let trIn = false;
        let brIn = false;

        if (leftHand) {
            let dxTop = leftHand.x - game.targets.tl.x;
            let dyTop = leftHand.y - game.targets.tl.y;
            if (Math.sqrt(dxTop*dxTop + dyTop*dyTop) < game.targets.tl.r * 1.5) tlIn = true;

            let dxBot = leftHand.x - game.targets.bl.x;
            let dyBot = leftHand.y - game.targets.bl.y;
            if (Math.sqrt(dxBot*dxBot + dyBot*dyBot) < game.targets.bl.r * 1.5) blIn = true;
        }

        if (rightHand) {
            let dxTop = rightHand.x - game.targets.tr.x;
            let dyTop = rightHand.y - game.targets.tr.y;
            if (Math.sqrt(dxTop*dxTop + dyTop*dyTop) < game.targets.tr.r * 1.5) trIn = true;

            let dxBot = rightHand.x - game.targets.br.x;
            let dyBot = rightHand.y - game.targets.br.y;
            if (Math.sqrt(dxBot*dxBot + dyBot*dyBot) < game.targets.br.r * 1.5) brIn = true;
        }

        // Left Arm Logic
        if (game.leftNext === 'TOP' && tlIn && !game.leftHandWasIn.tl) {
            game.score++;
            game.leftNext = 'BOTTOM';
            game.targets.tl.scale = 1.5;
            sounds.pop();
        } else if (game.leftNext === 'BOTTOM' && blIn && !game.leftHandWasIn.bl) {
            game.score++;
            game.leftNext = 'TOP';
            game.targets.bl.scale = 1.5;
            sounds.pop();
        }

        // Right Arm Logic
        if (game.rightNext === 'TOP' && trIn && !game.rightHandWasIn.tr) {
            game.score++;
            game.rightNext = 'BOTTOM';
            game.targets.tr.scale = 1.5;
            sounds.pop();
        } else if (game.rightNext === 'BOTTOM' && brIn && !game.rightHandWasIn.br) {
            game.score++;
            game.rightNext = 'TOP';
            game.targets.br.scale = 1.5;
            sounds.pop();
        }

        game.leftHandWasIn = { tl: tlIn, bl: blIn };
        game.rightHandWasIn = { tr: trIn, br: brIn };
    }
}

function drawTargets(ctx) {
    if (state === 'CALIBRATION') {
        // Draw Start Button
        ctx.save();
        ctx.beginPath();
        ctx.arc(game.startTarget.x, game.startTarget.y, game.startTarget.r * game.startTarget.scale, 0, Math.PI * 2);
        
        ctx.fillStyle = '#20D68A';
        ctx.shadowBlur = 30;
        ctx.shadowColor = ctx.fillStyle;
        ctx.lineWidth = 5;
        ctx.strokeStyle = '#fff';
        
        ctx.fill();
        ctx.stroke();
        
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 30px Outfit';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.shadowBlur = 0;
        ctx.fillText("START", game.startTarget.x, game.startTarget.y);
        ctx.restore();
    } else if (state === 'PLAYING') {
        // Helper to draw a single target
        const drawTarget = (target, isActive, hue) => {
            ctx.save();
            ctx.beginPath();
            ctx.arc(target.x, target.y, target.r * target.scale, 0, Math.PI * 2);
            
            if (isActive) {
                ctx.fillStyle = `hsl(${hue}, 100%, 60%)`;
                ctx.shadowBlur = 30;
                ctx.shadowColor = ctx.fillStyle;
                ctx.lineWidth = 5;
                ctx.strokeStyle = '#fff';
            } else {
                ctx.fillStyle = 'rgba(255,255,255,0.1)';
                ctx.shadowBlur = 0;
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            }
            ctx.fill();
            ctx.stroke();
            ctx.restore();
        };

        const timeHue = (performance.now() / 10) % 360;

        drawTarget(game.targets.tl, game.leftNext === 'TOP', timeHue);
        drawTarget(game.targets.bl, game.leftNext === 'BOTTOM', timeHue);
        
        drawTarget(game.targets.tr, game.rightNext === 'TOP', (timeHue + 180) % 360);
        drawTarget(game.targets.br, game.rightNext === 'BOTTOM', (timeHue + 180) % 360);
    }
}

function checkGameState() {
    if (state === 'CALIBRATION') {
        if (game.isReady) {
            startActualGame();
        }
    } else if (state === 'PLAYING') {
        if (game.finished) {
            state = 'RESULT';
            showResult();
        }
    }
}

function showResult() {
    els.inGameUI.classList.add('hidden');
    els.overlayResult.classList.remove('hidden');
    
    els.resTitle.innerText = "시간 종료!";
    els.resDetails.innerHTML = `최종 점수: <span style="color:var(--accent);font-weight:900;">${game.score}</span> 점`;
}

async function inferLoop() {
    if (inferLoopStarted) return;
    inferLoopStarted = true;

    const maxPoses = 1;

    while (renderLoopStarted) {
        if ((state === 'CALIBRATION' || state === 'PLAYING') && video && video.readyState >= 2) {
            try {
                const raw = await detector.estimatePoses(video, { maxPoses, flipHorizontal: false });

                const vw = video.videoWidth;
                lastPoses = raw.map(pose => ({
                    ...pose,
                    keypoints: pose.keypoints.map(kp => ({
                        ...kp,
                        x: vw - kp.x
                    }))
                }));
            } catch (e) {}
        }
        await new Promise(r => setTimeout(r, 50));
    }

    inferLoopStarted = false;
}

function renderLoop() {
    if (video.readyState < 2) {
        requestAnimationFrame(renderLoop);
        return;
    }

    if (lastPoses.length > 0) {
        displayPoses = lerpPoses(displayPoses, lastPoses);
    }
    const poses = displayPoses;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (state === 'CALIBRATION' || state === 'PLAYING') {
        let cw = canvas.width;
        let ch = canvas.height;
        let vw = video.videoWidth;
        let vh = video.videoHeight;
        
        let scaleX = cw / vw;
        let scaleY = ch / vh;
        
        drawTargets(ctx);
        processGameLogic(poses[0], 0, scaleX, scaleY);
        if(poses[0]) drawSkeleton(ctx, poses[0].keypoints, 0.3, 0, scaleX, scaleY);
        
        game.updateUI();
        checkGameState();
    }

    requestAnimationFrame(renderLoop);
}

window.onload = init;
