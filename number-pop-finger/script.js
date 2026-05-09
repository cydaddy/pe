/**
 * Number Pop Finger Game Logic
 * Uses MediaPipe Hands to track index finger.
 */

// Basic State
let state = 'LOADING'; // LOADING, MENU, WAITING_FOR_START, PLAYING, RESULT
let targetNumber = 10;
let colorHintEnabled = true;
let timerEnabled = true;

// Player/Game State
let progress = 0;
let startTime = 0;
let elapsedTime = 0;
let finished = false;
let finishTime = 0;
let circles = [];
let startOrb = null;

// TFJS & Camera Variables
let video, canvas, ctx;
let detector;

// Hand tracking state
let lastHands = [];
let displayHands = [];
const LERP_ALPHA = 0.4;
let inferLoopStarted = false;
let renderLoopStarted = false;
let resizeListenerAdded = false;

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
    start: () => {
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
    loader: document.querySelector('.loader'),
    
    time: document.getElementById('p1-time'),
    progress: document.getElementById('p1-progress'),
    status: document.getElementById('p1-status')
};

function updateUI() {
    els.progress.innerText = `${progress} / ${targetNumber}`;
    if (timerEnabled) {
        els.time.style.display = 'block';
        let timeToDisplay = finished ? finishTime : elapsedTime;
        let ms = timeToDisplay % 1000;
        let s = Math.floor((timeToDisplay / 1000) % 60);
        let m = Math.floor(timeToDisplay / 60000);
        els.time.innerText = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${Math.floor(ms/10).toString().padStart(2, '0')}`;
    } else {
        els.time.style.display = 'none';
    }
}

// MediaPipe Model
async function loadModel() {
    if (!window.tf) throw new Error('SCRIPT_LOAD_ERROR: TensorFlow.js');
    if (!window.handPoseDetection) throw new Error('SCRIPT_LOAD_ERROR: hand-pose-detection');
    try { await tf.setBackend('webgl'); await tf.ready(); }
    catch { await tf.setBackend('cpu'); await tf.ready(); }

    try {
        detector = await handPoseDetection.createDetector(
            handPoseDetection.SupportedModels.MediaPipeHands,
            { 
                runtime: 'mediapipe', 
                solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915',
                maxHands: 2, 
                modelType: 'lite' 
            }
        );
    } catch(e) {
        detector = await handPoseDetection.createDetector(
            handPoseDetection.SupportedModels.MediaPipeHands,
            { runtime: 'tfjs', maxHands: 2, modelType: 'lite' }
        );
    }
}

async function setupCamera() {
    video = document.getElementById('video');
    const secure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!secure) throw new Error('INSECURE_ORIGIN');
    if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width:{ideal:640}, height:{ideal:480}, facingMode:'user' }
    }).catch(e => { throw e.name === 'NotAllowedError' ? new Error('PERMISSION_DENIED') : e; });
    video.srcObject = stream;
    return new Promise(res => { video.onloadedmetadata = () => { video.play(); res(); }; });
}

let cameraReady = false, modelReady = false;

async function init() {
    els.overlayLoading.classList.remove('hidden');
    els.loader.classList.remove('hidden');
    els.retryBtn.classList.add('hidden');
    els.overlayLoading.querySelector('h2').innerText = "카메라와 AI 모델을 불러오는 중...";
    els.overlayLoading.querySelector('p').innerText = "잠시만 기다려주세요.";

    try {
        if (!cameraReady) { await setupCamera(); cameraReady = true; }
        if (!canvas) {
            canvas = document.getElementById('game-canvas');
            ctx = canvas.getContext('2d');
            resizeCanvas();
        }
        if (!resizeListenerAdded) {
            window.addEventListener('resize', resizeCanvas);
            resizeListenerAdded = true;
        }
        if (!modelReady) { await loadModel(); modelReady = true; }
        
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
        els.overlayLoading.querySelector('h2').innerText = "설정이 필요합니다";
        els.loader.classList.add('hidden');
        els.retryBtn.classList.remove('hidden');
        // error handling omitted for brevity, keeping simple text
        els.overlayLoading.querySelector('p').innerHTML = "오류: " + e.message;
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
    
    // Prepare game state
    progress = 0;
    elapsedTime = 0;
    finished = false;
    els.status.innerText = "";
    els.overlayStart.classList.add('hidden');
    els.inGameUI.classList.remove('hidden');
    
    generateStartOrb();
    state = 'WAITING_FOR_START';
};
els.btnRestart.onclick = () => {
    els.overlayResult.classList.add('hidden');
    els.overlayStart.classList.remove('hidden');
    state = 'MENU';
};

function getCanvasHandPositions() {
    let positions = [];
    if (!video) return positions;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    const cw = canvas.width;
    const ch = canvas.height;
    
    displayHands.forEach(hand => {
        let indexTip = hand.keypoints.find(kp => kp.name === 'index_finger_tip');
        if (indexTip) {
            let p = mapVideoToCanvas(vw - indexTip.x, indexTip.y, vw, vh, cw, ch);
            positions.push(p);
        }
    });
    return positions;
}

function generateStartOrb() {
    const cw = canvas.width;
    const ch = canvas.height;
    const margin = 100;
    
    let handPositions = getCanvasHandPositions();
    let x, y;
    let safeDistance = 300;
    
    if (handPositions.length === 0) {
        // 손이 화면에 없을 때 -> 중앙 배치
        x = cw / 2;
        y = ch / 2;
    } else {
        // 손이 화면에 있을 때 -> 손에서 최소 safeDistance 만큼 떨어진 랜덤 위치
        let attempts = 0;
        let valid = false;
        while (!valid && attempts < 50) {
            x = margin + Math.random() * (cw - margin * 2);
            y = margin + Math.random() * (ch - margin * 2);
            
            valid = true;
            for (let p of handPositions) {
                let dist = Math.sqrt(Math.pow(x - p.x, 2) + Math.pow(y - p.y, 2));
                if (dist < safeDistance) {
                    valid = false;
                    break;
                }
            }
            attempts++;
        }
    }
    
    startOrb = { x, y, r: 60 };
}

function generateCircles() {
    circles = [];
    const cw = canvas.width;
    const ch = canvas.height;
    const margin = 80;

    for (let i = 1; i <= targetNumber; i++) {
        let radius = 40 + Math.random() * 20;
        let maxRetries = 20;
        let x, y;
        let overlap = true;
        
        while (overlap && maxRetries > 0) {
            x = margin + radius + Math.random() * (cw - margin * 2 - radius * 2);
            y = margin + radius + Math.random() * (ch - margin * 2 - radius * 2);
            overlap = false;
            for (let j = 0; j < circles.length; j++) {
                let dx = x - circles[j].x;
                let dy = y - circles[j].y;
                if (Math.sqrt(dx*dx + dy*dy) < radius + circles[j].r + 15) {
                    overlap = true; break;
                }
            }
            maxRetries--;
        }
        circles.push({ num: i, x, y, r: radius, active: false });
    }
    if (circles.length > 0) circles[0].active = true;
}

function startActualGame() {
    sounds.start();
    generateCircles();
    startTime = performance.now();
    state = 'PLAYING';
    els.centerMsg.classList.add('hidden');
}

function mapVideoToCanvas(x, y, vw, vh, cw, ch) {
    const videoRatio = vw / vh;
    const canvasRatio = cw / ch;
    let drawW, drawH, offsetX, offsetY;

    if (canvasRatio > videoRatio) {
        drawW = cw;
        drawH = cw / videoRatio;
        offsetX = 0;
        offsetY = (ch - drawH) / 2;
    } else {
        drawH = ch;
        drawW = ch * videoRatio;
        offsetX = (cw - drawW) / 2;
        offsetY = 0;
    }

    return {
        x: offsetX + (x / vw) * drawW,
        y: offsetY + (y / vh) * drawH
    };
}

function lerp(a, b, alpha) { return a + (b - a) * alpha; }

function lerpHands(current, target) {
    if (!current || current.length !== target.length) return target;
    return target.map((hand, hi) => {
        const prev = current[hi];
        if (!prev) return hand;
        return {
            ...hand,
            keypoints: hand.keypoints.map((kp, ki) => {
                const pkp = prev.keypoints[ki];
                if (!pkp) return kp;
                return { ...kp, x: lerp(pkp.x, kp.x, LERP_ALPHA), y: lerp(pkp.y, kp.y, LERP_ALPHA) };
            })
        };
    });
}

function drawHandSkeleton(vw, vh, cw, ch) {
    const connections = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],[0,17]
    ];
    ctx.save();
    displayHands.forEach((hand, handIdx) => {
        let kps = hand.keypoints;
        ctx.strokeStyle = 'rgba(255,255,255,0.4)';
        ctx.lineWidth = 4;
        connections.forEach(([a,b]) => {
            if (!kps[a] || !kps[b]) return;
            const p1 = mapVideoToCanvas(vw - kps[a].x, kps[a].y, vw, vh, cw, ch);
            const p2 = mapVideoToCanvas(vw - kps[b].x, kps[b].y, vw, vh, cw, ch);
            ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        });
        kps.forEach((kp) => {
            if (!kp) return;
            const p = mapVideoToCanvas(vw - kp.x, kp.y, vw, vh, cw, ch);
            
            // Highlight index finger tip
            if (kp.name === 'index_finger_tip') {
                ctx.beginPath(); ctx.arc(p.x, p.y, 14, 0, Math.PI * 2);
                ctx.fillStyle = '#FFAE00';
                ctx.shadowBlur = 15; ctx.shadowColor = '#FFAE00';
                ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
            } else {
                ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
                ctx.fillStyle = '#1EA7FD';
                ctx.shadowBlur = 0;
                ctx.fill();
            }
        });
    });
    ctx.restore();
}

async function inferLoop() {
    if (inferLoopStarted) return;
    inferLoopStarted = true;
    while (renderLoopStarted) {
        if (state !== 'LOADING' && state !== 'RESULT' && video && video.readyState >= 2) {
            try {
                lastHands = await detector.estimateHands(video, { flipHorizontal: false });
            } catch(e) {}
        }
        await new Promise(r => setTimeout(r, 50));
    }
    inferLoopStarted = false;
}

function checkCollisions(handPositions) {
    if (state === 'WAITING_FOR_START' && startOrb) {
        for (let p of handPositions) {
            let dist = Math.sqrt(Math.pow(p.x - startOrb.x, 2) + Math.pow(p.y - startOrb.y, 2));
            if (dist < startOrb.r) {
                startActualGame();
                break;
            }
        }
    } else if (state === 'PLAYING') {
        if (finished) return;
        
        let nextTargetNum = progress + 1;
        let c = circles.find(circ => circ.num === nextTargetNum);
        if (c) {
            let hit = false;
            for (let p of handPositions) {
                let dist = Math.sqrt(Math.pow(p.x - c.x, 2) + Math.pow(p.y - c.y, 2));
                if (dist < c.r + 20) {
                    hit = true;
                    break;
                }
            }
            
            if (hit) {
                progress++;
                sounds.pop();
                if (progress === targetNumber) {
                    finished = true;
                    finishTime = elapsedTime;
                    sounds.win();
                    els.status.innerText = "성공!";
                    state = 'RESULT';
                    showResult();
                } else {
                    let nextC = circles.find(circ => circ.num === progress + 1);
                    if (nextC) nextC.active = true;
                }
            }
        }
    }
}

function drawCircles() {
    if (state === 'WAITING_FOR_START' && startOrb) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(startOrb.x, startOrb.y, startOrb.r, 0, Math.PI * 2);
        ctx.fillStyle = `hsl(${(performance.now()/10) % 360}, 100%, 60%)`;
        ctx.shadowBlur = 30; ctx.shadowColor = ctx.fillStyle;
        ctx.fill(); ctx.lineWidth = 4; ctx.strokeStyle = '#fff'; ctx.stroke();
        
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 28px Outfit';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.shadowBlur = 0;
        ctx.fillText('START', startOrb.x, startOrb.y);
        ctx.restore();
    } else if (state === 'PLAYING') {
        circles.forEach(c => {
            if (c.num <= progress) return; 
            if (c.num > progress + 10) return; 
            
            ctx.save();
            ctx.beginPath();
            ctx.arc(c.x, c.y, c.r, 0, Math.PI * 2);
            
            if (c.num === progress + 1 && colorHintEnabled) {
                ctx.fillStyle = `hsl(${(performance.now()/10) % 360}, 100%, 60%)`;
                ctx.shadowBlur = 20; ctx.shadowColor = ctx.fillStyle;
                ctx.lineWidth = 4; ctx.strokeStyle = '#fff';
            } else {
                ctx.fillStyle = 'rgba(255,255,255,0.2)';
                ctx.shadowBlur = 0; ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(255,255,255,0.5)';
            }
            ctx.fill(); ctx.stroke();
            
            ctx.fillStyle = '#fff'; ctx.shadowBlur = 0;
            ctx.font = 'bold ' + (c.r) + 'px Outfit';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(c.num, c.x, c.y);
            ctx.restore();
        });
    }
}

function showResult() {
    els.inGameUI.classList.add('hidden');
    els.overlayResult.classList.remove('hidden');
    
    els.resTitle.innerText = "임무 완수!";
    let ms = finishTime;
    let s = Math.floor((ms / 1000) % 60);
    let m = Math.floor(ms / 60000);
    let fmt = `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${Math.floor((ms%1000)/10).toString().padStart(2, '0')}`;
    
    els.resDetails.innerHTML = `기록: <span style="color:var(--accent);font-weight:900;">${fmt}</span>`;
}

function renderLoop() {
    if (video && video.readyState >= 2) {
        if (lastHands.length > 0 || displayHands.length > 0) {
            displayHands = lerpHands(displayHands, lastHands);
        }
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state === 'PLAYING') {
        if (!finished) elapsedTime = performance.now() - startTime;
    }

    if (state === 'WAITING_FOR_START' || state === 'PLAYING') {
        updateUI();
        
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        const cw = canvas.width;
        const ch = canvas.height;

        let handPositions = getCanvasHandPositions();
        
        drawCircles();
        drawHandSkeleton(vw, vh, cw, ch);
        
        checkCollisions(handPositions);
    }

    requestAnimationFrame(renderLoop);
}

window.onload = init;
