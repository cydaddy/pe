/**
 * Beat Snap! - Timing Game (v3 rewrite)
 * 
 * Mechanic: A ball moves L↔R on a track.
 * Player's wrist position is shown as a hand cursor on screen.
 * When the ball enters the center TARGET ZONE, move your hand cursor
 * to overlap the ball to CATCH it. Timing + position = score.
 */

// ── Config ────────────────────────────────────────────────────────────────────
const DIFFICULTY = {
    easy:   { baseSpeed: 0.0003, speedInc: 0.00001, perfectR: 120, goodR: 200 },
    normal: { baseSpeed: 0.0005, speedInc: 0.00002, perfectR: 100, goodR: 160 },
    hard:   { baseSpeed: 0.0007, speedInc: 0.00003, perfectR: 80,  goodR: 130 },
};

// ── State ─────────────────────────────────────────────────────────────────────
let state = 'LOADING'; // LOADING | MENU | COUNTDOWN | PLAYING | RESULT
let selectedRounds = 10;
let selectedDifficulty = 'easy';

let currentRound = 0;
let score = 0;
let perfectCount = 0;
let goodCount = 0;
let missCount = 0;

// Ball
let ballPos = 0;       // 0.0 (left) ~ 1.0 (right)
let ballDir = 1;
let ballSpeed = 0;

// Round state
let roundPhase = 'WAITING'; // START_DELAY | WAITING | JUDGED
let roundEndTimer = 0;
let roundStartTimer = 0;
let judged = false;

// Countdown
let countdownStart = 0;
let prevCountdownVal = 99;

// Hand cursor (canvas px)
let handX = -999, handY = -999;
let handDetected = false;
let handWasInTargetArea = false;

// Audio
let audioCtx;

// Feedback
let feedbackText = '';
let feedbackColor = '#fff';
let feedbackAlpha = 0;
let feedbackTimer = 0;

// Canvas / Video / Detector
let canvas, ctx, video, detector;

// ── Audio ─────────────────────────────────────────────────────────────────────
function initAudio() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
}
function tone(freq, type, dur, vol = 0.5) {
    if (!audioCtx) return;
    const o = audioCtx.createOscillator(), g = audioCtx.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, audioCtx.currentTime);
    g.gain.setValueAtTime(vol, audioCtx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + dur);
    o.connect(g); g.connect(audioCtx.destination);
    o.start(); o.stop(audioCtx.currentTime + dur);
}
const sounds = {
    perfect: () => { tone(880,'sine',0.15,0.8); setTimeout(()=>tone(1100,'sine',0.2,0.8),80); },
    good:    () => tone(660,'sine',0.15,0.6),
    miss:    () => tone(180,'sawtooth',0.3,0.7),
    tick:    () => tone(440,'square',0.07,0.3),
    go:      () => { tone(600,'sine',0.1,0.5); setTimeout(()=>tone(800,'sine',0.2,0.5),120); },
};

// ── DOM ───────────────────────────────────────────────────────────────────────
const els = {
    loading:   document.getElementById('loading-overlay'),
    menu:      document.getElementById('start-menu'),
    result:    document.getElementById('result-overlay'),
    hud:       document.getElementById('in-game-ui'),
    retry:     document.getElementById('retry-btn'),
    startBtn:  document.getElementById('start-btn'),
    restartBtn:document.getElementById('restart-btn'),
    hudRound:  document.getElementById('hud-round'),
    hudScore:  document.getElementById('hud-score'),
    feedback:  document.getElementById('timing-feedback'),
    resTitle:  document.getElementById('result-title'),
    resDetails:document.getElementById('result-details'),
    resRank:   document.getElementById('result-rank'),
    loader:    document.querySelector('.loader'),
};

// Toggle buttons
document.querySelectorAll('[data-rounds]').forEach(b => {
    b.onclick = () => {
        document.querySelectorAll('[data-rounds]').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        selectedRounds = parseInt(b.dataset.rounds);
    };
});
document.querySelectorAll('[data-difficulty]').forEach(b => {
    b.onclick = () => {
        document.querySelectorAll('[data-difficulty]').forEach(x => x.classList.remove('active'));
        b.classList.add('active');
        selectedDifficulty = b.dataset.difficulty;
    };
});

// ── Camera ────────────────────────────────────────────────────────────────────
async function setupCamera() {
    video = document.getElementById('video');
    const secure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!secure) throw new Error('INSECURE_ORIGIN');
    if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width:{ideal:320}, height:{ideal:240}, facingMode:'user' }
    }).catch(e => { throw e.name === 'NotAllowedError' ? new Error('PERMISSION_DENIED') : e; });
    video.srcObject = stream;
    return new Promise(res => { video.onloadedmetadata = () => { video.play(); res(); }; });
}

// ── Model ─────────────────────────────────────────────────────────────────────
async function loadModel() {
    if (!window.tf)                throw new Error('SCRIPT_LOAD_ERROR: TensorFlow.js');
    if (!window.handPoseDetection) throw new Error('SCRIPT_LOAD_ERROR: hand-pose-detection');
    try { await tf.setBackend('webgl'); await tf.ready(); }
    catch { await tf.setBackend('cpu'); await tf.ready(); }

    try {
        detector = await handPoseDetection.createDetector(
            handPoseDetection.SupportedModels.MediaPipeHands,
            { 
                runtime: 'mediapipe', 
                solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915',
                maxHands: 1, 
                modelType: 'lite' 
            }
        );
    } catch(e) {
        // Fallback to tfjs runtime if mediapipe fails
        detector = await handPoseDetection.createDetector(
            handPoseDetection.SupportedModels.MediaPipeHands,
            { runtime: 'tfjs', maxHands: 1, modelType: 'lite' }
        );
    }
}

// ── Init ──────────────────────────────────────────────────────────────────────
let cameraReady = false, modelReady = false, loopsGo = false;

async function init() {
    const h2 = els.loading.querySelector('h2');
    const p  = els.loading.querySelector('p');
    els.loading.classList.remove('hidden');
    els.loader.classList.remove('hidden');
    els.retry.classList.add('hidden');
    h2.innerText = '카메라와 손 인식 모델을 불러오는 중...';
    p.innerText  = '잠시만 기다려주세요.';
    try {
        if (!cameraReady) { await setupCamera(); cameraReady = true; }
        if (!canvas) {
            canvas = document.getElementById('game-canvas');
            ctx    = canvas.getContext('2d');
            resizeCanvas();
            window.addEventListener('resize', resizeCanvas);
        }
        if (!modelReady) { h2.innerText = '손 인식 AI 로딩 중...'; await loadModel(); modelReady = true; }
        els.loading.classList.add('hidden');
        els.menu.classList.remove('hidden');
        state = 'MENU';
        if (!loopsGo) { loopsGo = true; inferLoop(); requestAnimationFrame(renderLoop); }
    } catch(e) {
        console.error(e);
        h2.innerText = '오류가 발생했습니다';
        els.loader.classList.add('hidden');
        els.retry.classList.remove('hidden');
        if (e.message === 'INSECURE_ORIGIN')    p.innerHTML = 'HTTPS 또는 localhost에서 실행해주세요.';
        else if (e.message === 'PERMISSION_DENIED') p.innerText = '카메라 권한을 허용해주세요.';
        else p.innerHTML = '오류: <small>' + e.message + '</small>';
    }
}
els.retry.onclick = init;

function resizeCanvas() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }

// ── Game Flow ─────────────────────────────────────────────────────────────────
els.startBtn.onclick = () => { initAudio(); startGame(); };
els.restartBtn.onclick = () => {
    els.result.classList.add('hidden');
    els.menu.classList.remove('hidden');
    state = 'MENU';
};

function startGame() {
    els.menu.classList.add('hidden');
    els.hud.classList.remove('hidden');
    score = perfectCount = goodCount = missCount = currentRound = 0;
    ballPos = 0; ballDir = 1;
    state = 'COUNTDOWN';
    countdownStart = performance.now();
    prevCountdownVal = 99;
    updateHUD();
}

function beginRound() {
    const cfg = DIFFICULTY[selectedDifficulty];
    currentRound++;
    
    // Randomize speed slightly (0.9x ~ 1.1x)
    const speedMod = 0.9 + Math.random() * 0.2;
    ballSpeed = (cfg.baseSpeed + cfg.speedInc * (currentRound - 1)) * speedMod;
    
    // Randomize starting side (Left = 0, Right = 1)
    if (Math.random() > 0.5) {
        ballPos = 0; ballDir = 1;
    } else {
        ballPos = 1; ballDir = -1;
    }
    
    roundPhase = 'START_DELAY';
    // Random delay between 0.3s and 1.2s before the ball starts moving
    roundStartTimer = performance.now() + 300 + Math.random() * 900;
    
    judged = false;
    updateHUD();
}

function updateHUD() {
    els.hudRound.innerText = currentRound + ' / ' + selectedRounds;
    els.hudScore.innerText = score;
}

function showFeedback(text, color) {
    feedbackText  = text;
    feedbackColor = color;
    feedbackAlpha = 1;
    feedbackTimer = performance.now() + 800;
}

function judge(dist) {
    if (judged) return;
    judged = true;
    const cfg = DIFFICULTY[selectedDifficulty];
    if (dist < cfg.perfectR) {
        score += 3; perfectCount++;
        showFeedback('⭐ PERFECT!', '#FFD700');
        sounds.perfect();
    } else if (dist < cfg.goodR) {
        score += 1; goodCount++;
        showFeedback('✅ GOOD', '#20D68A');
        sounds.good();
    } else {
        missCount++;
        showFeedback('❌ MISS', '#FF5A5F');
        sounds.miss();
    }
    updateHUD();
    roundPhase = 'JUDGED';
    roundEndTimer = performance.now() + 1100;
}

function showResult() {
    els.hud.classList.add('hidden');
    els.result.classList.remove('hidden');
    state = 'RESULT';
    const pct = Math.round((perfectCount / selectedRounds) * 100);
    let rc, rk;
    if      (pct >= 80) { rc='rank-s'; rk='S'; }
    else if (pct >= 60) { rc='rank-a'; rk='A'; }
    else if (pct >= 40) { rc='rank-b'; rk='B'; }
    else                { rc='rank-c'; rk='C'; }
    els.resTitle.innerText = '결과!';
    els.resRank.className = 'result-rank ' + rc;
    els.resRank.innerText = rk;
    els.resDetails.innerHTML =
        `총 점수: <strong style="color:var(--accent)">${score}점</strong><br>` +
        `⭐ PERFECT: <strong>${perfectCount}</strong>&nbsp; ` +
        `✅ GOOD: <strong>${goodCount}</strong>&nbsp; ` +
        `❌ MISS: <strong>${missCount}</strong>`;
}

// ── Inference Loop ────────────────────────────────────────────────────────────
let inferRunning = false;
let lastHands = [];

async function inferLoop() {
    if (inferRunning) return;
    inferRunning = true;
    while (loopsGo) {
        if (video && video.readyState >= 2 && detector) {
            try {
                // flipHorizontal:false → we mirror manually for display
                lastHands = await detector.estimateHands(video, { flipHorizontal: false });
            } catch(_) { lastHands = []; }
        }
        await new Promise(r => setTimeout(r, 50));
    }
    inferRunning = false;
}

// Helper to map video coordinates to canvas coordinates (considering object-fit: cover)
function mapVideoToCanvas(vx, vy, vw, vh, cw, ch) {
    const videoRatio = vw / vh;
    const canvasRatio = cw / ch;
    let renderW, renderH, offsetX = 0, offsetY = 0;

    if (canvasRatio > videoRatio) {
        renderW = cw;
        renderH = cw / videoRatio;
        offsetY = (ch - renderH) / 2;
    } else {
        renderH = ch;
        renderW = ch * videoRatio;
        offsetX = (cw - renderW) / 2;
    }

    const x = offsetX + (vx / vw) * renderW;
    const y = offsetY + (vy / vh) * renderH;
    return { x, y };
}

// Map hand keypoints → canvas coordinates (mirrored)
function updateHandCursor() {
    if (!lastHands || lastHands.length === 0) { handDetected = false; return; }
    const hand = lastHands[0];
    if (!hand.keypoints || hand.keypoints.length === 0) { handDetected = false; return; }

    const vw = video.videoWidth  || 320;
    const vh = video.videoHeight || 240;
    const cw = canvas.width;
    const ch = canvas.height;

    // Use wrist (0) + middle finger base (9) to get palm center
    const kp = hand.keypoints;
    const wrist = kp[0], mBase = kp[9];
    if (!wrist) { handDetected = false; return; }

    const rawX = mBase ? (wrist.x + mBase.x) / 2 : wrist.x;
    const rawY = mBase ? (wrist.y + mBase.y) / 2 : wrist.y;

    // Mirror X coordinate before mapping
    const mirroredX = vw - rawX;
    
    const mapped = mapVideoToCanvas(mirroredX, rawY, vw, vh, cw, ch);
    handX = mapped.x;
    handY = mapped.y;
    handDetected = true;
}

// Draw hand skeleton on canvas (mirrored)
function drawHandSkeleton() {
    if (!lastHands || lastHands.length === 0) return;
    const hand = lastHands[0];
    if (!hand.keypoints) return;

    const vw = video.videoWidth  || 320;
    const vh = video.videoHeight || 240;
    const cw = canvas.width;
    const ch = canvas.height;
    const kps = hand.keypoints;

    const connections = [
        [0,1],[1,2],[2,3],[3,4],
        [0,5],[5,6],[6,7],[7,8],
        [5,9],[9,10],[10,11],[11,12],
        [9,13],[13,14],[14,15],[15,16],
        [13,17],[17,18],[18,19],[19,20],[0,17]
    ];

    ctx.save();
    ctx.strokeStyle = 'rgba(30,167,253,0.9)';
    ctx.lineWidth = 4;
    connections.forEach(([a,b]) => {
        if (!kps[a] || !kps[b]) return;
        const p1 = mapVideoToCanvas(vw - kps[a].x, kps[a].y, vw, vh, cw, ch);
        const p2 = mapVideoToCanvas(vw - kps[b].x, kps[b].y, vw, vh, cw, ch);
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.stroke();
    });
    kps.forEach((kp, i) => {
        if (!kp) return;
        const p = mapVideoToCanvas(vw - kp.x, kp.y, vw, vh, cw, ch);
        ctx.beginPath();
        ctx.arc(p.x, p.y, i === 0 ? 10 : 6, 0, Math.PI * 2);
        ctx.fillStyle = i === 0 ? '#FFAE00' : '#1EA7FD';
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
    ctx.restore();
}

// ── Render Loop ───────────────────────────────────────────────────────────────
let lastTime = 0;

function renderLoop(now) {
    requestAnimationFrame(renderLoop);
    const dt = now - lastTime || 16;
    lastTime = now;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Always update hand cursor and draw skeleton (visible even in MENU)
    updateHandCursor();
    if (state !== 'LOADING') drawHandSkeleton();

    // ── MENU: show skeleton + "손을 인식 중..." hint ──
    if (state === 'MENU') {
        if (handDetected) {
            // Draw palm cursor as glowing circle
            ctx.save();
            ctx.beginPath();
            ctx.arc(handX, handY, 24, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(32,214,138,0.3)';
            ctx.strokeStyle = '#20D68A';
            ctx.lineWidth = 3;
            ctx.shadowBlur = 20; ctx.shadowColor = '#20D68A';
            ctx.fill(); ctx.stroke();
            ctx.restore();
        }
        return;
    }

    // ── COUNTDOWN ──
    if (state === 'COUNTDOWN') {
        const elapsed = now - countdownStart;
        const val = 3 - Math.floor(elapsed / 1000);
        if (val !== prevCountdownVal) { prevCountdownVal = val; if(val > 0) sounds.tick(); }
        if (val <= 0) {
            state = 'PLAYING';
            beginRound();
            sounds.go();
        } else {
            ctx.save();
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.font = 'bold 240px Outfit';
            ctx.fillStyle = 'rgba(255,255,255,0.12)';
            ctx.fillText(val, canvas.width/2, canvas.height/2);
            ctx.restore();
        }
        return;
    }

    if (state !== 'PLAYING') return;

    // ── Move ball ──
    if (roundPhase === 'START_DELAY') {
        if (now >= roundStartTimer) {
            roundPhase = 'WAITING';
            sounds.tick(); // Audio cue that the ball has launched
        }
    } else if (roundPhase === 'WAITING') {
        ballPos += ballDir * ballSpeed * dt;
        if (ballPos >= 1) { ballPos = 1; ballDir = -1; }
        if (ballPos <= 0) { ballPos = 0; ballDir = 1; }
    }

    // ── Layout ──
    const cw = canvas.width, ch = canvas.height;
    const cx = cw / 2;
    const railY = ch * 0.5;
    const margin = cw * 0.1;
    const railW = cw - margin * 2;

    const cfg = DIFFICULTY[selectedDifficulty];
    const ballX = margin + ballPos * railW;
    const ballR = 38;
    const distFromCenter = Math.abs(ballPos - 0.5) * railW; // px from center
    const inGoodZone    = distFromCenter < cfg.goodR;
    const inPerfectZone = distFromCenter < cfg.perfectR;

    // ── Draw rail ──
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(margin, railY); ctx.lineTo(margin + railW, railY); ctx.stroke();
    ctx.restore();

    // ── Draw target zones ──
    // Good zone
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(cx - cfg.goodR, railY - 60, cfg.goodR * 2, 120, 14);
    ctx.fillStyle = 'rgba(32,214,138,0.10)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(32,214,138,0.45)';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();

    // Perfect zone
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(cx - cfg.perfectR, railY - 60, cfg.perfectR * 2, 120, 10);
    ctx.fillStyle = 'rgba(255,215,0,0.18)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,215,0,0.75)';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    ctx.restore();

    // Zone labels
    ctx.save();
    ctx.textAlign = 'center'; ctx.font = 'bold 15px Outfit';
    ctx.fillStyle = 'rgba(255,215,0,0.7)';
    ctx.fillText('PERFECT', cx, railY - 68);
    ctx.fillStyle = 'rgba(32,214,138,0.6)';
    // Place 'GOOD' text exactly aligned with the outer edges of the good zone
    ctx.fillText('GOOD', cx - cfg.goodR, railY - 68);
    ctx.fillText('GOOD', cx + cfg.goodR, railY - 68);
    ctx.restore();

    // ── Draw ball ──
    const hue = (now / 8) % 360;
    ctx.save();
    ctx.beginPath();
    ctx.arc(ballX, railY, ballR, 0, Math.PI * 2);
    ctx.fillStyle = inPerfectZone
        ? `hsl(${hue},100%,65%)`
        : inGoodZone
            ? `hsl(${(hue+120)%360},100%,65%)`
            : 'rgba(255,255,255,0.88)';
    if (inGoodZone) { ctx.shadowBlur = 45; ctx.shadowColor = ctx.fillStyle; }
    ctx.fill();
    ctx.restore();

    // ── Direction arrow on ball ──
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.font = '22px sans-serif';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    if (roundPhase === 'START_DELAY') {
        // Blink exclamation mark while waiting to launch
        if (Math.floor(now / 150) % 2 === 0) {
            ctx.fillStyle = '#FF5A5F';
            ctx.fillText('!', ballX, railY - 54);
        }
    } else {
        ctx.fillText(ballDir > 0 ? '→' : '←', ballX, railY - 54);
    }
    ctx.restore();

    // ── "CATCH!" blinking hint ──
    if (inGoodZone && roundPhase === 'WAITING') {
        const pulse = 1 + 0.07 * Math.sin(now / 70);
        ctx.save();
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.font = `bold ${Math.round(56 * pulse)}px Outfit`;
        ctx.fillStyle = inPerfectZone ? '#FFD700' : '#20D68A';
        ctx.shadowBlur = 25; ctx.shadowColor = ctx.fillStyle;
        ctx.globalAlpha = 0.9;
        ctx.fillText('✋ CATCH!', cx, railY + 100);
        ctx.restore();
    }

    // ── Hand skeleton + cursor ──
    drawHandSkeleton();

    const handDistToCenter = Math.sqrt((handX - cx) ** 2 + (handY - railY) ** 2);
    const handInTargetArea = handDistToCenter < cfg.goodR + 30; // Hand is inside the general target region

    if (handDetected) {
        ctx.save();
        ctx.beginPath();
        ctx.arc(handX, handY, 28, 0, Math.PI * 2);
        ctx.fillStyle = handInTargetArea && inGoodZone ? 'rgba(32,214,138,0.35)' : 'rgba(255,255,255,0.12)';
        ctx.strokeStyle = handInTargetArea && inGoodZone ? '#20D68A' : 'rgba(255,255,255,0.5)';
        ctx.lineWidth = 3;
        if (handInTargetArea && inGoodZone) { ctx.shadowBlur = 30; ctx.shadowColor = '#20D68A'; }
        ctx.fill(); ctx.stroke();
        ctx.restore();
    } else {
        // "No hand" hint top-center
        ctx.save();
        ctx.textAlign = 'center'; ctx.font = 'bold 18px Outfit';
        ctx.fillStyle = 'rgba(255,90,95,0.85)';
        ctx.fillText('✋ 손을 카메라에 보여주세요', cx, 50);
        ctx.restore();
    }

    // ── Judge: hand MUST enter the target area ──
    if (roundPhase === 'WAITING' && handDetected) {
        // Did the hand just move INTO the target area this frame?
        if (handInTargetArea && !handWasInTargetArea) {
            if (inGoodZone) {
                // Ball is here! Catch!
                judge(distFromCenter);
            } else {
                // Snapped into the zone, but ball is NOT here! Miss!
                judge(999);
            }
        }
    }

    // Update state for next frame
    if (handDetected) {
        handWasInTargetArea = handInTargetArea;
    } else {
        handWasInTargetArea = false;
    }

    // ── Auto-miss: ball has passed zone without being caught ──
    if (roundPhase === 'WAITING' && !inGoodZone && ballPos > 0.5 + cfg.goodR / railW && ballDir === 1) {
        judge(999);
    }
    if (roundPhase === 'WAITING' && !inGoodZone && ballPos < 0.5 - cfg.goodR / railW && ballDir === -1) {
        judge(999);
    }

    // ── Next round transition ──
    if (roundPhase === 'JUDGED' && now >= roundEndTimer) {
        if (currentRound >= selectedRounds) {
            showResult();
        } else {
            beginRound();
        }
    }

    // ── Feedback text ──
    if (feedbackAlpha > 0) {
        feedbackAlpha = Math.max(0, 1 - (now - (feedbackTimer - 800)) / 600);
        ctx.save();
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.font = 'bold 64px Outfit';
        ctx.fillStyle = feedbackColor;
        ctx.globalAlpha = feedbackAlpha;
        ctx.shadowBlur = 30; ctx.shadowColor = feedbackColor;
        ctx.fillText(feedbackText, cx, ch * 0.28);
        ctx.restore();
    }
}

// ── Boot ─────────────────────────────────────────────────────────────────────
window.onload = init;
