/**
 * 무궁화 꽃이 피었습니다 — AI 체육 게임
 * number-pop의 TF.js 엔진(카메라/모델/추론/렌더링)을 기반으로 재구성
 */

// ═══════════════════════════════════════════════
// [1] 전역 상태
// ═══════════════════════════════════════════════
let state = 'LOADING'; // LOADING | MENU | CALIBRATION | PLAYING | RESULT
let selectedMode = 'approach'; // approach | squat | pose | jump
let selectedDiff = 'normal';
let targetCount  = 15;

// ═══════════════════════════════════════════════
// [2] 난이도 설정값 테이블
// ═══════════════════════════════════════════════
const DIFFICULTY = {
    easy:   { greenMin:5, greenMax:8, redMin:2, redMax:3, graceMs:1000, moveSens:0.018, sqAngle:100, gameSec:90  },
    normal: { greenMin:3, greenMax:7, redMin:2, redMax:5, graceMs:500,  moveSens:0.013, sqAngle:90,  gameSec:60  },
    hard:   { greenMin:2, greenMax:5, redMin:3, redMax:7, graceMs:300,  moveSens:0.009, sqAngle:80,  gameSec:45  }
};

// ═══════════════════════════════════════════════
// [3] TF.js / 카메라 변수 (number-pop 동일)
// ═══════════════════════════════════════════════
let video, canvas, ctx;
let detector;
let currentModel = null;

let cameraReady   = false;
let modelReady    = false;
let renderLoopStarted = false;
let inferLoopStarted  = false;
let resizeAdded   = false;

let lastPoses    = [];
let displayPoses = [];
const LERP_ALPHA = 0.2;

// ═══════════════════════════════════════════════
// [4] 오디오 (number-pop 기반 + 신호등 효과음)
// ═══════════════════════════════════════════════
let audioCtx;

function initAudio() {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    if (audioCtx.state === 'suspended') audioCtx.resume();
}

function playTone(freq, type, duration, vol = 0.5) {
    if (!audioCtx) return;
    const osc  = audioCtx.createOscillator();
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
    clap:    () => { playTone(1000,'square',0.1,0.3); playTone(1200,'sine',0.15,0.3); },
    greenGo: () => { playTone(523,'sine',0.15,0.6); setTimeout(()=>playTone(659,'sine',0.2,0.6),100); },
    warnBeep:() => { playTone(800,'square',0.08,0.4); setTimeout(()=>playTone(800,'square',0.08,0.4),160); setTimeout(()=>playTone(800,'square',0.08,0.4),320); },
    redStop: () => { playTone(180,'sawtooth',0.4,0.8); },
    caught:  () => { playTone(120,'sawtooth',0.5,0.5); setTimeout(()=>playTone(80,'sawtooth',0.4,0.4),200); },
    count:   () => { playTone(660,'sine',0.12,0.5); },
    win:     () => { [400,500,600,700,800].forEach((f,i)=>setTimeout(()=>playTone(f,'sine',0.3,0.6),i*120)); },
    fail:    () => { playTone(200,'sawtooth',0.6,0.8); }
};

// ═══════════════════════════════════════════════
// [5] DOM 참조
// ═══════════════════════════════════════════════
const els = {
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingMsg:     document.getElementById('loading-msg'),
    retryBtn:       document.getElementById('retry-btn'),
    startMenu:      document.getElementById('start-menu'),
    startBtn:       document.getElementById('start-btn'),
    inGameUI:       document.getElementById('in-game-ui'),
    resultOverlay:  document.getElementById('result-overlay'),
    resultTitle:    document.getElementById('result-title'),
    resultDetails:  document.getElementById('result-details'),
    restartBtn:     document.getElementById('restart-btn'),
    menuBtn:        document.getElementById('menu-btn'),
    // HUD
    hudModeLabel:   document.getElementById('hud-mode-label'),
    hudTimer:       document.getElementById('hud-timer'),
    hudScore:       document.getElementById('hud-score'),
    // In-game elements
    lightBanner:    document.getElementById('light-banner'),
    centerWarning:  document.getElementById('center-warning'),
    centerMessage:  document.getElementById('center-message'),
    poseTarget:     document.getElementById('pose-target'),
    poseIcon:       document.getElementById('pose-icon'),
    poseLabel:      document.getElementById('pose-label'),
    progressWrap:   document.getElementById('progress-bar-wrap'),
    progressFill:   document.getElementById('progress-bar-fill'),
    progressLabel:  document.getElementById('progress-bar-label'),
    gameContainer:  document.getElementById('game-container'),
};

// ═══════════════════════════════════════════════
// [6] 카메라 + 모델 초기화 (number-pop 동일)
// ═══════════════════════════════════════════════
async function setupCamera() {
    video = document.getElementById('video');
    const isSecure = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
    if (!isSecure) throw new Error('INSECURE_ORIGIN');
    if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: false,
            video: { width: { ideal: 320 }, height: { ideal: 240 }, facingMode: 'user' }
        });
        video.srcObject = stream;
        return new Promise(resolve => { video.onloadedmetadata = () => { video.play(); resolve(video); }; });
    } catch (err) {
        if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') throw new Error('PERMISSION_DENIED');
        throw err;
    }
}

async function loadModel() {
    if (typeof tf === 'undefined') throw new Error('SCRIPT_LOAD_ERROR: @tensorflow/tfjs 스크립트 로드 실패');
    if (typeof poseDetection === 'undefined') throw new Error('SCRIPT_LOAD_ERROR: pose-detection 스크립트 로드 실패');
    try { await tf.setBackend('webgl'); await tf.ready(); }
    catch { await tf.setBackend('cpu'); await tf.ready(); }
    try {
        detector = await poseDetection.createDetector(
            poseDetection.SupportedModels.MoveNet,
            { modelType: poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING }
        );
        currentModel = poseDetection.SupportedModels.MoveNet;
        return;
    } catch {}
    detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.BlazePose,
        { runtime: 'mediapipe', solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5.1675469404', modelType: 'lite' }
    );
    currentModel = poseDetection.SupportedModels.BlazePose;
}

async function init() {
    els.loadingOverlay.classList.remove('hidden');
    els.loadingOverlay.classList.add('active');
    els.retryBtn.classList.add('hidden');
    els.loadingMsg.innerText = '잠시만 기다려주세요.';
    try {
        if (!cameraReady) { await setupCamera(); cameraReady = true; }
        if (!canvas) {
            canvas = document.getElementById('game-canvas');
            ctx = canvas.getContext('2d');
            resizeCanvas();
        }
        if (!resizeAdded) { window.addEventListener('resize', resizeCanvas); resizeAdded = true; }
        if (!modelReady) { els.loadingMsg.innerText = 'AI 모델 데이터 로딩 중...'; await loadModel(); modelReady = true; }
        els.loadingOverlay.classList.add('hidden');
        els.startMenu.classList.remove('hidden');
        state = 'MENU';
        if (!renderLoopStarted) { renderLoopStarted = true; requestAnimationFrame(renderLoop); inferLoop(); }
    } catch (e) {
        els.retryBtn.classList.remove('hidden');
        const p = els.loadingMsg;
        if (e.message === 'INSECURE_ORIGIN') p.innerText = 'HTTPS 또는 localhost 환경에서 실행해주세요.';
        else if (e.message === 'PERMISSION_DENIED') p.innerText = '카메라 권한을 허용해주세요.';
        else p.innerHTML = '오류: <small>' + e.message + '</small>';
    }
}

els.retryBtn.onclick = init;

function resizeCanvas() {
    canvas.width  = window.innerWidth;
    canvas.height = window.innerHeight;
}

// ═══════════════════════════════════════════════
// [7] 추론 루프 (number-pop 동일)
// ═══════════════════════════════════════════════
async function inferLoop() {
    if (inferLoopStarted) return;
    inferLoopStarted = true;
    while (renderLoopStarted) {
        if ((state === 'CALIBRATION' || state === 'PLAYING') && video && video.readyState >= 2) {
            try {
                const raw = await detector.estimatePoses(video, { maxPoses: 1, flipHorizontal: false });
                const vw = video.videoWidth;
                lastPoses = raw.map(pose => ({
                    ...pose,
                    keypoints: pose.keypoints.map(kp => ({ ...kp, x: vw - kp.x }))
                }));
            } catch {}
        }
        await new Promise(r => setTimeout(r, 50));
    }
    inferLoopStarted = false;
}

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
                return { ...kp, x: pkp.x + (kp.x - pkp.x) * LERP_ALPHA, y: pkp.y + (kp.y - pkp.y) * LERP_ALPHA };
            })
        };
    });
}

// ═══════════════════════════════════════════════
// [8] 스켈레톤 렌더링 (number-pop 동일)
// ═══════════════════════════════════════════════
function drawSkeleton(keypoints, scaleX, scaleY, color = '#FFFFFF') {
    const pairs = poseDetection.util.getAdjacentPairs(currentModel || poseDetection.SupportedModels.MoveNet);
    ctx.save();
    ctx.lineWidth = 7;
    ctx.strokeStyle = color;
    ctx.fillStyle = '#1EA7FD';
    pairs.forEach(([i, j]) => {
        const kp1 = keypoints[i], kp2 = keypoints[j];
        if (kp1.score >= 0.3 && kp2.score >= 0.3) {
            ctx.beginPath();
            ctx.moveTo(kp1.x * scaleX, kp1.y * scaleY);
            ctx.lineTo(kp2.x * scaleX, kp2.y * scaleY);
            ctx.stroke();
        }
    });
    keypoints.forEach(kp => {
        if (kp.score >= 0.3) {
            ctx.beginPath();
            ctx.arc(kp.x * scaleX, kp.y * scaleY, 7, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = color;
            ctx.stroke();
        }
    });
    const nose = keypoints.find(k => k.name === 'nose');
    if (nose && nose.score >= 0.3) {
        ctx.beginPath();
        ctx.arc(nose.x * scaleX, nose.y * scaleY, 22, 0, Math.PI * 2);
        ctx.fillStyle = '#FFAE00';
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.stroke();
    }
    ctx.restore();
}

// ═══════════════════════════════════════════════
// [9] 유틸: 관절 각도 계산 (3점)
// ═══════════════════════════════════════════════
function angleBetween(A, B, C) {
    // B가 꼭짓점
    const AB = { x: A.x - B.x, y: A.y - B.y };
    const CB = { x: C.x - B.x, y: C.y - B.y };
    const dot = AB.x * CB.x + AB.y * CB.y;
    const mag = Math.sqrt((AB.x**2 + AB.y**2) * (CB.x**2 + CB.y**2));
    if (mag === 0) return 180;
    return Math.acos(Math.max(-1, Math.min(1, dot / mag))) * (180 / Math.PI);
}

function kp(keypoints, name) { return keypoints.find(k => k.name === name); }

// ═══════════════════════════════════════════════
// [10] 박수 인식 (number-pop 동일)
// ═══════════════════════════════════════════════
function checkClap(keypoints) {
    const lw = kp(keypoints, 'left_wrist');
    const rw = kp(keypoints, 'right_wrist');
    if (lw && rw && lw.score > 0.3 && rw.score > 0.3) {
        const dist = Math.hypot(lw.x - rw.x, lw.y - rw.y);
        return dist < 55;
    }
    return false;
}

// ═══════════════════════════════════════════════
// [11] TrafficLight 클래스 (신호등 상태 머신)
// ═══════════════════════════════════════════════
class TrafficLight {
    constructor() { this.phase='green'; this.graceActive=false; this._timer=null; }
    start() { this.phase='green'; this._scheduleGreen(); }
    stop()  { clearTimeout(this._timer); this._timer=null; }
    get isRed()   { return this.phase==='red'; }
    get isGreen() { return this.phase==='green'; }
    get isWarn()  { return this.phase==='warn'; }
    _cfg() { return DIFFICULTY[selectedDiff]; }
    _scheduleGreen() {
        const cfg=this._cfg();
        const ms=(cfg.greenMin+Math.random()*(cfg.greenMax-cfg.greenMin))*1000;
        this.phase='green'; this.graceActive=false; updateLightUI(); sounds.greenGo();
        this._timer=setTimeout(()=>this._toWarn(),ms);
    }
    _toWarn() {
        this.phase='warn'; updateLightUI(); sounds.warnBeep();
        this._timer=setTimeout(()=>this._toRed(),1000);
    }
    _toRed() {
        this.phase='red'; this.graceActive=true; updateLightUI(); sounds.redStop();
        setTimeout(()=>{ this.graceActive=false; },this._cfg().graceMs);
        const cfg=this._cfg();
        const ms=(cfg.redMin+Math.random()*(cfg.redMax-cfg.redMin))*1000;
        this._timer=setTimeout(()=>this._scheduleGreen(),ms);
    }
}
const trafficLight=new TrafficLight();

function updateLightUI() {
    const gc=els.gameContainer;
    gc.classList.remove('state-green','state-warn','state-red');
    const b=els.lightBanner;
    b.classList.remove('light-green','light-warn','light-red');
    if(trafficLight.isGreen){ gc.classList.add('state-green'); b.classList.add('light-green'); b.innerHTML='🟢 움직여!'; }
    else if(trafficLight.isWarn){ gc.classList.add('state-warn'); b.classList.add('light-warn'); b.innerHTML='⚠️ 곧 멈춰!'; }
    else { gc.classList.add('state-red'); b.classList.add('light-red'); b.innerHTML='🔴 멈춰!'; }
}

// ═══════════════════════════════════════════════
// [12] MovementDetector
// ═══════════════════════════════════════════════
class MovementDetector {
    constructor() { this.history=[]; this.CHECK=['nose','left_shoulder','right_shoulder','left_hip','right_hip']; }
    update(kps) { this.history.push(kps); if(this.history.length>3) this.history.shift(); }
    getMovement(vw,vh) {
        if(this.history.length<2) return 0;
        const prev=this.history[this.history.length-2], curr=this.history[this.history.length-1];
        let mx=0;
        this.CHECK.forEach(n=>{
            const a=prev.find(k=>k.name===n), b=curr.find(k=>k.name===n);
            if(a&&b&&a.score>0.3&&b.score>0.3) mx=Math.max(mx,Math.hypot((b.x-a.x)/vw,(b.y-a.y)/vh));
        });
        return mx;
    }
    reset() { this.history=[]; }
}
const moveDetector=new MovementDetector();

// ═══════════════════════════════════════════════
// [13] 게임 전역 변수
// ═══════════════════════════════════════════════
let score=0, timeLeft=60, gameTimer=null;
let initialShoulderDist=null, prevShoulderDist=null, runWarningFrames=0;
let squatPhase='STANDING', jumpJackPhase='CLOSED';

const POSES=[
    {icon:'🙌',label:'양팔 만세!',check:kps=>{
        const lw=kp(kps,'left_wrist'),rw=kp(kps,'right_wrist'),ls=kp(kps,'left_shoulder'),rs=kp(kps,'right_shoulder');
        return lw&&rw&&ls&&rs&&lw.y<ls.y&&rw.y<rs.y;
    }},
    {icon:'💪',label:'오른팔 올려!',check:kps=>{
        const rw=kp(kps,'right_wrist'),rs=kp(kps,'right_shoulder'),lw=kp(kps,'left_wrist'),ls=kp(kps,'left_shoulder');
        return rw&&rs&&lw&&ls&&rw.y<rs.y&&lw.y>ls.y;
    }},
    {icon:'✈️',label:'T자 포즈!',check:kps=>{
        const lw=kp(kps,'left_wrist'),rw=kp(kps,'right_wrist'),ls=kp(kps,'left_shoulder'),rs=kp(kps,'right_shoulder');
        return lw&&rw&&ls&&rs&&Math.abs(lw.x-rw.x)>Math.abs(ls.x-rs.x)*1.8&&Math.abs(lw.y-ls.y)<60&&Math.abs(rw.y-rs.y)<60;
    }},
    {icon:'🧎',label:'쪼그려 앉기!',check:kps=>{
        const lh=kp(kps,'left_hip'),lk=kp(kps,'left_knee'),la=kp(kps,'left_ankle');
        return lh&&lk&&la&&lh.score>0.3&&lk.score>0.3&&la.score>0.3&&angleBetween(lh,lk,la)<110;
    }},
];
let currentPose=null,poseTimerId=null,poseSuccess=false,poseCooldown=false;

// ═══════════════════════════════════════════════
// [14] 경고 표시
// ═══════════════════════════════════════════════
let warningTimeout=null;
function showCaught(msg='💥 걸렸다!') {
    sounds.caught(); els.centerWarning.innerText=msg; els.centerWarning.classList.remove('hidden');
    clearTimeout(warningTimeout); warningTimeout=setTimeout(()=>els.centerWarning.classList.add('hidden'),1600);
}

// ═══════════════════════════════════════════════
// [15] 어깨 너비 헬퍼
// ═══════════════════════════════════════════════
function getShoulderDist(kps,sx,sy) {
    const ls=kp(kps,'left_shoulder'),rs=kp(kps,'right_shoulder');
    if(!ls||!rs||ls.score<0.3||rs.score<0.3) return null;
    return Math.hypot((ls.x-rs.x)*sx,(ls.y-rs.y)*sy);
}

// ═══════════════════════════════════════════════
// [16] 모드별 처리
// ═══════════════════════════════════════════════
function processApproach(kps,sx,sy) {
    if(trafficLight.isRed&&!trafficLight.graceActive) {
        if(moveDetector.getMovement(video.videoWidth,video.videoHeight)>DIFFICULTY[selectedDiff].moveSens) {
            initialShoulderDist=null; showCaught('💥 걸렸다! 처음부터!');
            els.progressFill.style.width='0%'; els.progressLabel.innerText='0%';
            setTimeout(()=>{ initialShoulderDist=getShoulderDist(kps,sx,sy); },600);
        }
        return;
    }
    if(!trafficLight.isGreen) return;
    const dist=getShoulderDist(kps,sx,sy);
    if(dist&&prevShoulderDist&&Math.abs(dist-prevShoulderDist)>15){ runWarningFrames=90; }
    prevShoulderDist=dist;
    if(runWarningFrames>0){ showCaught('🚫 뛰지 마세요!'); initialShoulderDist=null; runWarningFrames--; return; }
    if(!initialShoulderDist&&dist) initialShoulderDist=dist;
    if(initialShoulderDist&&dist) {
        const pct=Math.max(0,Math.min(100,((dist-initialShoulderDist)/(initialShoulderDist*0.8))*100));
        els.progressFill.style.width=pct+'%'; els.progressLabel.innerText=Math.round(pct)+'%';
    }
}

function processSquat(kps,sx,sy) {
    const cfg=DIFFICULTY[selectedDiff];
    if(trafficLight.isRed&&!trafficLight.graceActive) {
        if(moveDetector.getMovement(video.videoWidth,video.videoHeight)>cfg.moveSens) {
            score=Math.max(0,score-3); els.hudScore.innerText=score; showCaught('💥 -3회!');
        }
        return;
    }
    if(!trafficLight.isGreen) return;
    const lh=kp(kps,'left_hip'),lk=kp(kps,'left_knee'),la=kp(kps,'left_ankle');
    const rh=kp(kps,'right_hip'),rk=kp(kps,'right_knee'),ra=kp(kps,'right_ankle');
    if(!lh||!lk||!la||!rh||!rk||!ra||lh.score<0.3||lk.score<0.3||la.score<0.3) return;
    const s=k=>({x:k.x*sx,y:k.y*sy});
    const avg=(angleBetween(s(lh),s(lk),s(la))+angleBetween(s(rh),s(rk),s(ra)))/2;
    if(squatPhase==='STANDING'&&avg<cfg.sqAngle) squatPhase='SQUATTING';
    else if(squatPhase==='SQUATTING'&&avg>160) {
        squatPhase='STANDING'; score++; sounds.count(); els.hudScore.innerText=score;
        if(score>=targetCount) endGame(true);
    }
}

function processJumpJack(kps,sx,sy) {
    const cfg=DIFFICULTY[selectedDiff];
    if(trafficLight.isRed&&!trafficLight.graceActive) {
        if(moveDetector.getMovement(video.videoWidth,video.videoHeight)>cfg.moveSens) {
            score=Math.max(0,score-2); els.hudScore.innerText=score; showCaught('💥 -2회!');
        }
        return;
    }
    if(!trafficLight.isGreen) return;
    const lw=kp(kps,'left_wrist'),rw=kp(kps,'right_wrist'),ls=kp(kps,'left_shoulder'),rs=kp(kps,'right_shoulder');
    const la=kp(kps,'left_ankle'),ra=kp(kps,'right_ankle'),lh=kp(kps,'left_hip');
    if(!lw||!rw||!ls||!rs||!la||!ra||!lh||lw.score<0.3||rw.score<0.3) return;
    const sw=Math.abs((ls.x-rs.x)*sx),aw=Math.abs((la.x-ra.x)*sx);
    const up=lw.y*sy<ls.y*sy&&rw.y*sy<rs.y*sy, wide=aw>sw;
    const dn=lw.y*sy>lh.y*sy&&rw.y*sy>lh.y*sy, cls=aw<sw*0.5;
    if(jumpJackPhase==='CLOSED'&&up&&wide) jumpJackPhase='OPEN';
    else if(jumpJackPhase==='OPEN'&&dn&&cls) {
        jumpJackPhase='CLOSED'; score++; sounds.count(); els.hudScore.innerText=score;
        if(score>=targetCount) endGame(true);
    }
}

function processPose(kps,sx,sy) {
    if(!trafficLight.isRed||trafficLight.graceActive||!currentPose||poseSuccess||poseCooldown) return;
    const sc=kps.map(k=>({...k,x:k.x*sx,y:k.y*sy}));
    if(currentPose.check(sc)) {
        poseSuccess=true; score++; sounds.count(); els.hudScore.innerText=score;
        clearTimeout(poseTimerId); els.poseIcon.innerText='✅'; els.poseLabel.innerText='성공! +1점';
    }
}

// ═══════════════════════════════════════════════
// [17] 캘리브레이션
// ═══════════════════════════════════════════════
let calibrateReady=false;
function processCalibration(kps,sx,sy) {
    if(calibrateReady) return;
    const scaled=kps.map(k=>({...k,x:k.x*sx,y:k.y*sy}));
    if(checkClap(scaled)) {
        initialShoulderDist=getShoulderDist(kps,sx,sy);
        calibrateReady=true; sounds.clap(); els.centerMessage.classList.add('hidden'); startActualGame();
    }
}

// ═══════════════════════════════════════════════
// [18] 게임 시작 / 종료
// ═══════════════════════════════════════════════
function startActualGame() {
    state='PLAYING'; score=0; squatPhase='STANDING'; jumpJackPhase='CLOSED';
    poseSuccess=false; poseCooldown=false; currentPose=null; moveDetector.reset();
    els.hudScore.innerText=0; trafficLight.start();
    timeLeft=DIFFICULTY[selectedDiff].gameSec;
    els.hudTimer.innerText=timeLeft; els.hudTimer.classList.remove('danger');
    gameTimer=setInterval(()=>{
        timeLeft--; els.hudTimer.innerText=timeLeft;
        if(timeLeft<=10) els.hudTimer.classList.add('danger');
        if(timeLeft<=0){ clearInterval(gameTimer); endGame(false); }
    },1000);
    if(selectedMode==='pose') startPoseCycle();
    if(selectedMode==='approach'){ els.progressWrap.classList.remove('hidden'); canvas.addEventListener('pointerdown',onTouchCanvas); }
}

function onTouchCanvas() {
    if(state!=='PLAYING') return;
    if(!trafficLight.isGreen){ showCaught('🔴 초록불일 때 터치!'); return; }
    endGame(true);
}

function startPoseCycle() {
    let last=trafficLight.phase;
    const w=setInterval(()=>{
        if(state!=='PLAYING'){ clearInterval(w); return; }
        if(trafficLight.isRed&&last!=='red') {
            currentPose=POSES[Math.floor(Math.random()*POSES.length)];
            poseSuccess=false; poseCooldown=false;
            els.poseIcon.innerText=currentPose.icon; els.poseLabel.innerText=currentPose.label;
            els.poseTarget.classList.remove('hidden');
            clearTimeout(poseTimerId);
            poseTimerId=setTimeout(()=>{
                if(!poseSuccess){ score=Math.max(0,score-1); els.hudScore.innerText=score; els.poseIcon.innerText='❌'; els.poseLabel.innerText='실패! -1점'; sounds.caught(); }
                poseCooldown=true;
            },2000);
        }
        if(trafficLight.isGreen&&last!=='green') els.poseTarget.classList.add('hidden');
        last=trafficLight.phase;
    },100);
}

function endGame(isWin) {
    state='RESULT'; trafficLight.stop(); clearInterval(gameTimer);
    canvas.removeEventListener('pointerdown',onTouchCanvas);
    els.inGameUI.classList.add('hidden'); els.gameContainer.classList.remove('state-green','state-warn','state-red');
    isWin?sounds.win():sounds.fail();
    els.resultTitle.innerText=isWin?'🎉 성공!':'⏰ 시간 초과!';
    const nm={approach:'다가오기',squat:'스쿼트',pose:'포즈 따라하기',jump:'점핑잭'};
    let d=`<strong>모드:</strong> ${nm[selectedMode]}<br>`;
    if(selectedMode==='approach') d+=isWin?'화면 터치 성공! 🏅':'다음엔 더 빨리 다가와요!';
    else if(selectedMode==='squat') d+=`스쿼트: <span style="color:#20D68A;font-weight:900">${score}회</span> / ${targetCount}회`;
    else if(selectedMode==='pose')  d+=`포즈 성공: <span style="color:#20D68A;font-weight:900">${score}점</span>`;
    else d+=`점핑잭: <span style="color:#20D68A;font-weight:900">${score}회</span> / ${targetCount}회`;
    els.resultDetails.innerHTML=d; els.resultOverlay.classList.remove('hidden');
}

// ═══════════════════════════════════════════════
// [19] 메뉴 바인딩
// ═══════════════════════════════════════════════
document.querySelectorAll('.mode-btn').forEach(btn=>{
    btn.onclick=()=>{
        document.querySelectorAll('.mode-btn').forEach(b=>b.classList.remove('active'));
        btn.classList.add('active'); selectedMode=btn.dataset.mode;
        document.getElementById('target-count-group').style.display=(selectedMode==='approach')?'none':'';
    };
});
document.querySelectorAll('.toggle-btn[data-diff]').forEach(btn=>{
    btn.onclick=()=>{
        document.querySelectorAll('.toggle-btn[data-diff]').forEach(b=>b.classList.remove('active'));
        btn.classList.add('active'); selectedDiff=btn.dataset.diff;
    };
});
els.startBtn.onclick=()=>{ initAudio(); targetCount=parseInt(document.getElementById('target-count').value)||15; startCalibration(); };
els.restartBtn.onclick=()=>{ els.resultOverlay.classList.add('hidden'); startCalibration(); };
els.menuBtn.onclick=()=>{ els.resultOverlay.classList.add('hidden'); els.startMenu.classList.remove('hidden'); state='MENU'; };

function startCalibration() {
    state='CALIBRATION'; calibrateReady=false; initialShoulderDist=null;
    prevShoulderDist=null; runWarningFrames=0; moveDetector.reset();
    score=0; squatPhase='STANDING'; jumpJackPhase='CLOSED';
    els.startMenu.classList.add('hidden'); els.inGameUI.classList.remove('hidden');
    els.poseTarget.classList.add('hidden'); els.progressWrap.classList.add('hidden');
    els.centerWarning.classList.add('hidden'); els.centerMessage.classList.remove('hidden');
    els.centerMessage.innerHTML='멀리 떨어져 서서<br>박수를 쳐서 시작!';
    els.gameContainer.classList.remove('state-green','state-warn','state-red');
    els.lightBanner.innerHTML='👏 박수!'; els.lightBanner.className='light-green';
    const ml={approach:'🚶 다가오기',squat:'🏋️ 스쿼트',pose:'🙆 포즈',jump:'🦘 점핑잭'};
    els.hudModeLabel.innerText=ml[selectedMode]; els.hudScore.innerText='0';
    els.hudTimer.innerText=DIFFICULTY[selectedDiff].gameSec; els.hudTimer.classList.remove('danger');
}

// ═══════════════════════════════════════════════
// [20] 렌더 루프
// ═══════════════════════════════════════════════
function renderLoop() {
    if(video&&video.readyState>=2&&lastPoses.length>0) displayPoses=lerpPoses(displayPoses,lastPoses);
    ctx.clearRect(0,0,canvas.width,canvas.height);
    if(state==='CALIBRATION'||state==='PLAYING') {
        const cw=canvas.width,ch=canvas.height;
        const vw=video.videoWidth||320,vh=video.videoHeight||240;
        const sx=cw/vw,sy=ch/vh;
        const pose=displayPoses[0];
        if(pose) {
            const kps=pose.keypoints;
            const col=trafficLight.isRed?'#ef4444':trafficLight.isWarn?'#f59e0b':'#10b981';
            drawSkeleton(kps,sx,sy,col);
            moveDetector.update(kps);
            if(state==='CALIBRATION') processCalibration(kps,sx,sy);
            else if(selectedMode==='approach') processApproach(kps,sx,sy);
            else if(selectedMode==='squat')    processSquat(kps,sx,sy);
            else if(selectedMode==='jump')     processJumpJack(kps,sx,sy);
            else if(selectedMode==='pose')     processPose(kps,sx,sy);
        }
    }
    requestAnimationFrame(renderLoop);
}

// ═══════════════════════════════════════════════
// [21] 시작
// ═══════════════════════════════════════════════
window.onload=init;
