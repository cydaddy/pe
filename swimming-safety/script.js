'use strict';
const questions = window.POOL_QUESTIONS;
const $ = id => document.getElementById(id);
const escapeHTML = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const letters = ['①', '②', '③', '④'];
let round = [], position = 0, records = [], currentAnswer = null, activeBlank = 0, activeLeft = null, locked = false;
let practice = false;
const qNow = () => questions[round[position]];
const answeredCorrectly = (q, answer) => {
  if (q.type === 'multi') return Array.isArray(answer) && answer.length === q.answer.length && q.answer.every(v => answer.includes(v));
  if (Array.isArray(q.answer)) return Array.isArray(answer) && q.answer.every((v, i) => answer[i] === v);
  return answer === q.answer;
};
function isComplete(q) {
  if (q.type === 'multi') return currentAnswer.length === q.answer.length;
  if (Array.isArray(q.answer)) return currentAnswer.length === q.answer.length && currentAnswer.every(v => v !== null);
  return currentAnswer !== null;
}
function startRound(indices, isPractice = false) {
  round = indices;
  position = 0;
  records = [];
  practice = isPractice;
  $('welcome').hidden = true;
  $('result').hidden = true;
  $('quiz').hidden = false;
  showQuestion();
}
function initAnswer() {
  const q = qNow();
  currentAnswer = ['multi', 'order'].includes(q.type) ? [] : Array.isArray(q.answer) ? q.answer.map(() => null) : null;
  activeBlank = 0;
  activeLeft = null;
}
function showQuestion() {
  const q = qNow();
  locked = false;
  initAnswer();
  $('question-type').textContent = q.label;
  $('question-number').textContent = `MISSION ${String(round[position] + 1).padStart(2, '0')}`;
  $('question-scene').innerHTML = `<span>${q.icon}</span><span>${escapeHTML(q.scene)}</span>`;
  $('question-title').textContent = q.title;
  $('question-instruction').textContent = q.instruction;
  $('feedback').hidden = true;
  $('check').hidden = false;
  $('check').textContent = '답 확인하기';
  $('next').hidden = true;
  $('reset-answer').disabled = false;
  updateProgress();
  renderAnswers();
  $('question-title').focus({preventScroll:true});
  window.scrollTo({top:0, behavior:'instant'});
}
function updateProgress() {
  $('progress-label').textContent = `${practice ? '복습' : '미션'} ${position + 1} / ${round.length}`;
  $('score-label').textContent = `첫 답 정답 ${records.filter(r => r.correct).length}개`;
  $('progress-fill').style.width = `${records.length / round.length * 100}%`;
  const bar = document.querySelector('.progress-track');
  bar.setAttribute('aria-valuemax', round.length);
  bar.setAttribute('aria-valuenow', records.length);
}
function optionButton(text, i, selected, click, className = 'option') {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = className + (selected ? ' selected' : '');
  button.disabled = locked;
  button.setAttribute('aria-pressed', String(selected));
  button.innerHTML = `<span class="marker" aria-hidden="true">${selected ? '✓' : letters[i] || i + 1}</span><span>${escapeHTML(text)}</span>`;
  button.addEventListener('click', click);
  return button;
}
function keepFocus(render, selector) {
  render();
  const target = $('answers').querySelector(selector);
  if (target) target.focus({preventScroll:true});
}
function renderAnswers() {
  const q = qNow(), host = $('answers');
  host.replaceChildren();
  if (['choice', 'multi', 'ox'].includes(q.type)) {
    const options = document.createElement('div');
    options.className = 'options' + (q.type === 'ox' ? ' ox-options' : '');
    options.setAttribute('role', 'group');
    options.setAttribute('aria-label', q.instruction);
    q.options.forEach((text, i) => {
      const selected = q.type === 'multi' ? currentAnswer.includes(i) : currentAnswer === i;
      const b = optionButton(text, i, selected, () => {
        if (q.type === 'multi') {
          currentAnswer = currentAnswer.includes(i) ? currentAnswer.filter(v => v !== i) : [...currentAnswer, i];
        } else currentAnswer = i;
        keepFocus(renderAnswers, `[data-option="${i}"]`);
      });
      b.dataset.option = i;
      options.append(b);
    });
    host.append(options);
  } else if (q.type === 'word') {
    const sentence = document.createElement('div');
    sentence.className = 'word-sentence';
    q.parts.forEach((part, i) => {
      sentence.append(document.createTextNode(part));
      if (i >= q.answer.length) return;
      const b = document.createElement('button');
      b.className = 'blank' + (activeBlank === i ? ' active' : '');
      b.textContent = currentAnswer[i] === null ? '골라요' : q.options[currentAnswer[i]];
      b.setAttribute('aria-label', `${i + 1}번째 빈칸: ${b.textContent}`);
      b.setAttribute('aria-pressed', String(activeBlank === i));
      b.disabled = locked;
      b.dataset.blank = i;
      b.onclick = () => { activeBlank = i; keepFocus(renderAnswers, `[data-blank="${i}"]`); };
      sentence.append(b);
    });
    host.append(sentence);
    const label = document.createElement('p');
    label.className = 'bank-label';
    label.textContent = '단어 주머니 · 같은 단어를 여러 번 골라도 돼요';
    host.append(label);
    const bank = document.createElement('div');
    bank.className = 'word-bank';
    q.options.forEach((text, i) => {
      const b = document.createElement('button');
      b.textContent = text;
      b.disabled = locked;
      b.dataset.word = i;
      b.onclick = () => {
        currentAnswer[activeBlank] = i;
        const empty = currentAnswer.indexOf(null);
        if (empty >= 0) activeBlank = empty;
        keepFocus(renderAnswers, `[data-word="${i}"]`);
      };
      bank.append(b);
    });
    host.append(bank);
  } else if (q.type === 'match') {
    host.innerHTML = '<div class="match-caption"><span>표지판</span><span>알맞은 행동</span></div><div class="match-area"><svg class="match-lines" aria-hidden="true"></svg><div class="match-column left-column"></div><div class="match-column right-column"></div></div><p class="connection-list" aria-live="polite"></p>';
    q.left.forEach((text, i) => {
      const b = document.createElement('button');
      b.className = 'match-item' + (currentAnswer[i] !== null ? ' linked' : '') + (activeLeft === i ? ' active' : '');
      b.textContent = `${i + 1}. ${text}`;
      b.dataset.left = i;
      b.disabled = locked;
      b.setAttribute('aria-pressed', String(activeLeft === i));
      b.setAttribute('aria-label', `${text}${currentAnswer[i] === null ? ', 아직 연결하지 않음' : ', 연결: ' + q.right[currentAnswer[i]]}`);
      b.onclick = () => { activeLeft = i; keepFocus(renderAnswers, `[data-left="${i}"]`); };
      host.querySelector('.left-column').append(b);
    });
    q.right.forEach((text, i) => {
      const b = document.createElement('button');
      b.className = 'match-item' + (currentAnswer.includes(i) ? ' linked' : '');
      b.textContent = `${['ㄱ','ㄴ','ㄷ'][i]}. ${text}`;
      b.dataset.right = i;
      b.disabled = locked;
      b.onclick = () => {
        if (activeLeft === null) {
          host.querySelector('.connection-list').textContent = '왼쪽 표지판을 먼저 골라 주세요.';
          return;
        }
        currentAnswer = currentAnswer.map((v, j) => v === i && j !== activeLeft ? null : v);
        currentAnswer[activeLeft] = i;
        activeLeft = null;
        keepFocus(renderAnswers, `[data-right="${i}"]`);
      };
      host.querySelector('.right-column').append(b);
    });
    const links = currentAnswer.map((v, i) => v === null ? null : `${i + 1} ↔ ${['ㄱ','ㄴ','ㄷ'][v]}`).filter(Boolean);
    host.querySelector('.connection-list').textContent = links.length ? `연결한 선: ${links.join(' / ')}` : '왼쪽 표지판을 먼저 골라 주세요.';
    requestAnimationFrame(drawLines);
  } else if (q.type === 'order') {
    const slots = document.createElement('div');
    slots.className = 'order-slots';
    q.answer.forEach((_, i) => {
      const slot = document.createElement('div');
      slot.className = 'order-slot';
      slot.innerHTML = `<b>${i + 1}</b><span>${currentAnswer[i] === undefined ? '어떤 행동이 먼저일까요?' : escapeHTML(q.options[currentAnswer[i]])}</span>`;
      slots.append(slot);
    });
    host.append(slots);
    const options = document.createElement('div');
    options.className = 'options';
    q.options.forEach((text, i) => {
      const selected = currentAnswer.includes(i);
      const b = optionButton(text, i, selected, () => {
        currentAnswer = selected ? currentAnswer.filter(v => v !== i) : [...currentAnswer, i];
        keepFocus(renderAnswers, `[data-order="${i}"]`);
      });
      b.dataset.order = i;
      if (selected) b.querySelector('.marker').textContent = currentAnswer.indexOf(i) + 1;
      options.append(b);
    });
    host.append(options);
  } else if (q.type === 'sort') {
    q.items.forEach((text, row) => {
      const item = document.createElement('div');
      item.className = 'sort-row';
      item.innerHTML = `<p id="sort-label-${row}">${row + 1}. ${escapeHTML(text)}</p><div class="sort-options" role="group" aria-labelledby="sort-label-${row}"></div>`;
      q.options.forEach((label, i) => {
        const b = document.createElement('button');
        b.textContent = label;
        b.className = currentAnswer[row] === i ? 'selected' : '';
        b.setAttribute('aria-pressed', String(currentAnswer[row] === i));
        b.dataset.sort = `${row}-${i}`;
        b.disabled = locked;
        b.onclick = () => { currentAnswer[row] = i; keepFocus(renderAnswers, `[data-sort="${row}-${i}"]`); };
        item.querySelector('.sort-options').append(b);
      });
      host.append(item);
    });
  }
}
function drawLines() {
  const area = $('answers').querySelector('.match-area');
  if (!area) return;
  const svg = area.querySelector('svg'), rect = area.getBoundingClientRect();
  svg.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
  svg.replaceChildren();
  const colors = ['#087e8b', '#b46a17', '#8252b0'];
  currentAnswer.forEach((v, i) => {
    if (v === null) return;
    const left = area.querySelector(`[data-left="${i}"]`).getBoundingClientRect();
    const right = area.querySelector(`[data-right="${v}"]`).getBoundingClientRect();
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', left.right - rect.left + 4);
    line.setAttribute('y1', left.top - rect.top + left.height / 2);
    line.setAttribute('x2', right.left - rect.left - 4);
    line.setAttribute('y2', right.top - rect.top + right.height / 2);
    line.setAttribute('stroke', colors[i]);
    line.setAttribute('stroke-width', '3');
    line.setAttribute('stroke-linecap', 'round');
    svg.append(line);
  });
}
new ResizeObserver(drawLines).observe($('answers'));
function showFeedback(kind, title, message) {
  $('feedback').className = `feedback ${kind}`;
  $('feedback').innerHTML = `<strong>${escapeHTML(title)}</strong><p>${escapeHTML(message)}</p>`;
  $('feedback').hidden = false;
}
$('check').onclick = () => {
  const q = qNow();
  if (!isComplete(q)) {
    showFeedback('notice', '조금만 더 골라 볼까?', q.type === 'multi' ? `안전한 행동을 정확히 ${q.answer.length}개 골라 주세요.` : '빈칸이나 아직 고르지 않은 항목을 모두 채워 주세요.');
    return;
  }
  const correct = answeredCorrectly(q, currentAnswer);
  if (!records[position]) records[position] = {index:round[position], correct};
  updateProgress();
  locked = correct;
  showFeedback(correct ? '' : 'wrong', correct ? '잘했어! 안전 약속을 찾았네 ✓' : '괜찮아! 안전 약속을 함께 배워 보자.', q.explanation);
  $('check').hidden = correct;
  $('check').textContent = '다시 확인하기';
  $('next').hidden = false;
  $('next').textContent = position === round.length - 1 ? '탐험 결과 보기 →' : '다음 미션 →';
  $('reset-answer').disabled = correct;
  renderAnswers();
  (correct ? $('next') : $('check')).focus({preventScroll:true});
};
$('reset-answer').onclick = () => {
  if (locked) return;
  initAnswer();
  $('feedback').hidden = true;
  renderAnswers();
};
$('next').onclick = () => {
  if (!records[position]) return;
  if (position < round.length - 1) { position++; showQuestion(); }
  else showResult();
};
function answerText(q) {
  if (q.type === 'match') return q.left.map((left, i) => `${left} → ${q.right[q.answer[i]]}`).join(' / ');
  if (q.type === 'sort') return q.items.map((_, i) => `${i + 1}번 ${q.options[q.answer[i]]}`).join(' / ');
  if (Array.isArray(q.answer)) return q.answer.map(v => q.options[v]).join(q.type === 'order' ? ' → ' : ', ');
  return q.options[q.answer];
}
function showResult() {
  $('quiz').hidden = true;
  $('result').hidden = false;
  const score = records.filter(r => r.correct).length;
  const missed = records.filter(r => !r.correct).map(r => r.index);
  $('result').innerHTML = `<div class="result-hero"><div class="result-icon" aria-hidden="true">🏅</div><span class="eyebrow">${practice ? '복습 완료' : '10가지 안전 미션 완료'}</span><h1 id="result-title" tabindex="-1">${score === round.length ? '멋진 안전 탐험대!' : '안전 약속, 한 걸음 더!'}</h1><div class="result-score">첫 번째 답에서 <strong>${score}</strong> / ${round.length}문제 정답</div><p>${missed.length ? '틀린 문제는 다시 만나면 더 잘 기억할 수 있어요.' : '배운 약속을 실제 수영장에서도 꼭 지켜요!'}</p><div class="result-actions">${missed.length ? '<button class="primary" id="retry-missed">틀린 문제 다시 풀기</button>' : ''}<button class="secondary" id="restart">10문제 다시 풀기</button><button class="text-button" id="home">처음 화면</button></div></div><div class="review-list"><h2>오늘 배운 안전 약속</h2>${records.map(r => `<details class="review-item"><summary><span class="review-badge ${r.correct ? '' : 'retry'}">${r.correct ? '첫 답 정답' : '다시 기억'}</span><span>${r.index + 1}. ${escapeHTML(questions[r.index].rule)}</span></summary><p><strong>정답:</strong> ${escapeHTML(answerText(questions[r.index]))}</p><p>${escapeHTML(questions[r.index].explanation)}</p></details>`).join('')}</div>`;
  $('restart').onclick = () => startRound(questions.map((_, i) => i));
  if ($('retry-missed')) $('retry-missed').onclick = () => startRound(missed, true);
  $('home').onclick = () => {
    $('result').hidden = true;
    $('welcome').hidden = false;
    $('start').focus({preventScroll:true});
    window.scrollTo({top:0, behavior:'instant'});
  };
  $('result-title').focus({preventScroll:true});
  window.scrollTo({top:0, behavior:'instant'});
}
$('start').onclick = () => startRound(questions.map((_, i) => i));
$('teacher-open').onclick = () => {
  $('teacher-content').innerHTML = questions.map((q, i) => `<article class="teacher-question"><h3>${i + 1}. ${escapeHTML(q.title)} <small>(${q.label})</small></h3><p><strong>정답:</strong> ${escapeHTML(answerText(q))}</p><p>${escapeHTML(q.explanation)}</p></article>`).join('');
  $('teacher-dialog').showModal();
};
$('teacher-close').onclick = () => $('teacher-dialog').close();
function printMaterials(key) {
  $('print-sheet').innerHTML = `<div class="${key ? 'key-mode' : ''}"><h1>첨벙! 수영장 안전 탐험대</h1><div class="print-name"><span>${key ? '선생님용 정답·해설' : '학년: ____ 반: ____ 이름: __________'}</span><span>총 10문제</span></div><p class="print-intro">${key ? '첫 답 기준으로 문제당 1점. 여러 항목이 있는 문항은 모두 맞으면 1점입니다. 고학년은 답을 고른 이유를 함께 이야기해 보세요.' : '안전한 물놀이를 위한 10가지 미션! 안내에 따라 고르거나 선을 연결해요.'}</p>${questions.map((q, i) => {
    let body = '';
    if (q.type === 'word') body = `<p>${q.parts.map((p, j) => escapeHTML(p) + (j < q.answer.length ? ' ( __________ ) ' : '')).join('').replace(/\n/g, '<br>')}</p><div class="print-options">보기: ${q.options.map(escapeHTML).join(' / ')} (같은 단어를 여러 번 써도 돼요)</div>`;
    else if (q.type === 'match') body = `<div class="print-match"><div>${q.left.map((v,j) => `${j + 1}. ${escapeHTML(v)} ○`).join('<br>')}</div><div>${q.right.map((v,j) => `○ ${['ㄱ','ㄴ','ㄷ'][j]}. ${escapeHTML(v)}`).join('<br>')}</div></div>`;
    else if (q.type === 'sort') body = `<div class="print-sort">${q.items.map(v => `${escapeHTML(v)}　[ 안전 / 위험 ]`).join('<br>')}</div>`;
    else body = `<div class="print-options">${q.options.map((v,j) => `${letters[j]} ${escapeHTML(v)}`).join('<br>')}</div>${q.type === 'order' ? '<p>순서: (　　) → (　　) → (　　)</p>' : ''}`;
    return `<article class="print-question"><h2>${i + 1}. ${escapeHTML(q.title)} [${q.label}]</h2>${key ? `<div class="print-answer"><strong>정답:</strong> ${escapeHTML(answerText(q))}<br>${escapeHTML(q.explanation)}</div>` : `<p>${q.type === 'word' ? '보기에서 알맞은 단어를 골라 빈칸에 넣어요.' : q.type === 'match' ? '표지판과 알맞은 행동을 선으로 연결해요.' : q.type === 'order' ? '먼저 할 행동부터 보기 번호를 써요.' : escapeHTML(q.instruction)}</p>${body}`}</article>`;
  }).join('')}<p class="print-sources">참고: 미국 적십자 Water Safety for Kids / Swimming Safety, Pool Safely Safety Tips. 수영장에서는 선생님과 안전요원의 안내를 따라요.</p></div>`;
  $('teacher-dialog').close();
  window.print();
}
$('print-student').onclick = () => printMaterials(false);
$('print-key').onclick = () => printMaterials(true);
