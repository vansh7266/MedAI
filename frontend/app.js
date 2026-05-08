const API_BASE_URL = window.MEDAI_API_BASE_URL || 'http://localhost:8000';

const state = {
  currentSessionId: localStorage.getItem('medai-session-id') || generateUUID(),
  isAnalyzing: false,
  selectedPdf: null,
  activeTab: 'text'
};

localStorage.setItem('medai-session-id', state.currentSessionId);

const elements = {};

function cacheElements() {
  elements.particles = document.getElementById('particles');
  elements.connectionStatus = document.getElementById('connection-status');
  elements.statusText = elements.connectionStatus.querySelector('.status-text');
  elements.tabs = Array.from(document.querySelectorAll('.tab'));
  elements.tabContents = Array.from(document.querySelectorAll('.tab-content'));
  elements.reportText = document.getElementById('report-text');
  elements.charCount = document.getElementById('char-count');
  elements.dropZone = document.getElementById('drop-zone');
  elements.pdfInput = document.getElementById('pdf-input');
  elements.fileInfo = document.getElementById('file-info');
  elements.analyzeBtn = document.getElementById('analyze-btn');
  elements.btnText = elements.analyzeBtn.querySelector('.btn-text');
  elements.btnLoader = elements.analyzeBtn.querySelector('.btn-loader');
  elements.resultsPanel = document.getElementById('results-panel');
  elements.highlightedText = document.getElementById('highlighted-text');
  elements.entityLegend = document.getElementById('entity-legend');
  elements.gaugeArc = document.getElementById('gauge-arc');
  elements.gaugeNeedle = document.getElementById('gauge-needle');
  elements.riskLabel = document.getElementById('risk-label');
  elements.riskProbs = document.getElementById('risk-probs');
  elements.explanationText = document.getElementById('explanation-text');
  elements.chatToggle = document.getElementById('chat-toggle');
  elements.chatWindow = document.getElementById('chat-window');
  elements.chatClose = document.getElementById('chat-close');
  elements.chatMessages = document.getElementById('chat-messages');
  elements.chatInput = document.getElementById('chat-input');
  elements.chatSend = document.getElementById('chat-send');
}

function createParticles() {
  const symbols = ['💊', '🧬', '❤️', '⚕️', ''];
  elements.particles.innerHTML = '';
  for (let index = 0; index < 20; index += 1) {
    const particle = document.createElement('div');
    const useShape = index % 5 === 4;
    particle.className = useShape ? 'particle shape' : 'particle';
    particle.textContent = useShape ? '' : symbols[index % symbols.length];
    const size = Math.floor(Math.random() * 21) + 20;
    particle.style.left = `${Math.random() * 100}%`;
    particle.style.top = `${Math.random() * 100}%`;
    particle.style.width = `${size}px`;
    particle.style.height = `${Math.max(14, size / 2)}px`;
    particle.style.fontSize = `${size}px`;
    particle.style.animationDuration = `${15 + Math.random() * 10}s`;
    particle.style.animationDelay = `${Math.random() * -20}s`;
    particle.style.opacity = `${0.03 + Math.random() * 0.05}`;
    elements.particles.appendChild(particle);
  }
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    const data = await response.json();
    elements.connectionStatus.classList.remove('disconnected');
    elements.connectionStatus.classList.add('connected');
    elements.statusText.textContent = data.model_loaded ? 'Backend connected' : 'Backend warming up';
  } catch (error) {
    elements.connectionStatus.classList.remove('connected');
    elements.connectionStatus.classList.add('disconnected');
    elements.statusText.textContent = 'Backend offline';
  }
}

function switchTab(tabName) {
  state.activeTab = tabName;
  elements.tabs.forEach((tab) => {
    const isActive = tab.dataset.tab === tabName;
    tab.classList.toggle('active', isActive);
    tab.setAttribute('aria-selected', String(isActive));
  });
  elements.tabContents.forEach((content) => {
    content.classList.toggle('active', content.id === `${tabName}-tab`);
  });
}

function updateCharacterCount() {
  const length = elements.reportText.value.length;
  elements.charCount.textContent = `${length} / 5000`;
  elements.charCount.style.color = length >= 4900 ? '#EF4444' : '#64748B';
}

function validatePdf(file) {
  if (!file) {
    return 'No file selected.';
  }
  if (file.type !== 'application/pdf') {
    return 'Please choose a PDF file.';
  }
  if (file.size > 10 * 1024 * 1024) {
    return 'PDF must be smaller than 10MB.';
  }
  return '';
}

function selectPdf(file) {
  const error = validatePdf(file);
  if (error) {
    state.selectedPdf = null;
    elements.fileInfo.textContent = '';
    showToast(error);
    return;
  }
  state.selectedPdf = file;
  elements.fileInfo.textContent = `${file.name} · ${formatFileSize(file.size)}`;
  showToast('PDF ready for analysis.', 'success');
}

function setLoading(isLoading) {
  state.isAnalyzing = isLoading;
  elements.analyzeBtn.disabled = isLoading;
  elements.btnText.classList.toggle('hidden', isLoading);
  elements.btnLoader.classList.toggle('hidden', !isLoading);
}

async function handleAnalyze() {
  if (state.isAnalyzing) {
    return;
  }

  const text = elements.reportText.value.trim();
  const usePdf = state.activeTab === 'pdf';
  if (usePdf && !state.selectedPdf) {
    shakeInput();
    showToast('Please upload a PDF first.');
    return;
  }
  if (!usePdf && !text) {
    shakeInput();
    showToast('Please paste report text first.');
    return;
  }

  setLoading(true);
  try {
    const result = usePdf ? await analyzePDF(state.selectedPdf) : await analyzeText(text);
    renderEntities(result.entities || [], usePdf ? '' : text);
    renderRisk(result.risk_level, result.risk_probs || {});
    renderExplanation(result.explanation || 'Analysis complete.');
    showResults();
  } catch (error) {
    shakeInput();
    showToast(error.message || 'Analysis failed.');
  } finally {
    setLoading(false);
  }
}

async function analyzeText(text) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    return await parseApiResponse(response);
  } catch (error) {
    throw new Error(error.message || 'Unable to reach the analysis API.');
  }
}

async function analyzePDF(file) {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/analyze-pdf`, {
      method: 'POST',
      body: formData
    });
    return await parseApiResponse(response);
  } catch (error) {
    throw new Error(error.message || 'Unable to upload the PDF.');
  }
}

async function parseApiResponse(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }
  if (!response.ok) {
    const detail = typeof payload.detail === 'string' ? payload.detail : 'Server error while processing the request.';
    throw new Error(detail);
  }
  return payload;
}

function renderEntities(entities, originalText) {
  elements.highlightedText.innerHTML = '';
  elements.entityLegend.innerHTML = '';
  const uniqueTypes = [...new Set(entities.map((entity) => entity.type).filter(Boolean))];

  uniqueTypes.forEach((type) => {
    const item = document.createElement('div');
    item.className = 'legend-item';
    const dot = document.createElement('span');
    dot.className = `legend-dot legend-${type}`;
    const label = document.createElement('span');
    label.textContent = type;
    item.append(dot, label);
    elements.entityLegend.appendChild(item);
  });

  if (!entities.length) {
    elements.highlightedText.className = 'highlighted-text empty-state';
    elements.highlightedText.textContent = 'No named medical entities were detected in this report.';
    return;
  }

  elements.highlightedText.className = 'highlighted-text';
  if (!originalText) {
    const list = document.createElement('div');
    entities.forEach((entity) => {
      const span = document.createElement('span');
      span.className = `entity entity-${entity.type}`;
      span.textContent = `${entity.text} (${entity.type})`;
      list.appendChild(span);
      list.appendChild(document.createTextNode(' '));
    });
    elements.highlightedText.appendChild(list);
    return;
  }

  // Build non-overlapping matches so highlights preserve the original report text exactly.
  const matches = findEntityMatches(entities, originalText);
  let cursor = 0;
  matches.forEach((match) => {
    if (match.start > cursor) {
      elements.highlightedText.appendChild(document.createTextNode(originalText.slice(cursor, match.start)));
    }
    const span = document.createElement('span');
    span.className = `entity entity-${match.type}`;
    span.title = `${match.type} · ${Math.round((match.confidence || 0) * 100)}% confidence`;
    span.textContent = originalText.slice(match.start, match.end);
    elements.highlightedText.appendChild(span);
    cursor = match.end;
  });
  if (cursor < originalText.length) {
    elements.highlightedText.appendChild(document.createTextNode(originalText.slice(cursor)));
  }
}

function findEntityMatches(entities, text) {
  const lowerText = text.toLowerCase();
  const rawMatches = [];
  entities.forEach((entity) => {
    const needle = String(entity.text || '').trim().toLowerCase();
    if (!needle) {
      return;
    }
    let start = lowerText.indexOf(needle);
    while (start !== -1) {
      rawMatches.push({
        start,
        end: start + needle.length,
        type: entity.type,
        confidence: entity.confidence
      });
      start = lowerText.indexOf(needle, start + needle.length);
    }
  });

  rawMatches.sort((first, second) => first.start - second.start || second.end - first.end);
  const filtered = [];
  rawMatches.forEach((match) => {
    const overlaps = filtered.some((existing) => match.start < existing.end && match.end > existing.start);
    if (!overlaps) {
      filtered.push(match);
    }
  });
  return filtered;
}

function renderRisk(riskLevel, riskProbs) {
  const level = String(riskLevel || 'LOW').toUpperCase();
  const colors = { LOW: '#10B981', MEDIUM: '#F59E0B', HIGH: '#EF4444' };
  const rotations = { LOW: -60, MEDIUM: 0, HIGH: 60 };
  const offsets = { LOW: 165, MEDIUM: 85, HIGH: 0 };

  elements.riskLabel.className = `risk-label ${level.toLowerCase()}`;
  elements.riskLabel.textContent = level;
  elements.gaugeNeedle.setAttribute('transform', `rotate(${rotations[level] ?? -60} 100 100)`);
  elements.gaugeArc.style.stroke = colors[level] || colors.LOW;
  elements.gaugeArc.style.strokeDashoffset = String(offsets[level] ?? offsets.LOW);

  elements.riskProbs.innerHTML = '';
  ['LOW', 'MEDIUM', 'HIGH'].forEach((label) => {
    const probability = Number(riskProbs[label] || 0);
    const percentage = Math.max(0, Math.min(100, probability * 100));
    const item = document.createElement('div');
    item.className = 'prob-item';
    item.innerHTML = `
      <div class="prob-label"><span>${label}</span><span>${percentage.toFixed(0)}%</span></div>
      <div class="prob-bar"><div class="prob-fill" style="background:${colors[label]}"></div></div>
    `;
    elements.riskProbs.appendChild(item);
    requestAnimationFrame(() => {
      item.querySelector('.prob-fill').style.width = `${percentage}%`;
    });
  });
}

function renderExplanation(explanation) {
  elements.explanationText.textContent = explanation;
  elements.explanationText.classList.remove('fade-in');
  void elements.explanationText.offsetWidth;
  elements.explanationText.classList.add('fade-in');
}

function showResults() {
  elements.resultsPanel.classList.remove('hidden');
  Array.from(elements.resultsPanel.querySelectorAll('.result-card')).forEach((card, index) => {
    card.classList.remove('fade-in');
    card.style.animationDelay = `${index * 0.1}s`;
    void card.offsetWidth;
    card.classList.add('fade-in');
  });
  elements.resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function toggleChat() {
  const opening = elements.chatWindow.classList.contains('hidden');
  elements.chatWindow.classList.toggle('hidden');
  if (opening && elements.chatMessages.children.length === 0) {
    addMessage('Hi, I can help explain your report after analysis. Ask me about a test, value, or risk result.', false);
  }
}

async function sendMessage() {
  const message = elements.chatInput.value.trim();
  if (!message) {
    return;
  }
  addMessage(message, true);
  elements.chatInput.value = '';
  const typingNode = showTypingIndicator();

  try {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, session_id: state.currentSessionId })
    });
    const result = await parseApiResponse(response);
    typingNode.remove();
    state.currentSessionId = result.session_id || state.currentSessionId;
    localStorage.setItem('medai-session-id', state.currentSessionId);
    addMessage(result.response || 'I could not generate a response.', false);
  } catch (error) {
    typingNode.remove();
    addMessage(error.message || 'Chat is unavailable right now.', false);
  }
}

function addMessage(text, isUser) {
  const message = document.createElement('div');
  message.className = `message ${isUser ? 'user' : 'agent'}`;
  if (isUser) {
    message.textContent = text;
  } else {
    message.innerHTML = parseMarkdownLite(text);
  }
  elements.chatMessages.appendChild(message);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

function showTypingIndicator() {
  const message = document.createElement('div');
  message.className = 'message agent';
  message.innerHTML = '<span class="typing-indicator"><span></span><span></span><span></span></span>';
  elements.chatMessages.appendChild(message);
  elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  return message;
}

function parseMarkdownLite(text) {
  return escapeHTML(text)
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/^[-*]\s+(.*)$/gm, '• $1')
    .replace(/\n/g, '<br>');
}

function escapeHTML(text) {
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (character) => {
    const random = Math.random() * 16 | 0;
    const value = character === 'x' ? random : (random & 0x3 | 0x8);
    return value.toString(16);
  });
}

function showToast(message, type = 'error') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  window.setTimeout(() => {
    toast.remove();
  }, 3000);
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function shakeInput() {
  const target = state.activeTab === 'pdf' ? elements.dropZone : elements.reportText;
  target.classList.remove('shake');
  void target.offsetWidth;
  target.classList.add('shake');
}

function setupEventListeners() {
  elements.tabs.forEach((tab) => {
    tab.addEventListener('click', () => switchTab(tab.dataset.tab));
  });

  elements.reportText.addEventListener('input', updateCharacterCount);
  elements.dropZone.addEventListener('click', () => elements.pdfInput.click());
  elements.dropZone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      elements.pdfInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.add('drag-active');
    });
  });

  ['dragleave', 'drop'].forEach((eventName) => {
    elements.dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      elements.dropZone.classList.remove('drag-active');
    });
  });

  elements.dropZone.addEventListener('drop', (event) => {
    const file = event.dataTransfer.files[0];
    selectPdf(file);
  });

  elements.pdfInput.addEventListener('change', (event) => {
    selectPdf(event.target.files[0]);
  });

  elements.analyzeBtn.addEventListener('click', handleAnalyze);
  elements.chatToggle.addEventListener('click', toggleChat);
  elements.chatClose.addEventListener('click', toggleChat);
  elements.chatSend.addEventListener('click', sendMessage);
  elements.chatInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      sendMessage();
    }
  });

  window.addEventListener('resize', () => {
    if (window.innerWidth > 768) {
      elements.chatWindow.style.maxHeight = '';
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  cacheElements();
  createParticles();
  setupEventListeners();
  updateCharacterCount();
  checkHealth();
  window.setInterval(checkHealth, 10000);
});
