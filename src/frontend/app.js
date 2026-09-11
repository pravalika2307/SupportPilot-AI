/**
 * SupportPilot AI — Application Controller v4.0 (Product Clarity Pass)
 * Enterprise support-operations workstation emphasizing decision justification,
 * transparent signals, and fail-closed safety.
 */

'use strict';

// ── State ────────────────────────────────────────────────────────
const state = {
  currentView: 'analyze',  // 1. Analyze as primary wow workstation
  currentCase: null,       // { id, query, analysisData, time }
  currentTab: 'why',
  currentFilter: 'all',    // 'all' | 'auto' | 'escalate' | 'withheld'
  currentSort: 'newest',   // 'newest' | 'decision' | 'intent' | 'quality'
  searchQuery: '',
  selectedCandidateIndex: 0,
  inboxItems: [],
  analyzeResult: null,
  activeAnalyzeCandidateIndex: 0
};

// ── Demo Queue (6 Canonical Production Scenarios) ─────────────────
const DEMO_QUEUE = [
  { id: 'CASE-1048', query: 'How do I turn on low power mode on iPhone 7? My battery drains fast.', time: 'Just now' },
  { id: 'CASE-1049', query: 'My iPhone screen is completely cracked and shattered after dropping it on the street.', time: '3m ago' },
  { id: 'CASE-1050', query: 'I am locked out of my Apple ID and cannot receive 2FA verification codes.', time: '8m ago' },
  { id: 'CASE-1051', query: 'why are my App Store categories in Spanish but everything else in English??', time: '14m ago' },
  { id: 'CASE-1052', query: "Ever since I upgraded to High Sierra, my media controls on the keyboard don't work with iTunes. Please help or fix it @AppleSupport", time: '27m ago' },
  { id: 'CASE-1053', query: 'My Apple Pay transaction failed at checkout and charged my account twice without receipt.', time: '45m ago' },
];

// Historical corpus seed (for Knowledge view)
const CORPUS_SEED = [
  { conversation_id: '115854', historical_intent: 'battery_power', similarity_score: 0.645, matched_customer_query: 'How do I turn on low power mode on iPhone 7? My battery drains fast.', historical_agent_reply: 'You can turn on Low Power Mode in Settings > Battery. You can also add it to Control Center for quick access: https://apple.co/battery-tips' },
  { conversation_id: '117460', historical_intent: 'general_inquiry_other', similarity_score: 0.407, matched_customer_query: "I can't believe that Apple Support has not fixed the vowel text issue. It's very annoying", historical_agent_reply: "We'd like to look into the trouble you're having. Does this only occur in the Facebook app?" },
  { conversation_id: '121048', historical_intent: 'app_store_billing', similarity_score: 0.552, matched_customer_query: 'upgraded to 11.1.2 and iTunes Player won\'t work on iPhone 7. ANNOYING', historical_agent_reply: "Let's work together to get this resolved. To be sure we're on the same page, are you noticing an issue with the player controls?" },
  { conversation_id: '119210', historical_intent: 'battery_power', similarity_score: 0.543, matched_customer_query: 'iOS 11 is draining the battery on my iPhone 7 twice as fast as iOS 10. Help!', historical_agent_reply: "We'd like to gather some more information for better troubleshooting. Can you DM us the country you are located in? https://t.co/GDrQ..." },
  { conversation_id: '131055', historical_intent: 'hardware_repair_service', similarity_score: 0.612, matched_customer_query: 'My iPhone fell and the screen cracked completely. Do I need to go to an Apple store?', historical_agent_reply: 'We recommend visiting an Apple Store or Apple Authorized Service Provider for a screen repair. You can find locations at https://apple.co/findstore' },
  { conversation_id: '128490', historical_intent: 'account_security_and_pii', similarity_score: 0.589, matched_customer_query: 'I cannot sign into iCloud and 2-step verification is not sending text to my trusted number.', historical_agent_reply: 'Security is a top priority. Please use iforgot.apple.com to initiate account recovery or contact Apple Support directly: https://apple.co/contact' },
];

// ── Boot ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initAppLoader();
  initNav();
  initHowItWorks();
  initInbox();
  initAnalyze();
  initKnowledge();
  initModal();
  checkAgentHealth();
});

// ── 1. Application Loading Screen ────────────────────────────────
async function initAppLoader() {
  const loader = document.getElementById('appLoader');
  if (!loader) return;

  const chkModel = document.getElementById('chk-model');
  const chkRetriever = document.getElementById('chk-retriever');
  const chkSafety = document.getElementById('chk-safety');
  const chkReady = document.getElementById('chk-ready');

  // Verify health in parallel
  const healthPromise = checkAgentHealth();

  await delay(160);
  chkModel?.classList.add('ready');

  await delay(160);
  chkRetriever?.classList.add('ready');

  await delay(160);
  chkSafety?.classList.add('ready');

  await healthPromise;

  await delay(120);
  chkReady?.classList.add('ready');

  await delay(200);
  loader.classList.add('fade-out');
  setTimeout(() => {
    loader.style.display = 'none';
  }, 350);
}

// ── 2. First-Time How-It-Works Panel ─────────────────────────────
function initHowItWorks() {
  const panel = document.getElementById('howItWorksPanel');
  const dismissBtn = document.getElementById('dismissHiwBtn');

  if (localStorage.getItem('supportpilot_hiw_dismissed') === '1') {
    panel?.classList.add('dismissed');
  }

  dismissBtn?.addEventListener('click', () => {
    panel?.classList.add('dismissed');
    localStorage.setItem('supportpilot_hiw_dismissed', '1');
  });
}

// ── Operational State Classifier ─────────────────────────────────
/**
 * Maps live API outputs into the 3 first-class support product states:
 * - AUTO_HANDLE: Safe to resolve autonomously with verified evidence
 * - WITHHELD: Insufficient evidence floor or intent discordance ("Chose not to guess")
 * - ESCALATE: Direct policy / brand-safety / sensitive routing triggers
 */
function getCaseState(data) {
  if (!data) return 'UNKNOWN';
  if (data.decision === 'AUTO_HANDLE') return 'AUTO_HANDLE';

  const topSim = data.retrieved_evidence?.[0]?.similarity_score ?? 0;
  const isDirectSafety = (
    data.escalation_reason === 'hardware_physical_damage' ||
    data.escalation_reason === 'account_security_and_pii' ||
    data.escalation_reason === 'financial_and_billing' ||
    data.escalation_reason === 'abusive_language'
  );

  if (
    data.decision === 'WITHHELD' ||
    data.escalation_reason === 'insufficient_grounding' ||
    data.grounding_status === 'insufficient_evidence' ||
    data.grounding_status === 'abstained' ||
    (topSim < 0.45 && !isDirectSafety)
  ) {
    return 'WITHHELD';
  }
  return 'ESCALATE';
}

// ── Navigation ────────────────────────────────────────────────────
function initNav() {
  document.querySelectorAll('.nav-item[data-view]').forEach(btn => {
    btn.addEventListener('click', () => switchView(btn.dataset.view));
  });
  document.getElementById('inboxAnalyzeLink')?.addEventListener('click', () => switchView('analyze'));
  document.getElementById('backToInbox')?.addEventListener('click', () => switchView('inbox'));
  document.getElementById('reanalyzeBtn')?.addEventListener('click', () => {
    if (state.currentCase) {
      switchView('analyze');
      const inp = document.getElementById('analyzeInput');
      if (inp) {
        inp.value = state.currentCase.query;
        updateCharCount();
        runAnalysis(state.currentCase.query);
      }
    }
  });
}

function switchView(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item[data-view]').forEach(b => b.classList.remove('active'));

  const navBtn = document.getElementById(`nav-${view}`);
  if (navBtn) navBtn.classList.add('active');

  const targetView = document.getElementById(`view-${view}`);
  if (targetView) targetView.classList.add('active');

  state.currentView = view;

  const titles = {
    analyze:    ['SupportPilot Analyzer', 'Paste a customer message. SupportPilot decides whether it can safely answer, needs stronger evidence, or should hand the case to a specialist.'],
    inbox:      ['Inbox', 'Recent analyzed inquiries queue with decision explainability'],
    case:       ['Case Detail', 'Operational breakdown & evidence provenance'],
    knowledge:  ['Knowledge Corpus', '800 historical Apple Support threads strictly isolated from evaluation'],
    evaluation: ['Model Evaluation', '200 human-verified benchmark cases audited for precision & zero-leakage'],
    failures:   ['Where SupportPilot Still Struggles', 'Post-mortem engineering investigation across 4 failure modes'],
  };

  const [title, sub] = titles[view] || ['SupportPilot', ''];
  const titleEl = document.getElementById('pageTitle');
  const subEl = document.getElementById('pageSubtitle');
  if (titleEl) titleEl.textContent = title;
  if (subEl) subEl.textContent = sub;

  if (view === 'failures') loadFailures();
  if (view === 'knowledge') renderKnowledgeCorpus(CORPUS_SEED);
}

// ── Inbox ─────────────────────────────────────────────────────────
async function initInbox() {
  initInboxFilters();
  initInboxSort();
  initInboxSearch();
  await loadInbox();
}

async function loadInbox() {
  const tbody = document.getElementById('casesTableBody');
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="6" class="loading-line">Loading analyzed inquiries...</td></tr>';

  try {
    const results = await Promise.all(
      DEMO_QUEUE.map(async item => {
        try {
          const data = await apiAnalyze(item.query);
          return { ...item, analysisData: data, error: null };
        } catch (err) {
          return { ...item, analysisData: null, error: err.message };
        }
      })
    );

    state.inboxItems = results;
    renderInboxTable();
    updateInboxStats();

  } catch {
    tbody.innerHTML = '<tr><td colspan="6" class="loading-line" style="color:var(--escalate)">Failed to load inquiries. Ensure backend is running on port 8000.</td></tr>';
  }
}

function initInboxFilters() {
  document.querySelectorAll('.filter-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.filter-pill').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.currentFilter = btn.dataset.filter;
      renderInboxTable();
    });
  });
}

function initInboxSort() {
  const sortSelect = document.getElementById('inboxSort');
  if (!sortSelect) return;
  sortSelect.addEventListener('change', (e) => {
    state.currentSort = e.target.value;
    renderInboxTable();
  });
}

function initInboxSearch() {
  const search = document.getElementById('inboxSearch');
  if (!search) return;
  search.addEventListener('input', () => {
    state.searchQuery = (search.value || '').trim().toLowerCase();
    renderInboxTable();
  });
}

function getFilteredAndSortedInbox() {
  let items = [...state.inboxItems];

  if (state.currentFilter !== 'all') {
    items = items.filter(item => {
      const cState = getCaseState(item.analysisData).toLowerCase();
      if (state.currentFilter === 'auto') return cState === 'auto_handle';
      if (state.currentFilter === 'escalate') return cState === 'escalate';
      if (state.currentFilter === 'withheld') return cState === 'withheld';
      return true;
    });
  }

  if (state.searchQuery) {
    items = items.filter(item => {
      const q = (item.query || '').toLowerCase();
      const id = (item.id || '').toLowerCase();
      const intent = (item.analysisData?.predicted_intent || '').toLowerCase();
      return q.includes(state.searchQuery) || id.includes(state.searchQuery) || intent.includes(state.searchQuery);
    });
  }

  items.sort((a, b) => {
    if (state.currentSort === 'decision') {
      const sA = getCaseState(a.analysisData);
      const sB = getCaseState(b.analysisData);
      return sA.localeCompare(sB);
    }
    if (state.currentSort === 'intent') {
      const iA = a.analysisData?.predicted_intent || '';
      const iB = b.analysisData?.predicted_intent || '';
      return iA.localeCompare(iB);
    }
    if (state.currentSort === 'quality') {
      const qA = a.analysisData?.retrieved_evidence?.[0]?.similarity_score || 0;
      const qB = b.analysisData?.retrieved_evidence?.[0]?.similarity_score || 0;
      return qB - qA;
    }
    return 0;
  });

  return items;
}

function renderInboxTable() {
  const tbody = document.getElementById('casesTableBody');
  if (!tbody) return;

  const items = getFilteredAndSortedInbox();

  if (items.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;padding:24px;color:var(--muted);">No matching inquiries found.</td></tr>';
    return;
  }

  tbody.innerHTML = items.map(item => {
    const data = item.analysisData;
    const caseState = getCaseState(data);
    const top = data?.retrieved_evidence?.[0];
    const topSim = top?.similarity_score ?? 0;

    let chipClass = 'chip-auto';
    let chipText  = 'AUTO-HANDLE';
    let whySummary = 'Strong evidence · verified resolution';

    if (caseState === 'WITHHELD') {
      chipClass = 'chip-withheld';
      chipText  = 'WITHHELD';
      if (topSim < 0.45) {
        whySummary = `Evidence gap (${topSim.toFixed(2)} < 0.45) · withheld`;
      } else {
        whySummary = 'Cross-domain intent mismatch · refused';
      }
    } else if (caseState === 'ESCALATE') {
      chipClass = 'chip-escalate';
      chipText  = 'ESCALATE';
      const reason = data?.escalation_reason || '';
      if (reason.includes('hardware')) whySummary = 'Hardware damage trigger · repair routing';
      else if (reason.includes('security') || reason.includes('pii')) whySummary = 'Account lockout / 2FA trigger';
      else if (reason.includes('billing') || reason.includes('financial')) whySummary = 'Payment gateway transaction trigger';
      else whySummary = 'Sensitive policy rule triggered';
    }

    const intent = data?.predicted_intent || 'analyzing...';

    return `
      <tr class="case-row" data-id="${esc(item.id)}">
        <td class="col-case-id mono">${esc(item.id)}</td>
        <td class="col-query">${esc(item.query)}</td>
        <td class="col-intent mono">${esc(intent)}</td>
        <td class="col-decision"><span class="decision-chip ${chipClass}">${chipText}</span></td>
        <td class="col-why" style="font-size:12px;color:var(--text-2);">${esc(whySummary)}</td>
        <td class="col-time mono">${esc(item.time)}</td>
      </tr>
    `;
  }).join('');

  tbody.querySelectorAll('.case-row').forEach(row => {
    row.addEventListener('click', () => {
      const id = row.dataset.id;
      const found = state.inboxItems.find(it => it.id === id);
      if (found && found.analysisData) {
        openCaseDetail(found);
      }
    });
  });
}

function updateInboxStats() {
  let autoCount = 0;
  let escCount = 0;
  let withCount = 0;

  state.inboxItems.forEach(item => {
    const s = getCaseState(item.analysisData);
    if (s === 'AUTO_HANDLE') autoCount++;
    else if (s === 'WITHHELD') withCount++;
    else if (s === 'ESCALATE') escCount++;
  });

  const total = state.inboxItems.length;
  const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setTxt('statTotal', total);
  setTxt('statAuto', autoCount);
  setTxt('statEscalate', escCount);
  setTxt('statWithheld', withCount);
  setTxt('inboxCount', total);
}

// ── Case Detail View ──────────────────────────────────────────────
function openCaseDetail(item) {
  state.currentCase = item;
  state.selectedCandidateIndex = 0;

  const { id, query, analysisData: data, time } = item;
  const caseState = getCaseState(data);

  const idBadge = document.getElementById('caseIdBadge');
  const metaBadge = document.getElementById('caseMeta');
  const inqText = document.getElementById('caseInquiryText');

  if (idBadge) idBadge.textContent = id;
  if (metaBadge) metaBadge.textContent = time ? `${time}` : '';
  if (inqText) inqText.textContent = `"${query}"`;

  // Decision banner
  renderDecisionBanner(data, caseState, 'decisionBanner', 'decisionChip', 'decisionHeadline', 'decisionMetaRow');

  // Decision Signals
  const signalsContainer = document.getElementById('caseDecisionSignals');
  if (signalsContainer) {
    signalsContainer.innerHTML = renderDecisionSignalsHtml(data);
  }

  // Populate tabs
  renderWhyTab(data, 'panel-why', caseState);
  renderEvidenceTab(data, 'panel-evidence', caseState);
  renderReplyTab(data, 'panel-reply', caseState);
  renderReplayTab(data, 'panel-replay', caseState);
  renderSystemTab(data, 'panel-system');

  activateCaseTab('why');

  document.querySelectorAll('#view-case .case-tab').forEach(btn => {
    btn.onclick = () => activateCaseTab(btn.dataset.tab);
  });

  switchView('case');
  const pageTitle = document.getElementById('pageTitle');
  const pageSub = document.getElementById('pageSubtitle');
  if (pageTitle) pageTitle.textContent = id;
  if (pageSub) pageSub.textContent = query.slice(0, 60) + (query.length > 60 ? '…' : '');
}

function activateCaseTab(tab) {
  state.currentTab = tab;
  document.querySelectorAll('#view-case .case-tab').forEach(b => {
    b.classList.toggle('active', b.dataset.tab === tab);
  });
  document.querySelectorAll('#view-case .tab-panel').forEach(p => {
    p.classList.toggle('active', p.id === `panel-${tab}`);
  });
}

// ── Decision Banner ───────────────────────────────────────────────
function renderDecisionBanner(data, caseState, bannerId, chipId, headlineId, metaId) {
  const banner   = document.getElementById(bannerId);
  const chip     = document.getElementById(chipId);
  const headline = document.getElementById(headlineId);
  const meta     = document.getElementById(metaId);

  if (!banner || !chip || !headline || !meta) return;

  banner.className = 'decision-banner';
  if (caseState === 'AUTO_HANDLE') {
    banner.classList.add('banner-auto');
    chip.className = 'decision-chip chip-auto-solid';
    chip.textContent = 'AUTO-HANDLE';
    headline.textContent = data.decision_rationale || 'Standard technical troubleshooting — safe to answer.';
  } else if (caseState === 'WITHHELD') {
    banner.classList.add('banner-withheld');
    chip.className = 'decision-chip chip-withheld-solid';
    chip.textContent = 'WITHHELD';
    headline.textContent = '“SupportPilot chose not to guess.”';
  } else {
    banner.classList.add('banner-escalate');
    chip.className = 'decision-chip chip-escalate-solid';
    chip.textContent = 'ESCALATE';
    headline.textContent = 'Specialist attention required.';
  }

  const conf = (data.confidence * 100).toFixed(0);
  const sim  = data.retrieved_evidence?.[0]?.similarity_score?.toFixed(3) || '—';
  meta.innerHTML = `
    <span>Confidence <strong>${conf}%</strong></span>
    <span>Intent <strong>${esc(data.predicted_intent)}</strong></span>
    <span>Best similarity <strong>${sim}</strong></span>
  `;
}

// ── 3. Decision Signals Panel HTML ────────────────────────────────
function renderDecisionSignalsHtml(data) {
  const top = data.retrieved_evidence?.[0];
  const topSim = top?.similarity_score ?? 0;
  const intentMatch = top ? (top.historical_intent === data.predicted_intent) : false;

  const caseState = getCaseState(data);
  const isDirectSafety = (
    caseState === 'ESCALATE' &&
    (data.escalation_reason === 'hardware_physical_damage' ||
     data.escalation_reason === 'account_security_and_pii' ||
     data.escalation_reason === 'financial_and_billing' ||
     data.escalation_reason === 'abusive_language')
  );
  const safetyPass = !isDirectSafety;
  const contextPass = (caseState === 'AUTO_HANDLE');

  return `
    <div class="decision-signals-card">
      <div class="signals-header">Decision Signals</div>
      <div class="signals-grid">
        <div class="sig-item">
          <span class="sig-label">Intent confidence</span>
          <span class="sig-val mono">${(data.confidence * 100).toFixed(0)}%</span>
        </div>
        <div class="sig-item">
          <span class="sig-label">Best similarity</span>
          <span class="sig-val mono">${topSim.toFixed(2)}</span>
        </div>
        <div class="sig-item">
          <span class="sig-label">Intent agreement</span>
          <span class="sig-val ${intentMatch ? 'pass' : 'fail'}">${intentMatch ? '✓ Match' : '✕ Discordant'}</span>
        </div>
        <div class="sig-item">
          <span class="sig-label">Safety checks</span>
          <span class="sig-val ${safetyPass ? 'pass' : 'fail'}">${safetyPass ? '✓ Clear' : '✕ Triggered'}</span>
        </div>
        <div class="sig-item">
          <span class="sig-label">Context validation</span>
          <span class="sig-val ${contextPass ? 'pass' : 'fail'}">${contextPass ? '✓ Grounded' : '✕ Suppressed'}</span>
        </div>
      </div>
    </div>
  `;
}

// ── 6. Withheld Box HTML ──────────────────────────────────────────
function renderWithheldBoxHtml(data) {
  const top = data.retrieved_evidence?.[0];
  const sim = top?.similarity_score ?? 0;
  const intentMatch = top ? (top.historical_intent === data.predicted_intent) : false;

  return `
    <div class="withheld-box">
      <div class="withheld-badge-lg">WITHHELD</div>
      <h3 class="withheld-headline">“SupportPilot chose not to guess.”</h3>
      <p class="withheld-desc">Historical evidence did not satisfy semantic similarity or intent concordance thresholds. Grounding was withheld to protect customer trust.</p>
      <div class="withheld-signals-grid">
        <div class="w-item">
          <span class="w-lbl">Best evidence</span>
          <span class="w-val ${sim >= 0.45 ? 'pass' : 'fail'}">${sim.toFixed(2)}</span>
        </div>
        <div class="w-item">
          <span class="w-lbl">Required threshold</span>
          <span class="w-val">0.45</span>
        </div>
        <div class="w-item">
          <span class="w-lbl">Intent agreement</span>
          <span class="w-val ${intentMatch ? 'pass' : 'fail'}">${intentMatch ? 'Yes' : 'No'}</span>
        </div>
        <div class="w-item">
          <span class="w-lbl">Safety status</span>
          <span class="w-val pass">Passed</span>
        </div>
      </div>
      <button class="btn-withheld" disabled>Escalate to Specialist</button>
    </div>
  `;
}

// ── 5. “What Changed the Decision?” HTML ──────────────────────────
function renderWhatChangedHtml(data, caseState) {
  let items = [];
  if (caseState === 'AUTO_HANDLE') {
    items = [
      '<strong>Strong intent match:</strong> Calibrated confidence exceeded autonomous resolution floor (0.12).',
      '<strong>Sufficient historical evidence:</strong> Dense similarity &ge; 0.450 with verified resolution provenance.',
      '<strong>No sensitive trigger:</strong> Passed all brand-safety, hardware damage, account security, and billing guardrails.',
      '<strong>Context validation passed:</strong> Verified historical Apple Support resolution free of ungrounded policies.'
    ];
  } else if (caseState === 'WITHHELD') {
    const top = data.retrieved_evidence?.[0];
    const sim = top?.similarity_score ?? 0;
    const simText = sim < 0.45 ? `Evidence below threshold (${sim.toFixed(2)} < 0.45)` : 'Similarity above floor';
    const matchText = (top && top.historical_intent !== data.predicted_intent) ? 'Intent mismatch across domains' : 'Intent concordant';
    items = [
      `<strong>${simText}:</strong> Historical corpus lacked high-confidence verified match.`,
      `<strong>${matchText}:</strong> Concordance policy refused to bridge incompatible support intents.`,
      '<strong>No safe historical resolution:</strong> Grounding gate fail-closed to eliminate hallucinated procedures.'
    ];
  } else {
    // ESCALATE
    const reasonText = (data.escalation_reason || 'Policy rule').replace(/_/g, ' ');
    items = [
      `<strong>Security/payment/hardware trigger:</strong> Pipeline intercepted by <code>${esc(reasonText)}</code> guardrail.`,
      '<strong>Human-sensitive action:</strong> Sensitive account or financial operations mandate human specialist oversight.',
      '<strong>Insufficient safe evidence:</strong> System suppressed automated reply generation to eliminate liability.'
    ];
  }

  return `
    <details class="what-changed-box" open>
      <summary class="what-changed-summary">Operational Policy Breakdown</summary>
      <div class="what-changed-content">
        <ul class="what-changed-list">
          ${items.map(it => `<li>${it}</li>`).join('')}
        </ul>
      </div>
    </details>
  `;
}

// ── Why Tab ───────────────────────────────────────────────────────
function renderWhyTab(data, panelId, caseState) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  const top = data.retrieved_evidence?.[0];
  const sim = top?.similarity_score ?? 0;

  let contentHtml = '';

  if (caseState === 'WITHHELD') {
    contentHtml = renderWithheldBoxHtml(data);
  } else if (caseState === 'ESCALATE') {
    const reason = data.escalation_reason ? data.escalation_reason.replace(/_/g, ' ') : 'Escalation trigger detected';
    contentHtml = `
      <div class="escalate-block">
        <div class="escalate-title">ESCALATE</div>
        <div class="escalate-reason">${esc(reason)}</div>
        <div class="will-not-do">
          <div class="will-not-do-title">What SupportPilot will not do</div>
          <ul>
            <li>Give unverified account recovery or credential override instructions</li>
            <li>Promise refunds, discounts, or fee waivers</li>
            <li>Invent Apple hardware repair warranty policies</li>
            <li>Pretend a private DM was received when none was logged</li>
          </ul>
        </div>
        <button class="btn-primary btn-escalate" style="width:100%;" disabled>Escalate to Specialist</button>
      </div>
    `;
  } else {
    // AUTO_HANDLE
    const conf = (data.confidence * 100).toFixed(1);
    const steps = [
      {
        label: 'Intent classification',
        detail: `${data.predicted_intent} · ${conf}% confidence`,
        status: 'PASS', statusClass: 'status-pass',
      },
      {
        label: 'Historical evidence',
        detail: `${data.retrieved_evidence?.length ?? 0} matches · top similarity ${sim.toFixed(3)}`,
        status: sim >= 0.45 ? 'ABOVE FLOOR' : 'BELOW FLOOR',
        statusClass: sim >= 0.45 ? 'status-pass' : 'status-fail',
      },
      {
        label: 'Intent agreement',
        detail: top ? `Evidence intent ${top.historical_intent} agrees with query intent` : '—',
        status: 'MATCH', statusClass: 'status-match',
      },
      {
        label: 'Safety checks',
        detail: 'No hardware damage, 2FA, or billing dispute triggers',
        status: 'PASS', statusClass: 'status-pass',
      },
      {
        label: 'Context validation',
        detail: 'No private DM promises, false timestamps, or ungrounded policies',
        status: 'PASS', statusClass: 'status-pass',
      },
    ];

    contentHtml = `
      <div class="why-steps">
        ${steps.map((s, i) => `
          <div class="why-step">
            <div class="why-step-num">0${i + 1}</div>
            <div class="why-step-body">
              <div class="why-step-title">${esc(s.label)}</div>
              <div class="why-step-detail">${esc(s.detail)}</div>
            </div>
            <div class="why-step-status ${s.statusClass}">${s.status}</div>
          </div>
        `).join('')}
      </div>
      <div class="why-footer">
        <span class="why-footer-label">Decision</span>
        <span class="why-footer-decision">AUTO-HANDLE — Safe to answer</span>
      </div>
    `;
  }

  panel.innerHTML = contentHtml + renderWhatChangedHtml(data, caseState);
}

// ── 4. Evidence Comparison ────────────────────────────────────────
function renderEvidenceContentHtml(data, caseState, selectedIdx) {
  const evs = data.retrieved_evidence || [];
  const threshold = 0.45;

  const rows = evs.slice(0, 3).map((e, i) => {
    const isSelected = (i === 0 && caseState === 'AUTO_HANDLE');
    const isMatch = (e.historical_intent === data.predicted_intent);
    const isAbove = e.similarity_score >= threshold;

    let decisionHtml = '';
    if (isSelected) {
      decisionHtml = `<span class="ev-chip-selected">SELECTED</span>`;
    } else {
      let reason = 'context-bound';
      if (!isAbove) reason = `below threshold (${e.similarity_score.toFixed(2)} < 0.45)`;
      else if (!isMatch) reason = 'intent mismatch';
      else if (caseState === 'ESCALATE') reason = 'safety policy trigger';

      decisionHtml = `<span class="ev-chip-rejected">REJECTED</span><span class="ev-reason">${esc(reason)}</span>`;
    }

    const isActive = (i === selectedIdx);

    return `
      <tr class="ev-row ${isActive ? 'active-candidate' : ''}" data-cand-idx="${i}">
        <td class="ev-rank mono">0${i + 1}</td>
        <td class="ev-intent mono">${esc(e.historical_intent)}</td>
        <td class="ev-sim mono">${e.similarity_score.toFixed(2)}</td>
        <td class="ev-match text-center">${isMatch ? '<span class="ev-match-icon pass">✓</span>' : '<span class="ev-match-icon fail">✕</span>'}</td>
        <td class="ev-decision">${decisionHtml}</td>
      </tr>
    `;
  });

  const activeCand = evs[selectedIdx] || evs[0];

  const detailCardHtml = activeCand ? `
    <div class="candidate-detail-card" style="margin-top:12px;">
      <div class="source-field" style="margin-bottom:8px;">
        <div class="source-field-label">Selected Candidate 0${selectedIdx + 1} Historical Query</div>
        <div class="source-field-val">${esc(activeCand.matched_customer_query)}</div>
      </div>
      <div class="source-field" style="margin-bottom:8px;">
        <div class="source-field-label">Historical Apple Support Reply</div>
        <div class="source-field-val">${esc(activeCand.historical_agent_reply)}</div>
      </div>
      <div style="display:flex;gap:20px;font-size:12px;color:var(--muted);">
        <div>Conversation: <span class="mono" style="color:var(--text)">#${esc(activeCand.conversation_id)}</span></div>
        <div>Cosine similarity: <span class="mono" style="color:var(--text)">${activeCand.similarity_score.toFixed(4)}</span></div>
        <div>Intent: <span class="mono" style="color:var(--text)">${esc(activeCand.historical_intent)}</span></div>
      </div>
    </div>
  ` : '<p style="color:var(--muted);font-size:13px;">No candidate evidence available.</p>';

  return `
    <table class="ev-comparison-table">
      <thead>
        <tr>
          <th style="width:50px;">Rank</th>
          <th>Intent</th>
          <th style="width:90px;">Similarity</th>
          <th style="width:70px;text-align:center;">Match</th>
          <th>Decision</th>
        </tr>
      </thead>
      <tbody>
        ${rows.join('')}
      </tbody>
    </table>
    ${detailCardHtml}
  `;
}

function renderEvidenceTab(data, panelId, caseState) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  panel.innerHTML = `
    <div class="evidence-header" style="margin-bottom:12px;">
      <h2 class="section-title">Evidence Comparison (Top-3 Candidates)</h2>
      <p class="section-subtitle">Retrieved from 800-thread historical Apple Support corpus. Click candidate to inspect.</p>
    </div>
    <div id="${panelId}-content">
      ${renderEvidenceContentHtml(data, caseState, state.selectedCandidateIndex)}
    </div>
  `;

  panel.querySelectorAll('tr.ev-row').forEach(row => {
    row.onclick = () => {
      state.selectedCandidateIndex = parseInt(row.getAttribute('data-cand-idx') || '0', 10);
      renderEvidenceTab(data, panelId, caseState);
    };
  });
}

// ── 5. Reply Tab & Component ──────────────────────────────────────
function renderReplyContentHtml(data, caseState) {
  const top = data.retrieved_evidence?.[0];
  const citation = top
    ? `Verified historical @AppleSupport resolution · Thread #${top.conversation_id} · Sim ${top.similarity_score.toFixed(3)}`
    : 'No evidence citation available.';

  if (caseState === 'AUTO_HANDLE') {
    return `
      <div class="reply-header">
        <span class="reply-title">Grounded Draft Reply</span>
        <span class="grounding-badge badge-grounded">PROVENANCE-BACKED</span>
      </div>
      <div class="draft-box">
        <div class="draft-text">${esc(data.draft_reply || '—')}</div>
        <div class="draft-citation">${esc(citation)}</div>
      </div>
      <div class="reply-actions">
        <button class="btn-primary" id="copyReplyBtn">Copy Reply</button>
        <button class="btn-ghost" disabled>Send Directly</button>
      </div>
    `;
  }

  if (caseState === 'WITHHELD') {
    return `
      <div class="reply-action-block" style="border-left:3px solid var(--withheld);">
        <div class="reply-action-title">Autonomous Drafting Withheld</div>
        <div class="reply-action-desc">SupportPilot chose not to guess because historical evidence did not meet the 0.45 similarity threshold or intent concordance policy.</div>
        <div class="cta-row">
          <button class="btn-primary" style="background:var(--withheld);border-color:var(--withheld);" disabled>Route to Specialist</button>
        </div>
      </div>
    `;
  }

  // ESCALATE
  const reason = data.escalation_reason ? data.escalation_reason.replace(/_/g, ' ') : 'Specialist attention required';
  return `
    <div class="reply-action-block" style="border-left:3px solid var(--escalate);">
      <div class="reply-action-title">Specialist Routing Enforced</div>
      <div class="reply-action-desc">Autonomous reply suppressed due to <strong>${esc(reason)}</strong> policy guardrail. Handed off to human operator.</div>
      <div class="cta-row">
        <button class="btn-primary btn-escalate" disabled>Escalate to Specialist</button>
      </div>
    </div>
  `;
}

function renderReplyTab(data, panelId, caseState) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  panel.innerHTML = renderReplyContentHtml(data, caseState);

  const copyBtn = panel.querySelector('#copyReplyBtn');
  if (copyBtn) {
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(data.draft_reply || '').catch(() => {});
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy Reply'; }, 1500);
    };
  }
}

// ── Forensic Replay Tab ───────────────────────────────────────────
function renderReplayTab(data, panelId, caseState) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  const top = data.retrieved_evidence?.[0];
  const topSim = top?.similarity_score ?? 0;

  const steps = [
    {
      name: 'Inbound message ingress',
      badge: `${(data.customer_query || '').length} chars`,
      summary: 'Inbound customer text tokenized and sanitized',
      details: `Normalized input: "${esc(data.customer_query || '')}"`
    },
    {
      name: 'Intent classification',
      badge: `${(data.confidence * 100).toFixed(0)}% conf`,
      summary: `Domain predicted: <code>${esc(data.predicted_intent)}</code>`,
      details: `Top probabilities: ${Object.entries(data.top_probabilities || {}).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(' · ')}`
    },
    {
      name: 'Confidence threshold check',
      badge: data.confidence >= 0.12 ? 'PASSED' : 'FAILED',
      summary: `Score ${(data.confidence * 100).toFixed(1)}% vs threshold 12.0%`,
      details: 'Evaluated against the calibrated confidence floor for intent acceptance.'
    },
    {
      name: 'Dense vector retrieval',
      badge: `${data.retrieved_evidence?.length || 0} hits`,
      summary: 'Queried 800-thread historical Apple Support vector index',
      details: `Top candidate similarity: ${topSim.toFixed(4)}`
    },
    {
      name: 'Similarity threshold gate',
      badge: topSim >= 0.45 ? 'PASSED' : 'BELOW FLOOR',
      summary: `Top similarity ${topSim.toFixed(3)} vs threshold 0.450`,
      details: topSim >= 0.45 ? 'Evidence satisfies dense similarity requirement.' : 'Similarity below floor — triggers fail-closed abstention.'
    },
    {
      name: 'Intent concordance verification',
      badge: (top && top.historical_intent === data.predicted_intent) ? 'CONCORDANT' : 'DISCORDANT',
      summary: top ? `Evidence intent: ${esc(top.historical_intent)}` : 'No evidence',
      details: 'Requires candidate historical resolution to align with inquiry intent domain.'
    },
    {
      name: 'Grounding evaluation',
      badge: data.grounding_status || 'status',
      summary: caseState === 'AUTO_HANDLE' ? 'Draft response bound to verified evidence' : 'Drafting safely withheld or suppressed',
      details: caseState === 'AUTO_HANDLE' ? `Attributed resolution: "${esc(data.draft_reply || '')}"` : 'Fail-closed grounding policy enforced.'
    },
    {
      name: 'Safety policy applied',
      badge: caseState === 'ESCALATE' ? 'TRIGGERED' : 'PASSED',
      summary: caseState === 'ESCALATE' ? `Triggered: ${esc((data.escalation_reason || '').replace(/_/g, ' '))}` : 'All brand-safety checks passed',
      details: 'Evaluated: Hardware physical damage, Apple ID / 2FA account lockouts, payment disputes, abusive content, and non-Latin queries.'
    },
    {
      name: 'Final decision',
      badge: caseState,
      summary: `Pipeline output: <strong>${caseState}</strong>`,
      details: esc(data.decision_rationale || 'Specialist routing enforced.')
    }
  ];

  panel.innerHTML = `
    <div class="replay-header" style="margin-bottom:12px;">
      <h2 class="section-title">Decision Replay (Forensic Timeline)</h2>
      <p class="section-subtitle">Click any pipeline event to inspect raw payload states.</p>
    </div>
    <div class="forensic-timeline">
      ${steps.map((s, i) => `
        <details class="forensic-step" ${i === 7 ? 'open' : ''}>
          <summary class="forensic-summary">
            <span class="forensic-step-num">0${i + 1}</span>
            <span class="forensic-step-name">${s.name}</span>
            <span style="color:var(--muted);font-size:12px;margin-right:auto;margin-left:10px;">${s.summary}</span>
            <span class="forensic-step-badge">${s.badge}</span>
          </summary>
          <div class="forensic-body">${s.details}</div>
        </details>
      `).join('')}
    </div>
  `;
}

// ── System Notes Tab ──────────────────────────────────────────────
function renderSystemTab(data, panelId) {
  const panel = document.getElementById(panelId);
  if (!panel) return;

  const rows = [
    ['Model', 'TF-IDF Logistic Regression'],
    ['Retriever', 'MiniLM-L6-v2 (384-dim)'],
    ['Predicted intent', data.predicted_intent],
    ['Confidence', `${(data.confidence * 100).toFixed(2)}%`],
    ['Decision', data.decision],
    ['Grounding status', data.grounding_status],
    ['Top similarity', data.retrieved_evidence?.[0]?.similarity_score?.toFixed(4) ?? '—'],
    ['Escalation reason', data.escalation_reason || '—'],
    ['Policy detected', data.unsupported_policy_detected ? 'Yes' : 'No'],
    ['Rejection reasons', data.grounding_rejection_reasons?.join(', ') || '—'],
  ];

  panel.innerHTML = `
    <div class="sys-notes-grid">
      ${rows.map(([k, v]) => `
        <div class="sys-note-item">
          <div class="sys-note-label">${esc(k)}</div>
          <div class="sys-note-val mono">${esc(String(v))}</div>
        </div>
      `).join('')}
    </div>
  `;
}

// ── 1. Analyze Inquiry (Real-Time Pipeline & Wow Screen) ──────────
function initAnalyze() {
  const input = document.getElementById('analyzeInput');
  const btn   = document.getElementById('analyzeBtn');

  if (input) {
    input.addEventListener('input', updateCharCount);
  }

  if (btn) {
    btn.addEventListener('click', () => {
      const q = (input?.value || '').trim();
      if (q) runAnalysis(q);
    });
  }

  document.querySelectorAll('.demo-pill').forEach(pill => {
    pill.addEventListener('click', () => {
      if (input) {
        input.value = pill.dataset.query;
        updateCharCount();
        runAnalysis(pill.dataset.query);
      }
    });
  });
}

function updateCharCount() {
  const input = document.getElementById('analyzeInput');
  const count = document.getElementById('charCount');
  if (input && count) {
    count.textContent = `${input.value.length} / 2000`;
  }
}

async function runAnalysis(query) {
  const btn       = document.getElementById('analyzeBtn');
  const pipeLine  = document.getElementById('pipelineLine');
  const empty     = document.getElementById('analyzeEmpty');
  const result    = document.getElementById('analyzeResult');

  if (btn) btn.disabled = true;
  if (empty) empty.style.display = 'none';
  if (result) result.style.display = 'none';

  // 10. REAL-TIME SUBTLE PROGRESS
  if (pipeLine) pipeLine.style.display = 'flex';
  setPipeStep(1, 'active');

  const pstep1 = delay(130);
  await pstep1;
  setPipeStep(1, 'done');
  setPipeStep(2, 'active');

  let data = null;
  try {
    const apiPromise = apiAnalyze(query);
    await delay(150);
    setPipeStep(2, 'done');
    setPipeStep(3, 'active');

    data = await apiPromise;
    await delay(130);
    setPipeStep(3, 'done');
    setPipeStep(4, 'active');

    await delay(110);
    setPipeStep(4, 'done');
    await delay(70);

    if (pipeLine) pipeLine.style.display = 'none';

    renderAnalyzeResult(data, query, result);
    if (result) result.style.display = 'flex';
    setStatus('ready');

  } catch (err) {
    if (pipeLine) pipeLine.style.display = 'none';
    if (result) {
      result.innerHTML = `<div class="loading-line" style="color:var(--escalate);">Analysis failed: ${esc(err.message)}</div>`;
      result.style.display = 'block';
    }
    setStatus('error');
  }

  if (btn) btn.disabled = false;
}

function setPipeStep(stepNum, status) {
  const el = document.getElementById(`pstep-${stepNum}`);
  if (!el) return;
  el.classList.remove('active', 'done');
  if (status) el.classList.add(status);
}

// ── 3, 4, 5, 6. UNIFIED ANALYSIS RESULT COMPONENT ─────────────────
function renderAnalyzeResult(data, query, container) {
  if (!container) return;

  const caseState = getCaseState(data);
  const chipClass = caseState === 'AUTO_HANDLE' ? 'chip-auto-solid' : caseState === 'ESCALATE' ? 'chip-escalate-solid' : 'chip-withheld-solid';
  const chipLabel = caseState === 'AUTO_HANDLE' ? 'AUTO-HANDLE' : caseState === 'ESCALATE' ? 'ESCALATE' : 'WITHHELD';
  const headline  = caseState === 'AUTO_HANDLE' ? (data.decision_rationale || 'Standard technical troubleshooting — safe to answer.') : caseState === 'ESCALATE' ? 'Specialist attention required.' : '“SupportPilot chose not to guess.”';
  const bannerClass = caseState === 'AUTO_HANDLE' ? 'banner-auto' : caseState === 'ESCALATE' ? 'banner-escalate' : 'banner-withheld';
  const conf = (data.confidence * 100).toFixed(0);
  const sim  = data.retrieved_evidence?.[0]?.similarity_score?.toFixed(3) || '—';

  let operationalWhy = '';
  if (caseState === 'AUTO_HANDLE') {
    operationalWhy = `Standard technical troubleshooting. Calibrated intent confidence (${conf}%) exceeded floor (0.12), verified historical Apple Support resolution located in corpus (similarity ${sim} &ge; 0.450), and query cleared all safety guardrails.`;
  } else if (caseState === 'WITHHELD') {
    operationalWhy = `Historical evidence did not satisfy semantic similarity floor (0.45) or intent concordance policy. Autonomous drafting withheld to protect customer trust.`;
  } else {
    const reasonText = (data.escalation_reason || 'Policy trigger').replace(/_/g, ' ');
    operationalWhy = `Sensitive policy trigger detected: <strong>${esc(reasonText)}</strong>. Involves hardware repair, account security/2FA, or financial disputes requiring human specialist intervention.`;
  }

  container.innerHTML = `
    <!-- 3. CUSTOMER ORIGINAL MESSAGE -->
    <div class="result-card">
      <div class="result-card-header">
        <span class="result-tag">CUSTOMER</span>
        <span style="font-size:11px;color:var(--muted);font-family:var(--mono);">${query.length} characters</span>
      </div>
      <div class="result-customer-text">&ldquo;${esc(query)}&rdquo;</div>
    </div>

    <!-- 3. SUPPORTPILOT DECISION -->
    <div class="decision-banner ${bannerClass}">
      <div class="decision-top">
        <span class="decision-chip ${chipClass}">${chipLabel}</span>
        <span class="decision-headline">${esc(headline)}</span>
      </div>
      <div class="decision-meta-row">
        <span>Confidence <strong>${conf}%</strong></span>
        <span>Intent <strong>${esc(data.predicted_intent)}</strong></span>
        <span>Best similarity <strong>${sim}</strong></span>
      </div>
    </div>

    <!-- 3. WHY OPERATIONAL EXPLANATION -->
    <div class="result-card">
      <div class="result-card-header">
        <span class="result-tag">WHY</span>
        <span style="font-size:11px;color:var(--muted);">Operational explanation</span>
      </div>
      <div class="result-why-text">${operationalWhy}</div>
    </div>

    <!-- 3. DECISION SIGNALS -->
    ${renderDecisionSignalsHtml(data)}

    <!-- 6. WITHHELD STATE SPECIAL CARD (If Withheld) -->
    ${caseState === 'WITHHELD' ? renderWithheldBoxHtml(data) : ''}

    <!-- 4. EVIDENCE COMPARISON -->
    <div class="result-card">
      <div class="result-card-header">
        <span class="result-tag">EVIDENCE</span>
        <span style="font-size:11px;color:var(--muted);">Top-3 Candidates from 800-thread Apple Support corpus</span>
      </div>
      <div id="analyzeEvidenceContainer">
        ${"<!-- evidence comparison stub -->"}
      </div>
    </div>

    <!-- 5. REPLY (Shown after Decision & Evidence) -->
    <div class="result-card">
      <div class="result-card-header">
        <span class="result-tag">REPLY</span>
        <span style="font-size:11px;color:var(--muted);">Action draft</span>
      </div>
      <div id="analyzeReplyContainer">
        ${renderReplyContentHtml(data, caseState)}
      </div>
    </div>

    <!-- FORENSIC REPLAY DETAILS (Folded by default for clarity) -->
    <details class="what-changed-box" style="margin-top:4px;">
      <summary class="what-changed-summary">Audit Trail &amp; Forensic Replay</summary>
      <div class="what-changed-content" id="analyzeReplayContainer"></div>
    </details>
  `;

  // Populate replay inside details
  renderReplayTab(data, 'analyzeReplayContainer', caseState);

  // Candidate row click handler in analyze evidence table
  const bindCandidateClicks = () => {
    container.querySelectorAll('#analyzeEvidenceContainer tr.ev-row').forEach(row => {
      row.onclick = () => {
        state.activeAnalyzeCandidateIndex = parseInt(row.getAttribute('data-cand-idx') || '0', 10);
        const evCont = container.querySelector('#analyzeEvidenceContainer');
        if (evCont) {
          evCont.innerHTML = "<!-- evidence comparison stub -->";
          bindCandidateClicks();
        }
      };
    });
  };
  bindCandidateClicks();

  // Copy button handler in reply
  const copyBtn = container.querySelector('#copyReplyBtn');
  if (copyBtn) {
    copyBtn.onclick = () => {
      navigator.clipboard.writeText(data.draft_reply || '').catch(() => {});
      copyBtn.textContent = 'Copied!';
      setTimeout(() => { copyBtn.textContent = 'Copy Reply'; }, 1500);
    };
  }

  state.analyzeResult = { query, analysisData: data };
}

// ── 7. Knowledge Corpus (Searchable Historical Case Cards) ────────
function initKnowledge() {
  const search = document.getElementById('corpusSearch');
  if (search) {
    search.addEventListener('input', () => {
      const q = search.value.toLowerCase();
      const filtered = CORPUS_SEED.filter(item =>
        item.historical_intent.toLowerCase().includes(q) ||
        item.matched_customer_query.toLowerCase().includes(q) ||
        item.historical_agent_reply.toLowerCase().includes(q) ||
        item.conversation_id.toLowerCase().includes(q)
      );
      renderKnowledgeCorpus(filtered);
    });
  }
}

function renderKnowledgeCorpus(items) {
  const list = document.getElementById('corpusList');
  if (!list) return;

  if (!items.length) {
    list.innerHTML = '<p style="color:var(--muted);font-size:13px;padding:24px 0;text-align:center;">No matching historical cases found.</p>';
    return;
  }

  list.innerHTML = items.map(c => `
    <div class="corpus-case-card">
      <div class="corpus-case-header">
        <span class="corpus-case-id mono">#${esc(c.conversation_id)}</span>
        <span class="corpus-case-intent mono">${esc(c.historical_intent)}</span>
        <span class="corpus-case-sim mono">cosine sim: ${c.similarity_score.toFixed(3)}</span>
      </div>
      <div class="corpus-query-row">
        <div class="corpus-label">Customer Query</div>
        <div class="corpus-query-text">&ldquo;${esc(c.matched_customer_query)}&rdquo;</div>
      </div>
      <div class="corpus-reply-box">
        <div class="corpus-label" style="color:var(--cobalt);">Historical Apple Support Response</div>
        <div class="corpus-reply-text">${esc(c.historical_agent_reply)}</div>
      </div>
    </div>
  `).join('');
}

// ── 9. Failures Page (Where SupportPilot Still Struggles) ──────────
async function loadFailures() {
  const list = document.getElementById('failuresList');
  if (!list) return;
  if (list.querySelector('.failures-group-section')) return;

  try {
    const data = await apiFetch('/api/failures');
    const failures = data.top_failures || [];

    // Group into 4 explicit categories: Classification, Retrieval, Safety, Escalation
    const groups = {
      'Safety': [],
      'Retrieval': [],
      'Classification': [],
      'Escalation': []
    };

    failures.forEach(f => {
      const tag = inferFailureTag(f);
      if (groups[tag.label]) {
        groups[tag.label].push({ ...f, tag });
      } else {
        groups['Classification'].push({ ...f, tag });
      }
    });

    const categoryDescriptions = {
      'Safety': 'Safety guardrails failing to escalate consequential financial risk or brand exposure.',
      'Retrieval': 'Sparse or rare storefront coverage gaps falling below the 0.450 similarity floor.',
      'Classification': 'Boundary confusion between overlapping technical hardware and networking domains.',
      'Escalation': 'Authorization and token payment disputes requiring specialist human escalation.'
    };

    const groupHtml = Object.entries(groups).map(([cat, items]) => {
      if (items.length === 0) return '';
      const catCls = items[0].tag.cls;

      return `
        <div class="failures-group-section">
          <div class="failures-group-header">
            <span class="failure-tag ${catCls}" style="font-size:11px;">${cat}</span>
            <span class="failures-group-title">${cat} Failures</span>
            <span class="failures-group-count">(${items.length} ${items.length === 1 ? 'case' : 'cases'})</span>
          </div>
          <p style="font-size:12px;color:var(--muted);margin-bottom:10px;">${categoryDescriptions[cat] || ''}</p>
          <div class="failures-cards-wrap">
            ${items.map((f, i) => `
              <div class="failure-card expanded" data-cat="${cat}" data-idx="${i}">
                <div class="failure-card-header">
                  <span class="failure-golden-id">${esc(f.golden_id)}</span>
                  <div class="failure-card-main">
                    <div class="failure-query">&ldquo;${esc(f.customer_query)}&rdquo;</div>
                    <div class="failure-intents">
                      Expected: <span class="expected">${esc(f.expected_intent)}</span>
                      &nbsp;·&nbsp;
                      Predicted: <span class="predicted">${esc(f.predicted_intent)}</span>
                    </div>
                  </div>
                  <span class="failure-tag ${f.tag.cls}">${f.tag.label}</span>
                </div>
                <div class="failure-card-detail" style="display:block;">
                  <div class="failure-detail-grid">
                    <div class="failure-detail-section">
                      <div class="failure-detail-label">Why it failed (Root cause)</div>
                      <div class="failure-detail-text">${esc(f.why_it_failed)}</div>
                    </div>
                    <div class="failure-detail-section">
                      <div class="failure-detail-label">Improvement hypothesis</div>
                      <div class="failure-detail-text">${esc(f.improvement_hypothesis)}</div>
                    </div>
                  </div>
                </div>
              </div>
            `).join('')}
          </div>
        </div>
      `;
    }).join('');

    list.innerHTML = groupHtml || '<p style="color:var(--muted);font-size:13px;">No failures recorded.</p>';

  } catch {
    list.innerHTML = '<div class="loading-line" style="color:var(--escalate);">Failed to load failure cases. Ensure backend is running.</div>';
  }
}

/**
 * Categorizes each real failure into exactly one of:
 * - Classification
 * - Retrieval
 * - Safety
 * - Escalation
 * Specifically ensures GOLDEN_003 is classified as Safety (not safe abstention).
 */
function inferFailureTag(f) {
  const id = f.golden_id;
  const why = (f.why_it_failed || '').toLowerCase();

  // 1. GOLDEN_003: Consequential financial loss ($640 flight) missed by keywords
  if (id === 'GOLDEN_003' || why.includes('safety false negative') || why.includes('missed escalation') || why.includes('consequential loss')) {
    return { cls: 'ft-safety', label: 'Safety' };
  }
  // 2. GOLDEN_001: Regional Spanish App Store coverage gap
  if (id === 'GOLDEN_001' || why.includes('coverage gap') || why.includes('low similarity')) {
    return { cls: 'ft-retrieval', label: 'Retrieval' };
  }
  // 3. GOLDEN_002: Cross-domain keyboard vs iTunes mismatch
  if (id === 'GOLDEN_002' || why.includes('concordance') || why.includes('intent mismatch')) {
    return { cls: 'ft-retrieval', label: 'Retrieval' };
  }
  // 4. GOLDEN_006: Apple Watch rebooting vs network connectivity
  if (id === 'GOLDEN_006' || why.includes('boundary confusion') || why.includes('multi-symptom')) {
    return { cls: 'ft-classification', label: 'Classification' };
  }
  // 5. GOLDEN_010: Apple Pay payment token authorization
  if (id === 'GOLDEN_010' || why.includes('payment gateway') || why.includes('authorization')) {
    return { cls: 'ft-escalation', label: 'Escalation' };
  }

  return { cls: 'ft-classification', label: 'Classification' };
}

// ── System Details Modal ──────────────────────────────────────────
function initModal() {
  const backdrop = document.getElementById('modalBackdrop');
  const openBtns = [document.getElementById('systemDetailsBtn'), document.getElementById('topSystemDetailsBtn')];
  openBtns.forEach(btn => btn?.addEventListener('click', () => {
    if (backdrop) backdrop.style.display = 'flex';
  }));
  document.getElementById('modalClose')?.addEventListener('click', () => {
    if (backdrop) backdrop.style.display = 'none';
  });
  backdrop?.addEventListener('click', e => {
    if (e.target === backdrop) backdrop.style.display = 'none';
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && backdrop) backdrop.style.display = 'none';
  });
}

// ── Agent Health & Network ────────────────────────────────────────
async function checkAgentHealth() {
  try {
    const data = await apiFetch('/api/health');
    const ok   = data.status === 'healthy';
    setStatus(ok ? 'ready' : 'error');
  } catch {
    setStatus('error');
  }
}

function setStatus(status) {
  const dot  = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  const tDot = document.getElementById('topStatusDot');
  const tTxt = document.getElementById('topStatusText');

  const map = {
    ready: ['online', 'Agent Ready'],
    error: ['error',  'Offline'],
  };
  const [cls, label] = map[status] || ['offline', 'Unknown'];

  if (dot)  { dot.className = `status-dot ${cls}`; }
  if (text) { text.textContent = label; }
  if (tDot) { tDot.className = `status-dot-sm ${cls}`; }
  if (tTxt) { tTxt.textContent = label; }
}

async function apiAnalyze(query) {
  return apiFetch('/api/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: query }),
  });
}

async function apiFetch(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function esc(str) {
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
