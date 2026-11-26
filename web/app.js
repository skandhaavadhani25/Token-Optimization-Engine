/**
 * Main application bootstrap, state management, and tab routing.
 */

import { api } from './api.js';
import { notify, friendlyError } from './toast.js';
import { keyManager } from './key_manager.js';
import {
  normalizeRetrievalResult,
  normalizeAskResult,
  normalizeBenchmarkOne,
  normalizeBenchmarkResult,
  normalizeStats,
} from './normalizer.js';
import { renderResultsTable } from './components/results_table.js';
import { renderTokenPanel } from './components/token_panel.js';
import { renderContextViewer } from './components/context_viewer.js';
import { renderChangeViewer } from './components/change_viewer.js';
import { renderBenchmarkTable } from './components/benchmark_table.js';
import { initGraphView } from './graph.js';
import { initChat } from './chat.js';

// LocalStorage Helpers
function getSavedWorkspaceId() {
  try {
    return localStorage.getItem('cce_active_workspace_id') || null;
  } catch (_) {
    return null;
  }
}

function saveActiveWorkspaceId(id) {
  try {
    if (id) localStorage.setItem('cce_active_workspace_id', id);
    else localStorage.removeItem('cce_active_workspace_id');
  } catch (_) {}
}

function getSavedProvider() {
  try {
    return localStorage.getItem('cce_last_provider') || null;
  } catch (_) {
    return null;
  }
}

function saveLastProvider(prov) {
  try {
    if (prov) localStorage.setItem('cce_last_provider', prov);
  } catch (_) {}
}

function getSavedModel() {
  try {
    return localStorage.getItem('cce_last_model') || null;
  } catch (_) {
    return null;
  }
}

function saveLastModel(model) {
  try {
    if (model) localStorage.setItem('cce_last_model', model);
  } catch (_) {}
}

// Application State (Single source of truth: activeWorkspaceId)
const state = {
  activeWorkspaceId: getSavedWorkspaceId(),
  activeWorkspace: null,
  repoPath: null,
  activeTab: 'ask',
  budget: 4000,
  currentBuiltContext: null,
  currentBenchmarkOne: null,
  graphInitialized: false,
  graphView: null,
  providers: [],
  selectedProvider: null,
  selectedModel: null,
  chatController: null,
};

// Error Message Formatter
export function formatErrorMessage(prefix, err) {
  const parts = [];
  if (err && err.code) parts.push(`code: ${err.code}`);
  if (err && err.requestId) parts.push(`request id: ${err.requestId}`);
  else if (err && err.request_id) parts.push(`request id: ${err.request_id}`);
  const meta = parts.length > 0 ? ` (${parts.join(', ')})` : '';
  const message = err ? (err.message || String(err)) : 'Unknown error';
  const out = `${prefix}${meta}: ${escapeHtml(message)}`;
  return keyManager.redactText(out);
}

// Component Boundary Render Error Formatter
export function renderUiError(container, componentName, err, rawData) {
  console.error(`UI render error in ${componentName}:`, err);
  const safeData = keyManager.redactText(JSON.stringify(rawData, null, 2));
  container.innerHTML = `
    <div class="error-banner error-ui">
      <div><strong>Could not display results (UI error):</strong> ${escapeHtml(err ? err.message : 'Unknown rendering error')}</div>
      <details class="debug-details" style="margin-top: 8px;">
        <summary style="cursor: pointer; font-size: 11px; color: var(--text-secondary);">Debug: raw response</summary>
        <pre style="font-size: 11px; max-height: 200px; overflow: auto; margin-top: 4px; background: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px;"><code>${escapeHtml(safeData)}</code></pre>
      </details>
    </div>
  `;
}

// Status Chip Controller
function setStatus(type, label, tooltip = '') {
  const chip = document.getElementById('status-chip');
  const text = document.getElementById('status-text');
  if (!chip || !text) return;

  chip.className = `status-chip status-${type}`;
  text.textContent = label;
  if (tooltip) {
    chip.title = tooltip;
  } else {
    chip.removeAttribute('title');
  }
}

// Tab Switching
function switchTab(tabId, graphView) {
  state.activeTab = tabId;

  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabId);
  });

  document.querySelectorAll('.tab-pane').forEach(pane => {
    pane.classList.toggle('active', pane.id === `pane-${tabId}`);
  });

  if (tabId === 'graph' && graphView) {
    if (!state.activeWorkspaceId && !state.repoPath) {
      graphView.clear();
    } else {
      graphView.loadOverview({
        workspaceId: state.activeWorkspaceId,
        workspaceName: state.activeWorkspace?.name,
        repoPath: state.repoPath,
      });
    }
  } else if (tabId === 'stats') {
    loadStats();
  }
}

// Analyze Handler
async function handleAnalyze() {
  const query = document.getElementById('input-query').value.trim();
  if (!query) {
    notify('Please enter a task or query string.');
    return;
  }

  const resultsSlot = document.getElementById('results-table-slot');
  const tokenSlot = document.getElementById('token-panel-slot');
  const contextSlot = document.getElementById('context-viewer-slot');

  if (!state.activeWorkspaceId && !state.repoPath) {
    resultsSlot.innerHTML = '<div class="empty-state">Add your code first: use "Open / Upload Code" at the top.</div>';
    return;
  }

  const btnAnalyze = document.getElementById('btn-analyze');
  btnAnalyze.disabled = true;
  btnAnalyze.textContent = 'Analyzing...';
  resultsSlot.innerHTML = '<div class="empty-state">Finding the relevant code...</div>';

  let res;
  try {
    const useSemantic = Boolean(document.getElementById('check-sig-semantic').checked);
    const useKeyword = Boolean(document.getElementById('check-sig-keyword').checked);
    const useGraph = Boolean(document.getElementById('check-sig-graph').checked);
    const budget = parseInt(document.getElementById('input-budget').value, 10);

    // Retrieve hybrid context scoped to active workspace
    res = await api.retrieveContext({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
      query,
      maxTokens: budget,
      useSemantic,
      useKeyword,
      useGraph,
      explain: true,
    });
  } catch (apiErr) {
    resultsSlot.innerHTML = `<div class="error-banner">${formatErrorMessage('API Error', apiErr)}</div>`;
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = 'Analyze & Retrieve';
    return;
  }

  if (res && res.error) {
    resultsSlot.innerHTML = `<div class="error-banner">${formatErrorMessage('API Error', res.error)}</div>`;
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = 'Analyze & Retrieve';
    return;
  }

  let normRet;
  try {
    normRet = normalizeRetrievalResult(res);
    state.currentBuiltContext = normRet.context;
  } catch (normErr) {
    renderUiError(resultsSlot, 'Normalizer', normErr, res);
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = 'Analyze & Retrieve';
    return;
  }

  try {
    renderResultsTable(normRet, resultsSlot);
  } catch (renderErr) {
    renderUiError(resultsSlot, 'ResultsTable', renderErr, res);
  }

  if (normRet.context) {
    try {
      renderContextViewer(normRet.context, contextSlot);
    } catch (renderErr) {
      renderUiError(contextSlot, 'ContextViewer', renderErr, res);
    }

    try {
      renderTokenPanel(normRet.context, state.currentBenchmarkOne, tokenSlot);
    } catch (renderErr) {
      renderUiError(tokenSlot, 'TokenPanel', renderErr, res);
    }
  }

  // Fetch single-query baseline reduction
  try {
    const budget = parseInt(document.getElementById('input-budget').value, 10);
    const bmOne = await api.benchmarkOne({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
      query,
      budget,
    });
    const normBmOne = normalizeBenchmarkOne(bmOne);
    state.currentBenchmarkOne = normBmOne;
    if (normRet.context) {
      renderTokenPanel(normRet.context, normBmOne, tokenSlot);
    }
  } catch (_) {
  } finally {
    btnAnalyze.disabled = false;
    btnAnalyze.textContent = 'Analyze & Retrieve';
  }
}

// Ask AI Handler
async function handleAsk() {
  const query = document.getElementById('input-query').value.trim();
  if (!query) {
    notify('Please enter a task or query string.');
    return;
  }

  const changeSlot = document.getElementById('change-viewer-slot');

  if (!state.activeWorkspaceId && !state.repoPath) {
    changeSlot.innerHTML = '<div class="empty-state">Add your code first: use "Open / Upload Code" at the top.</div>';
    return;
  }

  const btnAsk = document.getElementById('btn-ask');
  btnAsk.disabled = true;
  btnAsk.textContent = 'Generating solution...';
  changeSlot.innerHTML = '<div class="empty-state">Asking the AI model...</div>';

  const prov = state.selectedProvider;
  const model = state.selectedModel;
  const isDryRun = Boolean(document.getElementById('toggle-dry-run').checked);
  const budget = parseInt(document.getElementById('input-budget').value, 10);
  const enteredKey = prov ? keyManager.getKey(prov) : null;

  let res;
  try {
    res = await api.ask({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
      query,
      maxTokens: budget,
      dryRun: isDryRun,
      provider: prov,
      model,
      key: enteredKey,
    });
  } catch (apiErr) {
    let errorHtml = formatErrorMessage('AI Assistant Error', apiErr);
    if (apiErr && apiErr.code === 'authentication_error') {
      const pName = prov ? prov.charAt(0).toUpperCase() + prov.slice(1) : 'Provider';
      errorHtml = `
        <strong>Authentication Error:</strong> ${escapeHtml(apiErr.message)}<br>
        <span style="font-size: 12px;">Please check your ${pName} API key in the Model & Key settings.</span>
      `;
    }
    changeSlot.innerHTML = `<div class="error-banner">${errorHtml}</div>`;
    btnAsk.disabled = false;
    btnAsk.textContent = 'Ask AI Assistant';
    return;
  }

  if (res && res.error) {
    changeSlot.innerHTML = `<div class="error-banner">${formatErrorMessage('AI Assistant Error', res.error)}</div>`;
    btnAsk.disabled = false;
    btnAsk.textContent = 'Ask AI Assistant';
    return;
  }

  let normAsk;
  try {
    normAsk = normalizeAskResult(res);
  } catch (normErr) {
    renderUiError(changeSlot, 'Normalizer', normErr, res);
    btnAsk.disabled = false;
    btnAsk.textContent = 'Ask AI Assistant';
    return;
  }

  try {
    renderChangeViewer(normAsk, changeSlot, (pinnedEntity) => {
      const qInput = document.getElementById('input-query');
      qInput.value = `${qInput.value} (relevant: ${pinnedEntity})`;
      handleAnalyze();
    });
  } catch (renderErr) {
    renderUiError(changeSlot, 'ChangeViewer', renderErr, res);
  }

  if (normAsk.context) {
    state.currentBuiltContext = normAsk.context;
    const contextSlot = document.getElementById('context-viewer-slot');
    const tokenSlot = document.getElementById('token-panel-slot');
    try {
      renderContextViewer(normAsk.context, contextSlot);
    } catch (renderErr) {
      renderUiError(contextSlot, 'ContextViewer', renderErr, res);
    }
    try {
      renderTokenPanel(normAsk.context, state.currentBenchmarkOne, tokenSlot);
    } catch (renderErr) {
      renderUiError(tokenSlot, 'TokenPanel', renderErr, res);
    }
  }

  btnAsk.disabled = false;
  btnAsk.textContent = 'Ask AI Assistant';
}

// Incremental Update & Rebuild
async function handleIncrementalUpdate() {
  if (!state.activeWorkspaceId && !state.repoPath) {
    notify('Please open or upload a workspace first.');
    return;
  }

  const btn = document.getElementById('btn-update');
  btn.disabled = true;
  setStatus('indexing', 'updating');

  try {
    const report = await api.updateIndex({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
    });
    if (report && report.error) {
      throw new Error(report.error.message || 'Update failed');
    }
    loadStats();
    if (state.activeTab === 'graph' && state.graphView) {
      state.graphView.loadOverview({
        workspaceId: state.activeWorkspaceId,
        workspaceName: state.activeWorkspace?.name,
        repoPath: state.repoPath,
      });
    }
    notify(`Incremental update complete!\nModified: ${report.modified?.length || 0}, Added: ${report.added?.length || 0}, Deleted: ${report.deleted?.length || 0}`);
  } catch (err) {
    setStatus('stale', 'error');
    notify(`Incremental update failed: ${friendlyError(err)}`);
  } finally {
    btn.disabled = false;
  }
}

async function handleRebuild() {
  if (!state.activeWorkspaceId && !state.repoPath) {
    notify('Please open or upload a workspace first.');
    return;
  }

  const wsName = state.activeWorkspace?.name || state.repoPath;
  if (!confirm(`Force complete re-index of '${wsName}'? This will rebuild metadata, code graph, and vector embeddings.`)) {
    return;
  }

  const btn = document.getElementById('btn-rebuild');
  btn.disabled = true;
  setStatus('indexing', 'rebuilding');

  try {
    const report = await api.indexRepository({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
    }, true);
    if (report && report.error) {
      throw new Error(report.error.message || 'Rebuild failed');
    }
    loadStats();
    if (state.activeTab === 'graph' && state.graphView) {
      state.graphView.loadOverview({
        workspaceId: state.activeWorkspaceId,
        workspaceName: state.activeWorkspace?.name,
        repoPath: state.repoPath,
      });
    }
    notify('Full re-index completed successfully!');
  } catch (err) {
    setStatus('stale', 'error');
    notify(`Rebuild failed: ${friendlyError(err)}`);
  } finally {
    btn.disabled = false;
  }
}

// Benchmark Runner
async function handleRunBenchmark() {
  const selectQuerySet = document.getElementById('select-query-set');
  const queryFile = selectQuerySet ? selectQuerySet.value : '';
  if (!queryFile) {
    notify('Please select an evaluation query set.');
    return;
  }

  const slot = document.getElementById('benchmark-results-slot');
  if (!state.activeWorkspaceId && !state.repoPath) {
    if (slot) slot.innerHTML = '<div class="empty-state">Add your code first: use "Open / Upload Code" at the top.</div>';
    notify('Please open or upload a repository to run benchmarks.');
    return;
  }

  const callAi = document.getElementById('check-benchmark-dry-run')?.checked === false;
  const budget = parseInt(document.getElementById('input-bm-budget')?.value, 10) || 4000;
  const btn = document.getElementById('btn-run-benchmark');
  if (btn) {
    btn.disabled = true;
    btn.textContent = callAi ? 'Running (Calling AI)...' : 'Measuring Tokens...';
  }

  if (slot) slot.innerHTML = '<div class="empty-state">Running the benchmark. This can take a moment...</div>';

  try {
    const result = await api.runBenchmark({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
      queryFile,
      budget,
      callAi,
    });
    if (result && result.error) {
      slot.innerHTML = `<div class="error-banner">${formatErrorMessage('Benchmark Error', result.error)}</div>`;
      return;
    }
    const normResult = normalizeBenchmarkResult(result);
    renderBenchmarkTable(normResult, slot);
  } catch (err) {
    slot.innerHTML = `<div class="error-banner">${formatErrorMessage('Benchmark failed', err)}</div>`;
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = 'Run Benchmark';
    }
  }
}

// Load Querysets
async function loadBenchmarkQuerySets() {
  const select = document.getElementById('select-query-set');
  if (!select) return;

  try {
    const list = await api.getBenchmarkQuerySets();
    if (!list || list.length === 0) {
      select.innerHTML = '<option value="">No question sets found in the eval folder</option>';
      return;
    }
    select.innerHTML = list.map(qs => `
      <option value="${escapeHtml(qs.path)}">${escapeHtml(qs.filename)} (${qs.query_count} queries)</option>
    `).join('');
  } catch (_) {
    select.innerHTML = '<option value="">Could not load question sets</option>';
  }
}

// Load Stats
async function loadStats() {
  const slot = document.getElementById('stats-container-slot');
  const badge = document.getElementById('stats-workspace-badge');

  if (!state.activeWorkspaceId && !state.repoPath) {
    if (badge) badge.textContent = 'No project';
    setStatus('no-repo', 'no project open', 'No project is open. Choose one from the list or add your code.');
    if (slot) {
      slot.innerHTML = `
        <div class="empty-state">
          <strong>No project open</strong><p style="margin-bottom: 12px;">Add your code first, then stats appear here.</p>
          <button id="btn-stats-upload-trigger" class="btn btn-primary btn-sm">Add your code</button>
        </div>
      `;
      document.getElementById('btn-stats-upload-trigger')?.addEventListener('click', () => {
        document.getElementById('btn-open-workspace-modal')?.click();
      });
    }
    return;
  }

  if (badge) {
    badge.textContent = state.activeWorkspace?.name || state.repoPath || 'Workspace';
  }

  let rawStats;
  try {
    rawStats = await api.getStats({
      workspaceId: state.activeWorkspaceId,
      repoPath: state.repoPath,
    });
  } catch (err) {
    setStatus('no-repo', 'no project open', err.message);
    if (slot) {
      slot.innerHTML = `<div class="error-banner">Failed to fetch stats: ${escapeHtml(err.message)}</div>`;
    }
    return;
  }

  if (rawStats && rawStats.error) {
    setStatus('no-repo', 'no project open', rawStats.error.message || 'Repository error');
    if (slot) {
      slot.innerHTML = `<div class="error-banner">${formatErrorMessage('Stats Error', rawStats.error)}</div>`;
    }
    return;
  }

  let stats;
  try {
    stats = normalizeStats(rawStats);
  } catch (normErr) {
    renderUiError(slot, 'Normalizer', normErr, rawStats);
    return;
  }

  const db = stats.database;
  const graph = stats.graph;
  const emb = stats.embeddings;
  const isStale = stats.is_stale;
  const staleReasons = stats.stale_reasons;

  if (isStale) {
    const missingParts = [];
    if (!db.files || db.files === 0) missingParts.push('index missing');
    if (!graph.nodes || graph.nodes === 0) missingParts.push('graph missing');
    else if (staleReasons.some(r => r.includes('stale'))) missingParts.push('graph stale');
    if (!emb.total_items || emb.total_items === 0) missingParts.push('embeddings missing');
    else if (staleReasons.some(r => r.includes('differs'))) missingParts.push('model mismatch');

    const label = missingParts.length > 0 ? `needs update (${missingParts.join(', ')})` : 'needs update';
    setStatus('stale', label, staleReasons.join('; '));
  } else {
    setStatus('ready', 'ready', 'Index, graph, and embeddings are ready and fresh');
  }

  slot.innerHTML = `
    ${isStale ? `
      <div class="needs-context-alert" style="margin-bottom: 20px;">
        <strong>Warning: Workspace index needs update:</strong> ${escapeHtml(staleReasons.join('; ') || 'Files have changed')}
        <br>Click 'Update Index' or 'Rebuild' to refresh.
      </div>
    ` : ''}

    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px;">
      <div class="analysis-card">
        <h4 style="margin-bottom: 12px; font-size: 14px;">Parsed Code (MongoDB)</h4>
        <div style="display: grid; gap: 6px; font-size: 13px;">
          <div><span>Files:</span> <strong>${db.files.toLocaleString()}</strong></div>
          <div><span>Functions:</span> <strong>${db.functions.toLocaleString()}</strong></div>
          <div><span>Classes:</span> <strong>${db.classes.toLocaleString()}</strong></div>
          <div><span>Methods:</span> <strong>${db.methods.toLocaleString()}</strong></div>
          <div><span title="Function calls found in the source">Calls found:</span> <strong>${db.calls.toLocaleString()}</strong></div>
          <div><span title="Files the parser could not read">Unreadable files:</span> <strong>${db.files_with_errors}</strong></div>
        </div>
      </div>

      <div class="analysis-card">
        <h4 style="margin-bottom: 12px; font-size: 14px;">Code Graph (NetworkX)</h4>
        <div style="display: grid; gap: 6px; font-size: 13px;">
          <div><span>Total Nodes:</span> <strong>${graph.nodes.toLocaleString()}</strong></div>
          <div><span>Total Edges:</span> <strong>${graph.edges.toLocaleString()}</strong></div>
          <div><span title="Calls linked to a function inside your project">Calls linked in graph:</span> <strong>${graph.resolved_calls !== null ? graph.resolved_calls : 'N/A'}</strong></div>
          <div><span title="Share of calls the graph could connect">Link rate:</span> <strong>${graph.resolution_rate !== null ? `${(graph.resolution_rate * 100).toFixed(1)}%` : 'N/A'}</strong></div>
        </div>
      </div>

      <div class="analysis-card">
        <h4 style="margin-bottom: 12px; font-size: 14px;">Vector Embeddings (FAISS)</h4>
        <div style="display: grid; gap: 6px; font-size: 13px;">
          <div><span>Indexed Items:</span> <strong>${emb.total_items.toLocaleString()}</strong></div>
          <div><span>Model:</span> <code>${escapeHtml(emb.model_name)}</code></div>
          <div><span>Dimension:</span> <strong>${emb.dimension}</strong></div>
        </div>
      </div>
    </div>
  `;
}

// Workspace change handler triggered by Chat controller
function handleWorkspaceChanged(ws) {
  state.activeWorkspace = ws;
  state.activeWorkspaceId = ws ? ws.id : null;
  state.repoPath = ws ? ws.working_path : null;
  saveActiveWorkspaceId(state.activeWorkspaceId);

  // Sync topbar select
  const topbarSelect = document.getElementById('topbar-select-workspace');
  if (topbarSelect && state.chatController) {
    topbarSelect.innerHTML = '<option value="">(No project open)</option>';
    const list = state.chatController.getWorkspaces();
    list.forEach(w => {
      const opt = document.createElement('option');
      opt.value = w.id;
      const count = (w.file_counts && w.file_counts.python) || 0;
      opt.textContent = `${w.name} (${w.source}) · ${count} py files`;
      topbarSelect.appendChild(opt);
    });
    const addOpt = document.createElement('option');
    addOpt.value = '__open_modal__';
    addOpt.textContent = '+ Open / Upload Code...';
    topbarSelect.appendChild(addOpt);
    topbarSelect.value = state.activeWorkspaceId || '';
  }

  // Update tab headers & badges
  const graphBadge = document.getElementById('graph-workspace-badge');
  if (graphBadge) graphBadge.textContent = ws ? ws.name : 'No project';
  const statsBadge = document.getElementById('stats-workspace-badge');
  if (statsBadge) statsBadge.textContent = ws ? ws.name : 'No project';

  loadStats();
  if (state.activeTab === 'graph' && state.graphView) {
    if (state.activeWorkspaceId || state.repoPath) {
      state.graphView.loadOverview({
        workspaceId: state.activeWorkspaceId,
        workspaceName: ws?.name,
        repoPath: state.repoPath,
      });
    } else {
      state.graphView.clear();
    }
  }
}

// Bootstrap Initialization
if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    // Topbar workspace selector
    const topbarSelect = document.getElementById('topbar-select-workspace');
    if (topbarSelect) {
      topbarSelect.addEventListener('change', () => {
        const val = topbarSelect.value;
        if (val === '__open_modal__') {
          document.getElementById('btn-open-workspace-modal')?.click();
          topbarSelect.value = state.activeWorkspaceId || '';
          return;
        }
        state.chatController?.switchWorkspace(val || null);
      });
    }

    // Budget slider label
    const budgetSlider = document.getElementById('input-budget');
    const budgetVal = document.getElementById('budget-val');
    if (budgetSlider && budgetVal) {
      budgetSlider.addEventListener('input', () => {
        budgetVal.textContent = parseInt(budgetSlider.value, 10).toLocaleString();
      });
    }

    // Initialize Graph View
    const graphView = initGraphView({
      containerId: 'cy-container',
      panelId: 'graph-side-panel',
      onUseAsTarget: (symbol) => {
        document.getElementById('input-query').value = `Optimize ${symbol}`;
        switchTab('ask', graphView);
        handleAnalyze();
      },
    });
    state.graphView = graphView;

    // Tab listeners
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        switchTab(btn.dataset.tab, graphView);
      });
    });

    // Action Buttons
    document.getElementById('btn-analyze')?.addEventListener('click', handleAnalyze);
    document.getElementById('btn-ask')?.addEventListener('click', handleAsk);
    document.getElementById('btn-update')?.addEventListener('click', handleIncrementalUpdate);
    document.getElementById('btn-rebuild')?.addEventListener('click', handleRebuild);
    document.getElementById('btn-run-benchmark')?.addEventListener('click', handleRunBenchmark);
    document.getElementById('btn-refresh-stats')?.addEventListener('click', loadStats);

    // Graph Search & Overview
    document.getElementById('btn-graph-search')?.addEventListener('click', () => {
      const symbol = document.getElementById('input-graph-search').value.trim();
      const hops = parseInt(document.getElementById('select-graph-hops').value, 10) || 2;
      graphView.centerOnSymbol({
        workspaceId: state.activeWorkspaceId,
        workspaceName: state.activeWorkspace?.name,
        repoPath: state.repoPath,
      }, symbol, hops);
    });

    document.getElementById('btn-graph-overview')?.addEventListener('click', () => {
      graphView.loadOverview({
        workspaceId: state.activeWorkspaceId,
        workspaceName: state.activeWorkspace?.name,
        repoPath: state.repoPath,
      });
    });

    // "View in graph" from a chat trace: show the target and the neighbours that were walked
    window.addEventListener('cce:focus-graph', (e) => {
      const ref = e.detail && e.detail.ref;
      if (!ref) return;
      switchTab('graph', null);
      document.getElementById('input-graph-search').value = ref;
      graphView.centerOnSymbol({
        workspaceId: state.activeWorkspaceId,
        workspaceName: state.activeWorkspace?.name,
        repoPath: state.repoPath,
      }, ref, 1);
    });

    // Initial loads
    setupByokControls();
    loadProviders();
    loadBenchmarkQuerySets();
    loadStats();
    state.chatController = initChat(state, setStatus, handleWorkspaceChanged);
  });
}

const PROVIDER_KEY_URLS = {
  gemini: 'https://aistudio.google.com/app/apikey',
  claude: 'https://console.anthropic.com/',
  groq: 'https://console.groq.com/keys',
  openrouter: 'https://openrouter.ai/keys',
  openai: 'https://platform.openai.com/api-keys',
  'openai-compatible': 'https://platform.openai.com/api-keys',
};

function setupByokControls() {
  const keyInput = document.getElementById('input-api-key');
  const toggleBtn = document.getElementById('btn-toggle-key-visibility');
  const validateBtn = document.getElementById('btn-validate-key');
  const clearBtn = document.getElementById('btn-clear-key');
  const rememberCheckbox = document.getElementById('check-remember-key');
  const rememberWarning = document.getElementById('remember-warning');
  const httpWarning = document.getElementById('http-insecure-warning');

  if (
    typeof window !== 'undefined' &&
    window.location.protocol === 'http:' &&
    !['localhost', '127.0.0.1', '::1'].includes(window.location.hostname)
  ) {
    if (httpWarning) httpWarning.style.display = 'inline-block';
  }

  if (toggleBtn && keyInput) {
    toggleBtn.addEventListener('click', () => {
      const isPassword = keyInput.type === 'password';
      keyInput.type = isPassword ? 'text' : 'password';
      toggleBtn.title = isPassword ? 'Hide the key' : 'Show the key';
      toggleBtn.setAttribute('aria-pressed', String(isPassword));
    });
  }

  if (rememberCheckbox) {
    rememberCheckbox.addEventListener('change', () => {
      if (rememberWarning) {
        rememberWarning.style.display = rememberCheckbox.checked ? 'block' : 'none';
      }
      const prov = state.selectedProvider;
      if (prov && keyManager.hasKey(prov)) {
        const rawKey = keyManager.getKey(prov);
        keyManager.setKey(prov, rawKey, rememberCheckbox.checked);
      }
    });
  }

  if (validateBtn) {
    validateBtn.addEventListener('click', () => {
      handleValidateKey(true);
    });
  }

  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      const prov = state.selectedProvider;
      if (!prov) return;
      keyManager.clearKey(prov);
      if (keyInput) keyInput.value = '';
      setKeyStatus('', '');
      updateAskButtonState();
      const pObj = state.providers.find(p => p.name === prov);
      updateModelForProvider(pObj);
    });
  }

  let debounceTimer = null;
  if (keyInput) {
    keyInput.addEventListener('input', () => {
      const val = keyInput.value.trim();
      if (val && !val.includes('••••')) {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          handleValidateKey(false);
        }, 800);
      }
    });

    keyInput.addEventListener('blur', () => {
      const val = keyInput.value.trim();
      if (val && !val.includes('••••')) {
        clearTimeout(debounceTimer);
        handleValidateKey(false);
      }
    });
  }
}

async function handleValidateKey(isManual = false) {
  const prov = state.selectedProvider;
  if (!prov || prov === 'ollama') return;

  const keyInput = document.getElementById('input-api-key');
  const rememberCheckbox = document.getElementById('check-remember-key');
  const validateBtn = document.getElementById('btn-validate-key');
  if (!keyInput) return;

  let enteredKey = keyInput.value.trim();
  if (enteredKey.includes('••••')) {
    enteredKey = keyManager.getKey(prov);
  }

  if (!enteredKey) {
    if (isManual) {
      setKeyStatus('invalid', 'Please enter an API key');
    }
    return;
  }

  setKeyStatus('validating', 'Validating key...');
  if (validateBtn) validateBtn.disabled = true;

  try {
    const inputModel = document.getElementById('input-model');
    const targetModel = inputModel ? inputModel.value.trim() : null;
    const res = await api.validateKey(prov, enteredKey, targetModel);

    if (res && res.valid) {
      keyManager.setKey(prov, enteredKey, rememberCheckbox ? rememberCheckbox.checked : false);
      keyInput.value = keyManager.maskKey(enteredKey);
      setKeyStatus('valid', '✓ Key valid');

      if (Array.isArray(res.models) && res.models.length > 0) {
        populateModelList(res.models);
        if (!targetModel || !res.models.includes(targetModel)) {
          const defaultM = res.models[0];
          if (inputModel) inputModel.value = defaultM;
          state.selectedModel = defaultM;
          saveLastModel(defaultM);
        }
      }

      updateAskButtonState();
      if (state.chatController) state.chatController.updateActiveModelBadge();
    } else {
      const msg = res && res.message ? res.message : 'Invalid API key';
      setKeyStatus('invalid', msg);
      updateAskButtonState();
    }
  } catch (err) {
    const msg = err && err.message ? err.message : 'Validation failed';
    setKeyStatus('invalid', msg);
    updateAskButtonState();
  } finally {
    if (validateBtn) validateBtn.disabled = false;
  }
}

function setKeyStatus(type, message) {
  const statusEl = document.getElementById('key-status-msg');
  if (!statusEl) return;
  statusEl.className = `key-status-msg ${type}`;
  statusEl.textContent = message;
}

function populateModelList(models) {
  const datalist = document.getElementById('model-suggestions');
  if (!datalist || !Array.isArray(models)) return;
  datalist.innerHTML = '';
  models.forEach(m => {
    const opt = document.createElement('option');
    opt.value = m;
    datalist.appendChild(opt);
  });
}

function updateAskButtonState() {
  const btnAsk = document.getElementById('btn-ask');
  const noticeEl = document.getElementById('ask-btn-notice');
  const prov = state.selectedProvider;
  if (!btnAsk) return;

  const pObj = state.providers.find(p => p.name === prov);
  const isCloud = prov && prov !== 'ollama';
  const hasUserKey = keyManager.hasKey(prov);
  const hasServerKey = Boolean(pObj && pObj.configured);

  if (isCloud && !hasUserKey && !hasServerKey) {
    btnAsk.disabled = true;
    const pNameCapitalized = prov.charAt(0).toUpperCase() + prov.slice(1);
    if (noticeEl) {
      noticeEl.textContent = `Enter your ${pNameCapitalized} API key above, or switch to Ollama which needs no key`;
    }
  } else {
    btnAsk.disabled = false;
    if (noticeEl) {
      noticeEl.textContent = '';
    }
  }
}

async function loadProviders() {
  const selProv = document.getElementById('select-provider');
  const hintEl = document.getElementById('provider-setup-hint');
  try {
    const list = await api.getProviders();
    state.providers = Array.isArray(list) ? list : [];
    if (state.providers.length === 0) {
      if (selProv) {
        selProv.innerHTML = '<option value="">No providers available</option>';
      }
      if (hintEl) {
        hintEl.textContent = 'No AI providers available on server.';
      }
      return;
    }
    populateProviderControls();
  } catch (err) {
    console.error('Failed to load providers list:', err);
    if (selProv) {
      selProv.innerHTML = `<option value="">Error: ${escapeHtml(err.message || 'Failed to load')}</option>`;
      selProv.style.borderColor = 'var(--danger-color)';
    }
    if (hintEl) {
      hintEl.textContent = `Error loading providers: ${err.message}`;
      hintEl.style.color = 'var(--danger-text)';
    }
  }
}

function populateProviderControls() {
  const selProv = document.getElementById('select-provider');
  const inputModel = document.getElementById('input-model');
  if (!selProv || !inputModel) return;

  selProv.innerHTML = '';
  const savedProvider = getSavedProvider();
  const savedModel = getSavedModel();

  state.providers.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.name;
    const cfgTag = p.configured ? ' (server key)' : '';
    opt.textContent = `${p.name}${cfgTag}`;
    selProv.appendChild(opt);
  });

  let activeP = state.providers.find(p => p.name === savedProvider);
  if (!activeP) {
    activeP = state.providers.find(p => p.configured) || state.providers[0];
  }

  if (activeP) {
    selProv.value = activeP.name;
    state.selectedProvider = activeP.name;
    updateModelForProvider(activeP, savedModel);
  }

  selProv.onchange = () => {
    const prov = state.providers.find(p => p.name === selProv.value);
    state.selectedProvider = selProv.value;
    saveLastProvider(selProv.value);
    updateModelForProvider(prov);
  };

  inputModel.oninput = () => {
    state.selectedModel = inputModel.value.trim();
    saveLastModel(state.selectedModel);
    if (state.chatController) state.chatController.updateActiveModelBadge();
  };
}

function updateModelForProvider(prov, preferredModel = null) {
  const inputModel = document.getElementById('input-model');
  const hintEl = document.getElementById('provider-setup-hint');
  const datalist = document.getElementById('model-suggestions');
  const keyContainer = document.getElementById('key-input-container');
  const keyInput = document.getElementById('input-api-key');
  const rememberCheckbox = document.getElementById('check-remember-key');
  const rememberWarning = document.getElementById('remember-warning');
  const linkGetKey = document.getElementById('link-get-key');

  if (!prov || !inputModel) return;

  const isOllama = prov.name === 'ollama';
  if (keyContainer) {
    keyContainer.style.display = isOllama ? 'none' : 'flex';
  }

  if (linkGetKey) {
    const url = PROVIDER_KEY_URLS[prov.name];
    if (url) {
      linkGetKey.href = url;
      linkGetKey.style.display = 'inline';
    } else {
      linkGetKey.style.display = 'none';
    }
  }

  if (!isOllama) {
    const isRemembered = keyManager.isRemembered(prov.name);
    if (rememberCheckbox) rememberCheckbox.checked = isRemembered;
    if (rememberWarning) rememberWarning.style.display = isRemembered ? 'block' : 'none';

    if (keyManager.hasKey(prov.name)) {
      const rawKey = keyManager.getKey(prov.name);
      if (keyInput) keyInput.value = keyManager.maskKey(rawKey);
      setKeyStatus('valid', '✓ Key saved');

      api.getProviderModels(prov.name, rawKey).then(res => {
        if (res && Array.isArray(res.models) && res.models.length > 0) {
          populateModelList(res.models);
        }
      }).catch(() => {});
    } else {
      if (keyInput) keyInput.value = '';
      if (prov.configured) {
        setKeyStatus('', 'Server key active (enter key to override)');
      } else {
        setKeyStatus('', '');
      }
    }
  }

  updateAskButtonState();

  if (datalist) {
    datalist.innerHTML = '';
    if (Array.isArray(prov.available_models)) {
      prov.available_models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m;
        datalist.appendChild(opt);
      });
    }
  }

  const modelVal = preferredModel || prov.default_model || (prov.available_models && prov.available_models[0]) || '';
  inputModel.value = modelVal;
  state.selectedModel = modelVal;

  if (hintEl) {
    if (!prov.configured && prov.setup_hint && !keyManager.hasKey(prov.name)) {
      hintEl.textContent = prov.setup_hint;
      hintEl.title = prov.setup_hint;
    } else {
      hintEl.textContent = '';
      hintEl.title = '';
    }
  }

  if (state.chatController) {
    state.chatController.updateActiveModelBadge();
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
