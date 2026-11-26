/**
 * Component explaining HOW a request used the code graph and what it cost.
 * Reads from the normalized trace view model (normalizeTrace) and shows:
 *   1. Token savings versus pasting the whole repo / whole files
 *   2. Which retrieval signals ran (semantic, keyword, graph) and what each found
 *   3. The graph edges that were walked from the target
 *   4. Ordered operations with timings
 *   5. What was included, compressed, or pruned
 */

import { normalizeTrace } from '../normalizer.js';

const OP_LABELS = {
  intent_detection: 'Detect intent',
  target_identification: 'Identify target symbol',
  semantic_search: 'Semantic search',
  keyword_search: 'Keyword search',
  ast_call_expansion: 'AST call expansion',
  graph_expansion: 'Code graph expansion',
  context_packing: 'Pack & compress context',
  token_accounting: 'Token accounting',
  llm_completion: 'LLM completion',
};

const SIGNAL_LABELS = {
  semantic: 'Semantic',
  keyword: 'Keyword',
  graph: 'Graph',
};

const HIT_LABELS = {
  semantic: 'semantic',
  keyword: 'keyword',
  ast_call: 'AST call',
  graph_callee: 'graph callee',
  graph_caller: 'graph caller',
  exact_target: 'exact match',
};

export function renderTracePanel(traceData, container) {
  if (!container) return;
  const vm = traceData && traceData.operations !== undefined ? traceData : normalizeTrace(traceData);
  if (!vm) {
    container.innerHTML = '<div class="empty-state">No trace recorded for this request.</div>';
    return;
  }
  container.innerHTML = renderTraceHtml(vm);
}

export function renderTraceHtml(vm) {
  if (!vm) return '<div class="empty-state">No trace recorded for this request.</div>';
  return `
    <div class="trace-panel">
      ${renderSavingsHero(vm)}
      ${renderTargetSummary(vm)}
      ${renderSignals(vm)}
      ${renderGraphWalk(vm)}
      ${renderOperations(vm)}
      ${renderSelection(vm)}
      ${vm.warnings.length ? `<div class="trace-warnings">${vm.warnings.map(w => `<div>&#9888; ${escapeHtml(w)}</div>`).join('')}</div>` : ''}
    </div>
  `;
}

/** One-line digest shown under a chat message; links to the full panel. */
export function renderTraceSummaryLine(vm) {
  if (!vm || !vm.savings) return '';
  const s = vm.savings;
  const sent = s.provider_input_tokens ?? s.prompt_tokens;
  const edges = vm.graph_edges.length;
  const verdict = s.saved_vs_full_repo_pct > 0
    ? `${s.saved_vs_full_repo_pct.toFixed(1)}% fewer tokens than the full repo`
    : 'repo is small enough that no savings apply';
  return `
    <button type="button" class="trace-summary-line" title="Open the full trace">
      <span class="trace-summary-dot"></span>
      <span><strong>${sent.toLocaleString()}</strong> tokens sent &middot; ${escapeHtml(verdict)}
      &middot; ${edges} graph edge${edges === 1 ? '' : 's'} walked</span>
      <span class="trace-summary-link">How? &rarr;</span>
    </button>
  `;
}

function renderSavingsHero(vm) {
  const s = vm.savings;
  if (!s) return '';

  const sent = s.provider_input_tokens ?? s.prompt_tokens;
  const sentLabel = s.provider_input_tokens !== null ? 'billed by provider' : 'sent to model';
  const max = Math.max(s.naive_full_repo_tokens, s.whole_files_tokens, sent, 1);
  const pct = n => Math.max(1.5, Math.min(100, (n / max) * 100));
  const positive = s.saved_vs_full_repo_pct > 0;
  const headline = positive ? `${s.saved_vs_full_repo_pct.toFixed(1)}%` : '0%';
  const factor = positive && s.reduction_factor > 0 ? `${s.reduction_factor.toLocaleString()}&times; smaller` : 'no reduction';

  const wholeRow = s.context_files > 0
    ? barRow(`Whole files used (${s.context_files})`, s.whole_files_tokens, pct(s.whole_files_tokens), 'bar-mid')
    : '';

  return `
    <section class="trace-section trace-hero ${positive ? 'trace-hero-positive' : 'trace-hero-neutral'}">
      <div class="trace-hero-head">
        <div>
          <div class="trace-hero-label">Token savings vs. sending the full repo</div>
          <div class="trace-hero-value">${headline}<span class="trace-hero-factor">${factor}</span></div>
        </div>
        ${s.is_estimate ? '<span class="trace-badge trace-badge-warn" title="Exact tokenizer unavailable; counts use ~4 chars per token">estimated counts</span>' : ''}
      </div>
      <div class="trace-bars">
        ${barRow(`Naive: all ${s.repo_files} file${s.repo_files === 1 ? '' : 's'} pasted`, s.naive_full_repo_tokens, pct(s.naive_full_repo_tokens), 'bar-naive')}
        ${wholeRow}
        ${barRow(`This request (${sentLabel})`, sent, pct(sent), 'bar-actual')}
      </div>
      <div class="trace-hero-foot">
        <span>Context ${s.context_tokens.toLocaleString()} tokens from ${s.context_files} of ${s.repo_files} files</span>
        ${s.compression_saved_tokens > 0 ? `<span>Compression removed ${s.compression_saved_tokens.toLocaleString()} tokens</span>` : ''}
        <span>Counter: ${escapeHtml(s.counter)}</span>
      </div>
      ${!positive && s.repo_files > 0 ? '<div class="trace-note">On a repository this small, the prompt scaffolding costs about as much as the code itself. Savings grow with repository size.</div>' : ''}
    </section>
  `;
}

function barRow(label, tokens, widthPct, cls) {
  return `
    <div class="trace-bar-row">
      <div class="trace-bar-label"><span>${escapeHtml(label)}</span><strong>${tokens.toLocaleString()}</strong></div>
      <div class="trace-bar-track"><div class="trace-bar-fill ${cls}" style="width:${widthPct.toFixed(1)}%"></div></div>
    </div>
  `;
}

function renderTargetSummary(vm) {
  if (!vm.target) {
    return `
      <section class="trace-section">
        <h5 class="trace-heading">Target</h5>
        <div class="trace-muted">No single target symbol was identified; the top-ranked candidates were used.</div>
      </section>`;
  }
  return `
    <section class="trace-section">
      <h5 class="trace-heading">Target</h5>
      <div class="trace-target-row">
        <code class="trace-ref">${escapeHtml(vm.target)}</code>
        <span class="trace-badge">${escapeHtml(vm.target_confidence)} confidence</span>
        <span class="trace-badge trace-badge-info">${escapeHtml(vm.intent)}</span>
        <button type="button" class="btn btn-ghost btn-xs trace-focus-graph" data-ref="${escapeHtml(vm.target)}">View in graph</button>
      </div>
      ${vm.target_reason ? `<div class="trace-muted">${escapeHtml(vm.target_reason)}</div>` : ''}
    </section>
  `;
}

function renderSignals(vm) {
  const byName = {};
  for (const h of vm.hits) byName[h.signal] = (byName[h.signal] || 0) + 1;
  const hitCount = {
    semantic: byName.semantic || 0,
    keyword: byName.keyword || 0,
    graph: (byName.graph_callee || 0) + (byName.graph_caller || 0),
  };

  const chips = vm.signals.map(sig => {
    const label = SIGNAL_LABELS[sig.name] || sig.name;
    const detail = sig.state === 'used'
      ? `${hitCount[sig.name] ?? 0} hit${hitCount[sig.name] === 1 ? '' : 's'}`
      : sig.detail || sig.state;
    return `
      <div class="signal-chip signal-${sig.state}">
        <span class="signal-name">${escapeHtml(label)}</span>
        <span class="signal-state">${escapeHtml(detail)}</span>
      </div>`;
  }).join('');

  const topHits = vm.hits
    .filter(h => h.signal !== 'exact_target')
    .sort((a, b) => b.score - a.score)
    .slice(0, 8)
    .map(h => `
      <tr>
        <td><code class="trace-ref">${escapeHtml(shortRef(h.ref))}</code></td>
        <td><span class="hit-signal hit-${escapeHtml(h.signal)}">${escapeHtml(HIT_LABELS[h.signal] || h.signal)}</span></td>
        <td class="num">${h.score.toFixed(2)}</td>
      </tr>`).join('');

  return `
    <section class="trace-section">
      <h5 class="trace-heading">Retrieval signals</h5>
      <div class="signal-row">${chips || '<span class="trace-muted">No signal data.</span>'}</div>
      <div class="trace-muted">${vm.candidates_considered} candidates considered &rarr; ${vm.candidates_returned} ranked for packing</div>
      ${topHits ? `
        <details class="trace-details">
          <summary>Top hits by signal</summary>
          <table class="trace-table"><thead><tr><th>Symbol</th><th>Signal</th><th class="num">Score</th></tr></thead><tbody>${topHits}</tbody></table>
        </details>` : ''}
    </section>
  `;
}

function renderGraphWalk(vm) {
  const graphSignal = vm.signals.find(s => s.name === 'graph');
  const total = vm.graph_nodes ? `${vm.graph_nodes.toLocaleString()} nodes / ${vm.graph_edges_total.toLocaleString()} edges in graph` : '';

  if (!vm.graph_edges.length) {
    const why = graphSignal && graphSignal.state !== 'used'
      ? graphSignal.detail
      : 'The target has no resolved callers or callees.';
    return `
      <section class="trace-section">
        <h5 class="trace-heading">Code graph walk</h5>
        <div class="trace-muted">No edges walked: ${escapeHtml(why)}</div>
      </section>`;
  }

  const callees = vm.graph_edges.filter(e => e.direction === 'callee');
  const callers = vm.graph_edges.filter(e => e.direction === 'caller');
  const row = e => {
    const other = e.direction === 'callee' ? e.target : e.source;
    return `
      <li class="edge-row">
        <span class="edge-rel">${escapeHtml(e.rel)}</span>
        <code class="trace-ref">${escapeHtml(shortRef(other))}</code>
        <span class="edge-outcome edge-${escapeHtml(e.outcome)}">${escapeHtml(e.outcome.replace('_', ' '))}</span>
        ${e.confidence ? `<span class="trace-muted">${escapeHtml(e.confidence)}</span>` : ''}
      </li>`;
  };

  return `
    <section class="trace-section">
      <h5 class="trace-heading">Code graph walk <span class="trace-heading-meta">${escapeHtml(total)}</span></h5>
      ${callees.length ? `<div class="edge-group-title">Calls (outgoing)</div><ul class="edge-list">${callees.map(row).join('')}</ul>` : ''}
      ${callers.length ? `<div class="edge-group-title">Called by (incoming)</div><ul class="edge-list">${callers.map(row).join('')}</ul>` : ''}
      <div class="trace-muted">Only these neighbours of the target were pulled in, instead of every file.</div>
    </section>
  `;
}

function renderOperations(vm) {
  if (!vm.operations.length) return '';
  const rows = vm.operations.map(op => {
    const icon = op.status === 'ran' ? '&#10003;' : (op.status === 'failed' ? '&#10007;' : '&ndash;');
    return `
      <li class="op-row op-${escapeHtml(op.status)}">
        <span class="op-step">${op.step}</span>
        <span class="op-icon">${icon}</span>
        <div class="op-body">
          <div class="op-name">${escapeHtml(OP_LABELS[op.name] || op.name)}${op.status !== 'ran' ? ` <em>(${escapeHtml(op.status)})</em>` : ''}</div>
          <div class="op-detail">${escapeHtml(op.detail)}</div>
        </div>
        <span class="op-time">${formatSeconds(op.seconds)}</span>
      </li>`;
  }).join('');
  return `
    <section class="trace-section">
      <h5 class="trace-heading">Operations</h5>
      <ol class="op-list">${rows}</ol>
    </section>
  `;
}

function renderSelection(vm) {
  const sel = vm.selection;
  if (!sel) return '';

  const rows = sel.included.map(it => {
    const saved = it.original_tokens > 0 ? it.original_tokens - it.tokens : 0;
    return `
      <tr>
        <td><code class="trace-ref" title="${escapeHtml(it.file)}:${escapeHtml(it.lines)}">${escapeHtml(shortRef(it.ref))}</code></td>
        <td><span class="role-pill role-${escapeHtml(it.role)}">${escapeHtml(it.role)}</span></td>
        <td><span class="level-pill level-${escapeHtml(it.level)}">${escapeHtml(it.level)}</span></td>
        <td class="num">${it.tokens.toLocaleString()}${saved > 0 ? `<span class="trace-saved"> -${saved.toLocaleString()}</span>` : ''}</td>
      </tr>`;
  }).join('');

  const omitted = sel.omitted.length
    ? `<details class="trace-details">
        <summary>${sel.omitted.length} pruned candidate${sel.omitted.length === 1 ? '' : 's'}</summary>
        <ul class="omit-list">${sel.omitted.slice(0, 40).map(o => `<li><code class="trace-ref">${escapeHtml(shortRef(o.ref))}</code> <span class="trace-muted">${escapeHtml(humanizeReason(o.reason))}</span></li>`).join('')}</ul>
      </details>`
    : '<div class="trace-muted">Nothing was pruned.</div>';

  return `
    <section class="trace-section">
      <h5 class="trace-heading">Selected context <span class="trace-heading-meta">${sel.included.length} of ${sel.candidates_in} candidates${sel.downgraded_count ? `, ${sel.downgraded_count} downgraded` : ''}</span></h5>
      <table class="trace-table">
        <thead><tr><th>Symbol</th><th>Role</th><th>Detail</th><th class="num">Tokens</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
      ${omitted}
    </section>
  `;
}

function humanizeReason(reason) {
  const map = {
    budget_exceeded: 'did not fit the token budget',
    max_items_limit: 'item limit reached',
    post_verification_budget_cutoff: 'dropped after verifying the rendered size',
  };
  return map[reason] || reason;
}

function shortRef(ref) {
  const parts = String(ref).split('.');
  return parts.length > 3 ? `…${parts.slice(-3).join('.')}` : String(ref);
}

function formatSeconds(sec) {
  if (!sec) return '<1 ms';
  return sec < 1 ? `${Math.max(1, Math.round(sec * 1000))} ms` : `${sec.toFixed(2)} s`;
}

function escapeHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
