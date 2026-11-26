/**
 * Component to render ranked hybrid retrieval results.
 * Reads strictly from normalized view models to guarantee safe rendering.
 */

import { normalizeRetrievalResult } from '../normalizer.js';

export function renderResultsTable(retrievalData, container) {
  if (!retrievalData) {
    container.innerHTML = '<div class="empty-state">No retrieval candidates found for query.</div>';
    return;
  }

  // Normalize if not already normalized
  const vm = retrievalData.candidates && retrievalData.candidates[0]?.is_target !== undefined
    ? retrievalData
    : normalizeRetrievalResult(retrievalData);

  if (!vm.candidates || vm.candidates.length === 0) {
    container.innerHTML = '<div class="empty-state">No retrieval candidates found for query.</div>';
    return;
  }

  const targetCand = vm.target_candidate;

  let headerHtml = `
    <div class="analysis-card">
      <div class="analysis-meta">
        <div><strong>Query:</strong> <span class="badge badge-code">${escapeHtml(vm.query)}</span></div>
        <div><strong>Intent:</strong> <span class="badge badge-subtle">${escapeHtml(vm.intent.toUpperCase())}</span></div>
        ${vm.identifiers.length > 0 ? `<div><strong>Identifiers:</strong> ${vm.identifiers.map(id => `<span class="badge badge-code">${escapeHtml(id)}</span>`).join(' ')}</div>` : ''}
        ${vm.scope ? `<div><strong>Scope:</strong> <span class="badge badge-subtle">${escapeHtml(vm.scope)}</span></div>` : ''}
      </div>
      ${vm.target ? `
        <div class="target-banner">
          <span class="target-title">Target:</span>
          <span class="target-name">${escapeHtml(vm.target)}</span>
          <span class="badge badge-target">PRIMARY TARGET</span>
          ${targetCand ? `<span class="target-loc">${escapeHtml(targetCand.file)}${targetCand.start_line ? `:${targetCand.start_line}-${targetCand.end_line}` : ''}</span>` : ''}
          ${vm.target_confidence ? `<span class="target-conf">Confidence: ${escapeHtml(vm.target_confidence)}</span>` : ''}
        </div>
      ` : ''}
    </div>
  `;

  let rowsHtml = vm.candidates.map((cand, idx) => {
    const lineRange = cand.start_line ? `:${cand.start_line}-${cand.end_line}` : '';

    return `
      <tr class="${cand.is_target ? 'row-target' : ''}">
        <td class="col-rank">#${idx + 1}</td>
        <td class="col-entity">
          <div class="entity-qname">
            <strong>${escapeHtml(cand.ref)}</strong>
            ${cand.entity_type ? `<span class="badge badge-subtle">${escapeHtml(cand.entity_type)}</span>` : ''}
            ${cand.is_target ? '<span class="badge badge-target">TARGET</span>' : ''}
            ${cand.is_graph_discovery ? '<span class="badge badge-graph-discovery" title="Discovered exclusively through graph traversal">Graph Discovery</span>' : ''}
            ${cand.match_type && cand.match_type !== 'graph' ? `<span class="badge badge-code" title="Match Mechanism">${escapeHtml(cand.match_type)}</span>` : ''}
          </div>
          <div class="entity-file">${escapeHtml(cand.file)}${lineRange}</div>
        </td>
        <td class="col-score">
          <div class="score-pill">${cand.score.toFixed(3)}</div>
        </td>
        <td class="col-breakdown">
          <div class="score-bars">
            <div class="bar-row" title="Semantic: ${cand.sem_score.toFixed(3)}">
              <span class="bar-label">Sem</span>
              <div class="bar-track"><div class="bar-fill bar-sem" style="width: ${(cand.sem_score * 100).toFixed(1)}%"></div></div>
            </div>
            <div class="bar-row" title="Keyword / Exact: ${cand.kw_score.toFixed(3)}">
              <span class="bar-label">KW</span>
              <div class="bar-track"><div class="bar-fill bar-kw" style="width: ${(cand.kw_score * 100).toFixed(1)}%"></div></div>
            </div>
            <div class="bar-row" title="Graph: ${cand.gr_score.toFixed(3)}">
              <span class="bar-label">Graph</span>
              <div class="bar-track"><div class="bar-fill bar-gr" style="width: ${(cand.gr_score * 100).toFixed(1)}%"></div></div>
            </div>
          </div>
        </td>
      </tr>
    `;
  }).join('');

  container.innerHTML = `
    ${headerHtml}
    <div class="table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Entity & Location</th>
            <th>Score</th>
            <th>Signals Breakdown</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
        </tbody>
      </table>
    </div>
  `;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
