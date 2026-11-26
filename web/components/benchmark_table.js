/**
 * Component to render benchmark comparison tables and bar charts.
 * Reads from normalized benchmark view models.
 */

import { normalizeBenchmarkResult } from '../normalizer.js';

export function renderBenchmarkTable(benchmarkData, container) {
  if (!benchmarkData) {
    container.innerHTML = '<div class="empty-state">No benchmark data available. Run a benchmark to view results.</div>';
    return;
  }

  const vm = benchmarkData.raw !== undefined ? benchmarkData : normalizeBenchmarkResult(benchmarkData);
  const summaries = vm.summaries;

  if (!summaries || summaries.length === 0) {
    container.innerHTML = '<div class="empty-state">No benchmark data available. Run a benchmark to view results.</div>';
    return;
  }

  const graphSummary = summaries.find(s => s.strategy_name === 'graph');
  const maxMeanTokens = Math.max(...summaries.map(s => s.mean_context_tokens || 1));

  // 1. Bar Chart Comparison (CSS-based)
  const barsHtml = summaries.map(s => {
    const isGraph = s.strategy_name === 'graph';
    const pct = Math.max(2, Math.min(100, (s.mean_context_tokens / maxMeanTokens) * 100));
    return `
      <div class="bm-bar-row ${isGraph ? 'bm-bar-graph' : ''}">
        <div class="bm-bar-name">
          <code>${escapeHtml(s.strategy_name)}</code>
          ${isGraph ? '<span class="badge badge-target">OURS</span>' : ''}
        </div>
        <div class="bm-bar-track">
          <div class="bm-bar-fill ${isGraph ? 'fill-accent' : 'fill-muted'}" style="width: ${pct}%"></div>
        </div>
        <div class="bm-bar-val">
          ${Math.round(s.mean_context_tokens).toLocaleString()} tokens
        </div>
      </div>
    `;
  }).join('');

  // 2. Tabular Comparison
  const rowsHtml = summaries.map(s => {
    const isGraph = s.strategy_name === 'graph';
    let redMean = 'BASELINE';
    let redAgg = 'BASELINE';

    if (!isGraph && graphSummary) {
      redMean = `${s.mean_of_reductions_pct >= 0 ? '+' : ''}${s.mean_of_reductions_pct.toFixed(1)}%`;
      redAgg = `${s.reduction_of_means_pct >= 0 ? '+' : ''}${s.reduction_of_means_pct.toFixed(1)}%`;
    }

    const covStr = s.mean_coverage_pct !== null && s.mean_coverage_pct !== undefined
      ? `${s.mean_coverage_pct.toFixed(1)}%`
      : 'N/A';

    return `
      <tr class="${isGraph ? 'row-target' : ''}">
        <td><strong><code>${escapeHtml(s.strategy_name)}</code></strong></td>
        <td>${s.mean_files_inspected.toFixed(1)}</td>
        <td><strong>${Math.round(s.mean_context_tokens).toLocaleString()}</strong></td>
        <td>${Math.round(s.median_context_tokens).toLocaleString()}</td>
        <td>${s.mean_build_latency_seconds.toFixed(3)}s</td>
        <td><span class="badge ${s.mean_coverage_pct >= 50 ? 'badge-success' : 'badge-subtle'}">${covStr}</span></td>
        <td>${s.over_limit_count}</td>
        <td class="col-red"><strong>${redMean}</strong></td>
        <td class="col-red">${redAgg}</td>
      </tr>
    `;
  }).join('');

  container.innerHTML = `
    <div class="bm-card">
      <div class="bm-header">
        <div>
          <h3 class="bm-title">Context Tokens by Strategy</h3>
          <div class="bm-meta">
            <span>Repo: <strong>${vm.repo_file_count} files</strong> (${vm.repo_total_tokens.toLocaleString()} source tokens)</span>
            <span>&bull;</span>
            <span>Counter: <strong>${escapeHtml(vm.counter_used)} (approx)</strong></span>
            <span>&bull;</span>
            <span>Budget: <strong>${vm.budget} tokens</strong></span>
          </div>
        </div>
      </div>

      <div class="bm-bars-container">
        ${barsHtml}
      </div>

      <div class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th>Strategy</th>
              <th>Files</th>
              <th>Mean Tokens</th>
              <th>Med Tokens</th>
              <th>Latency</th>
              <th>Coverage</th>
              <th>Over Limit</th>
              <th>Reduct (Mean)</th>
              <th>Reduct (Agg)</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      </div>
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
