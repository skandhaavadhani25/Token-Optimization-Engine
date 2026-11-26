/**
 * Component to render token utilization and baseline reduction metrics.
 * Reads from normalized context and benchmark view models.
 */

import { normalizeContext, normalizeBenchmarkOne } from '../normalizer.js';

export function renderTokenPanel(contextData, benchmarkOneData, container) {
  if (!contextData) {
    container.innerHTML = '';
    return;
  }

  const vm = contextData.level_counts !== undefined ? contextData : normalizeContext(contextData);
  const bmOne = benchmarkOneData ? (benchmarkOneData.raw !== undefined ? benchmarkOneData : normalizeBenchmarkOne(benchmarkOneData)) : null;

  const tokens = vm.tokens;
  const budget = vm.budget_total;
  const utilization = Math.min(100, Math.max(0, vm.utilization * 100));
  const levels = vm.level_counts;

  // Baseline comparison badge
  let reductionHtml = '';
  if (bmOne) {
    const red = bmOne.reduction_pct;
    const isPositive = red >= 0;
    const wfTokens = bmOne.whole_file_tokens;
    reductionHtml = `
      <div class="savings-card ${isPositive ? 'savings-positive' : 'savings-neutral'}">
        <div class="savings-label">Reduction vs Single-File Baseline</div>
        <div class="savings-value">${isPositive ? '+' : ''}${red.toFixed(1)}%</div>
        <div class="savings-subtext">
          <span>Target File: <strong>${wfTokens.toLocaleString()}</strong> tokens</span>
          <span>&rarr;</span>
          <span>Graph Context: <strong>${tokens.toLocaleString()}</strong> tokens</span>
        </div>
        <div class="savings-counter-note">Measured via ${escapeHtml(bmOne.counter)} (approx)</div>
      </div>
    `;
  } else {
    reductionHtml = `
      <div class="savings-card savings-loading">
        <div class="savings-label">Single-File Baseline</div>
        <div class="savings-subtext">Computing comparison...</div>
      </div>
    `;
  }

  container.innerHTML = `
    <div class="token-panel-card">
      <div class="token-header">
        <div class="token-title">Prompt Context Budget</div>
        <div class="token-utilization-badge">${utilization.toFixed(1)}% Used</div>
      </div>

      <div class="token-gauge">
        <div class="token-gauge-bar">
          <div class="token-gauge-fill" style="width: ${utilization}%"></div>
        </div>
        <div class="token-gauge-labels">
          <span><strong>${tokens.toLocaleString()}</strong> tokens used</span>
          <span><strong>${budget.toLocaleString()}</strong> token limit</span>
        </div>
      </div>

      <div class="detail-levels-strip">
        <div class="level-chip chip-full" title="Complete source code">
          <span class="chip-count">${levels.FULL}</span>
          <span class="chip-name">FULL</span>
        </div>
        <div class="level-chip chip-trimmed" title="Source trimmed of comments & literals">
          <span class="chip-count">${levels.TRIMMED}</span>
          <span class="chip-name">TRIMMED</span>
        </div>
        <div class="level-chip chip-signature" title="Signature + docstring + call summary">
          <span class="chip-count">${levels.SIGNATURE}</span>
          <span class="chip-name">SIGNATURE</span>
        </div>
        <div class="level-chip chip-stub" title="Single-line stub definition">
          <span class="chip-count">${levels.STUB}</span>
          <span class="chip-name">STUB</span>
        </div>
        <div class="level-chip chip-ref" title="Qualified name and location">
          <span class="chip-count">${levels.REFERENCE}</span>
          <span class="chip-name">REF</span>
        </div>
      </div>

      ${reductionHtml}
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
