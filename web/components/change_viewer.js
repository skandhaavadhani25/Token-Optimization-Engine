/**
 * Component to render AI completion results, proposed code replacements,
 * syntax validation flags, risks, and interactive NEEDS_CONTEXT chips.
 * Reads from normalized ask view models.
 */

import { normalizeAskResult } from '../normalizer.js';

export function renderChangeViewer(askData, container, onPinEntity) {
  if (!askData) {
    container.innerHTML = '';
    return;
  }

  const vm = askData.raw !== undefined ? askData : normalizeAskResult(askData);

  // Safety disclaimer
  const safetyBanner = `
    <div class="safety-banner">
      <span class="safety-icon">&bull;</span>
      <span><strong>Safety Guarantee:</strong> The engine only PROPOSES code replacements and never writes directly to your repository.</span>
    </div>
  `;

  // Dry run mode display
  if (vm.dry_run) {
    const tr = vm.token_report;
    container.innerHTML = `
      ${safetyBanner}
      <div class="dry-run-card">
        <div class="dry-run-header">
          <div class="dry-run-title">Dry Run Preview (Zero API Calls)</div>
          <div class="badge badge-subtle">dry_run = true</div>
        </div>
        <p class="dry-run-desc">
          Below is the exact prompt message constructed for the model. No external API call was made.
        </p>
        <div class="token-report-card">
          <h4>Prompt Token Footprint</h4>
          <div class="tr-grid">
            <div><span>Context Tokens:</span> <strong>${tr.context_tokens.toLocaleString()}</strong></div>
            <div><span>Budget Limit:</span> <strong>${tr.budget.toLocaleString()}</strong></div>
            <div><span>Utilization:</span> <strong>${(tr.utilization * 100).toFixed(1)}%</strong></div>
            <div><span>Files in Context:</span> <strong>${tr.files_in_context}</strong> of ${tr.files_in_repo}</div>
          </div>
        </div>
        <details class="prompt-details" open>
          <summary>System Prompt</summary>
          <pre><code>${escapeHtml(vm.system_prompt)}</code></pre>
        </details>
        <details class="prompt-details" open>
          <summary>User Context Message</summary>
          <pre><code>${escapeHtml(vm.messages[0] ? vm.messages[0].content : '')}</code></pre>
        </details>
      </div>
    `;
    return;
  }

  const answer = vm.answer;
  const tokenReport = vm.token_report;

  // Token Report Card
  let costStr = tokenReport.cost_display || 'Unknown';
  if (tokenReport.is_free_tier || costStr.toLowerCase() === 'free tier') {
    costStr = '<span class="badge badge-free-tier">Free Tier</span>';
  } else if (tokenReport.estimated_cost_usd !== null && tokenReport.estimated_cost_usd !== undefined) {
    costStr = `$${tokenReport.estimated_cost_usd.toFixed(4)}`;
  }

  const estBadge = tokenReport.usage_estimated ? ' <span class="badge-est">(est)</span>' : '';
  const provName = tokenReport.provider || vm.provider || 'default';
  const modelName = tokenReport.model || vm.model || 'default';

  const tokenReportHtml = `
    <div class="token-report-card">
      <div class="tr-header">
        <span>Execution & Token Metrics</span>
        <span class="provider-model-tag"><strong>${escapeHtml(provName)}</strong> / ${escapeHtml(modelName)}</span>
      </div>
      <div class="tr-grid">
        <div><span>Provider Input:</span> <strong>${tokenReport.provider_input_tokens !== null ? tokenReport.provider_input_tokens.toLocaleString() + estBadge : 'N/A' + estBadge}</strong></div>
        <div><span>Provider Output:</span> <strong>${tokenReport.provider_output_tokens !== null ? tokenReport.provider_output_tokens.toLocaleString() + estBadge : 'N/A' + estBadge}</strong></div>
        <div><span>Cost:</span> <strong>${costStr}</strong></div>
        <div><span>Total Latency:</span> <strong>${tokenReport.total_seconds.toFixed(2)}s</strong> (LLM: ${tokenReport.llm_seconds.toFixed(2)}s)</div>
        ${tokenReport.key_source ? `<div><span>Key Source:</span> <span class="badge ${tokenReport.key_source === 'user' ? 'badge-user-key' : 'badge-server-key'}">${escapeHtml(tokenReport.key_source)}</span></div>` : ''}
      </div>
    </div>
  `;

  // Analysis section: display structured analysis or fallback to completion text
  const rawText = vm.completion || (vm.raw && vm.raw.completion ? (typeof vm.raw.completion === 'string' ? vm.raw.completion : vm.raw.completion.text) : '');
  const displayText = (answer && answer.analysis && answer.analysis.trim())
    ? answer.analysis.trim()
    : (rawText && String(rawText).trim() ? String(rawText).trim() : '');

  const analysisHtml = displayText ? `
    <div class="answer-section">
      <h3 class="section-heading">AI Response & Analysis</h3>
      <div class="analysis-prose" style="white-space: pre-wrap;">${escapeHtml(displayText)}</div>
    </div>
  ` : '';

  // Needs context chips
  let needsContextHtml = '';
  if (answer && answer.needs_context && answer.needs_context.length > 0) {
    const chips = answer.needs_context.map(name => `
      <button class="needs-context-chip" data-entity="${escapeHtml(name)}" title="Click to pin this entity in prompt retrieval">
        + Pin <code>${escapeHtml(name)}</code>
      </button>
    `).join(' ');

    needsContextHtml = `
      <div class="needs-context-alert">
        <div class="nc-title">Model Requested Additional Context:</div>
        <p class="nc-desc">The AI noted that inspecting the following entities would improve its answer. Click any chip to pin it into context:</p>
        <div class="nc-chips">${chips}</div>
      </div>
    `;
  }

  // Changes section
  let changesHtml = '';
  if (answer && answer.changes && answer.changes.length > 0) {
    const changeCards = answer.changes.map(ch => {
      const syntaxChip = ch.syntax_valid
        ? '<span class="badge badge-success">Syntax Valid</span>'
        : '<span class="badge badge-danger">Syntax Error</span>';

      const targetChip = ch.unknown_target
        ? '<span class="badge badge-warning" title="Target function was not in the provided prompt context">Target Not In Context</span>'
        : '<span class="badge badge-subtle">Target Verified</span>';

      return `
        <div class="change-card">
          <div class="change-header">
            <div class="change-target">
              <strong>${escapeHtml(ch.qualified_name || 'Replacement Code')}</strong>
              <span class="change-file">${escapeHtml(ch.file_path || '')}</span>
            </div>
            <div class="change-badges">
              ${syntaxChip}
              ${targetChip}
            </div>
          </div>
          <div class="change-code-wrapper">
            <pre><code class="language-python">${escapeHtml(ch.code || '')}</code></pre>
          </div>
        </div>
      `;
    }).join('');

    changesHtml = `
      <div class="answer-section">
        <h3 class="section-heading">Proposed Code Changes (${answer.changes.length})</h3>
        <div class="changes-list">${changeCards}</div>
      </div>
    `;
  } else {
    changesHtml = `
      <div class="answer-section">
        <div class="empty-state">No specific code replacements proposed by the assistant.</div>
      </div>
    `;
  }

  // Raw completion collapsible
  const completionHtml = vm.completion ? `
    <details class="prompt-details" style="margin-top: 20px;">
      <summary>Raw Provider Completion</summary>
      <pre><code>${escapeHtml(vm.completion)}</code></pre>
    </details>
  ` : '';

  container.innerHTML = `
    ${safetyBanner}
    ${tokenReportHtml}
    ${analysisHtml}
    ${needsContextHtml}
    ${changesHtml}
    ${completionHtml}
  `;

  // Attach event handlers for pin buttons
  if (onPinEntity) {
    container.querySelectorAll('.needs-context-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const entity = btn.dataset.entity;
        if (entity) onPinEntity(entity);
      });
    });
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
