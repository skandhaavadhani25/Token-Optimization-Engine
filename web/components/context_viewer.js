/**
 * Component to inspect the exact prompt context built by the engine.
 * Reads from normalized context view models.
 */

import { normalizeContext } from '../normalizer.js';

export function renderContextViewer(contextData, container) {
  if (!contextData) {
    container.innerHTML = '';
    return;
  }

  const vm = contextData.level_counts !== undefined ? contextData : normalizeContext(contextData);
  if (!vm.text) {
    container.innerHTML = '';
    return;
  }

  const sections = parseContextSections(vm.text);

  let sectionsHtml = sections.map((sec, idx) => {
    return `
      <details class="context-section" ${idx < 3 ? 'open' : ''}>
        <summary class="section-summary">
          <span class="section-title">${escapeHtml(sec.title)}</span>
          <span class="section-chars">${sec.content.length.toLocaleString()} chars</span>
        </summary>
        <div class="section-body">
          <pre><code>${escapeHtml(sec.content.trim())}</code></pre>
        </div>
      </details>
    `;
  }).join('');

  container.innerHTML = `
    <div class="context-viewer-card">
      <div class="context-viewer-header">
        <div class="cv-title">Assembled Prompt Context Preview</div>
        <div class="cv-meta">
          <span>${vm.tokens.toLocaleString()} tokens</span>
          <span>&bull;</span>
          <span>${vm.items.length} items included</span>
        </div>
      </div>
      <div class="context-sections-list">
        ${sectionsHtml}
      </div>
    </div>
  `;
}

function parseContextSections(text) {
  const titles = [
    'TASK:',
    'TARGET:',
    'DEPENDENCY MAP:',
    'RELATED FUNCTIONS:',
    'RELEVANT IMPORTS:',
    'SOURCE CODE:',
    'OMITTED',
  ];

  const sections = [];
  const lines = (text || '').split('\n');
  let currentTitle = 'HEADER';
  let currentLines = [];

  for (const line of lines) {
    const matched = titles.find(t => line.startsWith(t));
    if (matched) {
      if (currentLines.length > 0) {
        sections.push({ title: currentTitle, content: currentLines.join('\n') });
      }
      currentTitle = matched.replace(':', '');
      currentLines = [line];
    } else {
      currentLines.push(line);
    }
  }

  if (currentLines.length > 0) {
    sections.push({ title: currentTitle, content: currentLines.join('\n') });
  }

  return sections;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
