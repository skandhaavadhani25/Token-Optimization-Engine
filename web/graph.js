/**
 * Cytoscape.js interactive dependency graph module.
 */

import { api } from './api.js';

let cyInstance = null;
let currentCenterNode = null;
let currentTarget = null;
let lastGraphData = null;

const NODE_COLORS = {
  project: '#f97316',
  folder: '#f59e0b',
  file: '#0ea5e9',
  class: '#8b5cf6',
  function: '#6366f1',
  method: '#14b8a6',
  external: '#a1a1aa',
  unresolved: '#71717a',
  text_only: '#0ea5e9',
};

const LEGEND = [
  ['Function', 'function'],
  ['Method', 'method'],
  ['Class', 'class'],
  ['File', 'file'],
  ['Library / outside', 'external'],
];

const prefersReducedMotion = () =>
  window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

export function initGraphView({ containerId, panelId, onUseAsTarget }) {
  const container = document.getElementById(containerId);
  const panel = document.getElementById(panelId);
  if (!container) return;

  // Colour key so people can read the graph without guessing
  const card = container.parentElement;
  if (card && !card.querySelector('.graph-legend')) {
    const legend = document.createElement('div');
    legend.className = 'graph-legend';
    legend.setAttribute('aria-label', 'Graph colour key');
    legend.innerHTML = LEGEND
      .map(([label, type]) => `<span><i style="background:${NODE_COLORS[type]}"></i>${label}</span>`)
      .join('');
    card.appendChild(legend);
  }

  const checkHideExternal = document.getElementById('check-hide-external');
  if (checkHideExternal) {
    checkHideExternal.addEventListener('change', () => {
      applyFilterExternal(checkHideExternal.checked);
    });
  }

  const btnRebuild = document.getElementById('btn-rebuild-graph');
  if (btnRebuild) {
    btnRebuild.addEventListener('click', async () => {
      if (!currentTarget || (!currentTarget.workspaceId && !currentTarget.repoPath)) {
        notify('No active workspace to rebuild.');
        return;
      }
      btnRebuild.disabled = true;
      btnRebuild.textContent = 'Rebuilding...';
      renderBanner(panel, 'Rebuilding graph index...');
      try {
        await api.indexRepository(currentTarget, true);
        await graphController.loadOverview(currentTarget);
      } catch (err) {
        renderBanner(panel, `Rebuild failed: ${err.message}`, true);
      } finally {
        btnRebuild.disabled = false;
        btnRebuild.textContent = 'Rebuild Graph';
      }
    });
  }

  function applyFilterExternal(hide) {
    if (!cyInstance) return;
    const extNodes = cyInstance.elements('node[type = "external"], node[type = "unresolved"]');
    if (hide) {
      extNodes.style('display', 'none');
    } else {
      extNodes.style('display', 'element');
    }
  }

  const graphController = {
    async loadOverview(target) {
      currentTarget = target;
      updateWorkspaceBadge(target);
      if (!target || (!target.workspaceId && !target.repoPath)) {
        renderEmptyState(container, panel);
        return;
      }
      currentCenterNode = null;
      renderBanner(panel, 'Loading graph overview...');
      try {
        const data = await api.getGraph({
          workspaceId: target.workspaceId,
          repoPath: target.repoPath,
          hops: 2,
          maxNodes: 120,
        });
        lastGraphData = data;
        renderCytoscape(data, container, panel, onUseAsTarget, target);
      } catch (err) {
        renderBanner(panel, `Error loading graph: ${err.message}`, true);
      }
    },

    async centerOnSymbol(target, symbol, hops = 2) {
      currentTarget = target;
      updateWorkspaceBadge(target);
      if (!target || (!target.workspaceId && !target.repoPath) || !symbol) return;
      currentCenterNode = symbol;
      renderBanner(panel, `Searching for '${symbol}'...`);
      try {
        const data = await api.getGraph({
          workspaceId: target.workspaceId,
          repoPath: target.repoPath,
          center: symbol,
          hops,
          maxNodes: 150,
        });
        lastGraphData = data;
        renderCytoscape(data, container, panel, onUseAsTarget, target, true);
      } catch (err) {
        renderBanner(panel, `Symbol '${symbol}' not found: ${err.message}`, true);
      }
    },

    showIndexingState(message = 'Indexing code and building graph...') {
      container.innerHTML = `
        <div class="empty-state" style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%;">
          <div class="spinner" style="margin-bottom: 12px; font-size: 24px;">🔄</div>
          <h4>${escapeHtml(message)}</h4>
          <p style="font-size: 12px; color: var(--text-secondary); margin-top: 6px;">Graph will refresh automatically once indexing finishes.</p>
        </div>
      `;
    },

    clear() {
      currentTarget = null;
      lastGraphData = null;
      renderEmptyState(container, panel);
    },
  };

  return graphController;
}

function updateWorkspaceBadge(target) {
  const badge = document.getElementById('graph-workspace-badge');
  if (!badge) return;
  if (!target || (!target.workspaceId && !target.repoPath)) {
    badge.textContent = 'No project';
    return;
  }
  const name = target.workspaceName || target.workspaceId || target.repoPath || 'Workspace';
  badge.textContent = name;
}

function renderEmptyState(container, panel) {
  if (cyInstance) {
    cyInstance.destroy();
    cyInstance = null;
  }
  container.innerHTML = '<div class="empty-state">Add your code first: use "Open / Upload Code" at the top.</div>';
  if (panel) {
    panel.innerHTML = '<div class="panel-placeholder">Click any graph node to inspect symbol details, callers, callees, and dependencies.</div>';
  }
  const countsEl = document.getElementById('graph-counts-badge');
  if (countsEl) countsEl.textContent = '';
  const badge = document.getElementById('graph-workspace-badge');
  if (badge) badge.textContent = 'No project';
}

function renderCytoscape(graphData, container, panel, onUseAsTarget, target, isNeighborhood = false) {
  if (typeof window.cytoscape === 'undefined') {
    container.innerHTML = '<div class="empty-state">The graph drawing library could not load. Check your internet connection and refresh the page.</div>';
    return;
  }

  if (cyInstance) {
    cyInstance.destroy();
    cyInstance = null;
  }

  const elements = [];
  const rawData = graphData || {};
  const nodes = Array.isArray(rawData.nodes) ? rawData.nodes : [];
  const edges = Array.isArray(rawData.edges) ? rawData.edges : [];
  const totalNodes = rawData.total_nodes || nodes.length;
  const isTruncated = Boolean(rawData.truncated);

  // Compute counts
  let funcCount = 0;
  let classCount = 0;
  let callEdgesCount = 0;
  let textOnlyCount = 0;

  for (const n of nodes) {
    const t = n.type || 'function';
    if (t === 'function' || t === 'method') funcCount++;
    else if (t === 'class') classCount++;
    else if (t === 'text_only') textOnlyCount++;
  }

  for (const e of edges) {
    if (e.rel === 'CALLS') callEdgesCount++;
  }

  const countsBadge = document.getElementById('graph-counts-badge');
  if (countsBadge) {
    countsBadge.textContent = `Functions: ${funcCount} | Classes: ${classCount} | Calls: ${callEdgesCount}`;
  }

  if (nodes.length === 0) {
    container.innerHTML = '<div class="empty-state">No functions found yet. Upload Python code, then click "Rebuild Graph".</div>';
    panel.innerHTML = '<div class="panel-placeholder">No graph data.</div>';
    return;
  }

  // Check if no functions or classes exist
  const hasCallables = funcCount > 0 || classCount > 0;

  // Add nodes
  for (const n of nodes) {
    const nodeType = n.type || 'function';
    const label = n.label || n.name || n.id || 'unknown';
    const degree = (n.in_degree || 0) + (n.out_degree || 0);
    const size = Math.max(22, Math.min(60, 22 + degree * 2.5));
    const isDim = nodeType === 'external' || nodeType === 'unresolved';

    elements.push({
      group: 'nodes',
      data: {
        id: n.id,
        label: truncateLabel(label),
        fullLabel: label,
        type: nodeType,
        color: NODE_COLORS[nodeType] || '#8b949e',
        size,
        opacity: isDim ? 0.45 : 1.0,
        borderStyle: isDim ? 'dashed' : 'solid',
        note: n.note || '',
      },
    });
  }

  // Add edges
  for (const e of edges) {
    const rel = e.rel || 'CALLS';
    let lineStyle = 'solid';
    let opacity = 0.7;

    if (rel === 'IMPORTS') {
      lineStyle = 'dashed';
      opacity = 0.5;
    } else if (rel === 'INHERITS') {
      lineStyle = 'dotted';
      opacity = 0.8;
    } else if (rel === 'DEFINES') {
      lineStyle = 'dashed';
      opacity = 0.4;
    } else if (rel === 'CALLS') {
      opacity = Math.max(0.4, Math.min(1.0, e.confidence || 0.8));
    }

    elements.push({
      group: 'edges',
      data: {
        id: `${e.source}->${e.target}:${rel}:${Math.random()}`,
        source: e.source,
        target: e.target,
        rel,
        confidence: e.confidence,
        resolution: e.resolution,
        lineno: e.lineno,
        lineStyle,
        opacity,
      },
    });
  }

  const layoutConfig = isNeighborhood
    ? {
        name: 'cose',
        animate: prefersReducedMotion() ? false : 'end',
        animationDuration: 500,
        randomize: false,
        componentSpacing: 100,
        nodeOverlap: 20,
        padding: 40,
      }
    : {
        name: 'concentric',
        animate: !prefersReducedMotion(),
        animationDuration: 700,
        animationEasing: 'ease-out-cubic',
        concentric: node => node.data('size') || 20,
        levelWidth: () => 10,
        padding: 50,
      };

  container.innerHTML = '';
  const labelColor = cssVar('--text-primary', '#09090b');
  const labelOutline = cssVar('--bg-primary', '#ffffff');
  const edgeColor = cssVar('--text-muted', '#71717a');
  const accent = cssVar('--accent-color', '#4f46e5');
  const pink = cssVar('--accent-color', '#4f46e5');
  cyInstance = window.cytoscape({
    container,
    elements,
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'data(color)',
          label: 'data(label)',
          color: labelColor,
          'text-outline-color': labelOutline,
          'text-outline-width': 2,
          'font-size': '11px',
          'font-family': 'Geist, Inter, sans-serif',
          'text-valign': 'bottom',
          'text-margin-y': 5,
          width: 'data(size)',
          height: 'data(size)',
          'border-width': 2,
          'border-color': labelOutline,
          'border-style': 'data(borderStyle)',
          opacity: 'data(opacity)',
        },
      },
      {
        selector: 'node:selected',
        style: {
          'border-width': 3,
          'border-color': pink,
          'shadow-blur': 12,
          'shadow-color': pink,
          opacity: 1.0,
        },
      },
      {
        selector: 'edge',
        style: {
          width: 1.5,
          'line-color': edgeColor,
          'target-arrow-color': edgeColor,
          'target-arrow-shape': 'triangle',
          'curve-style': 'bezier',
          'line-style': 'data(lineStyle)',
          opacity: 'data(opacity)',
        },
      },
      {
        selector: 'edge:selected',
        style: {
          width: 3,
          'line-color': accent,
          'target-arrow-color': accent,
          opacity: 1.0,
        },
      },
    ],
    layout: layoutConfig,
  });

  const checkHideExternal = document.getElementById('check-hide-external');
  if (checkHideExternal && checkHideExternal.checked) {
    cyInstance.elements('node[type = "external"], node[type = "unresolved"]').style('display', 'none');
  }

  let bannerHtml = '';
  if (!hasCallables) {
    bannerHtml += `
      <div class="warning-banner" style="margin-bottom: 8px;">
        ℹ️ This file has no functions or classes to display in the call graph.
      </div>
    `;
  }
  if (textOnlyCount > 0) {
    bannerHtml += `
      <div class="info-banner" style="margin-bottom: 8px;">
        📄 ${textOnlyCount} non-Python file(s) displayed as text-only (no call graph).
      </div>
    `;
  }
  if (isTruncated) {
    bannerHtml += `
      <div class="graph-banner-alert" style="margin-bottom: 8px;">
        <span>Showing <strong>${nodes.length}</strong> of <strong>${totalNodes}</strong> nodes</span>
      </div>
    `;
  }

  panel.innerHTML = bannerHtml + '<div class="graph-hint">Click any node or edge to inspect details.</div>';

  cyInstance.on('tap', 'node', async evt => {
    const node = evt.target;
    const nodeId = node.data('id');
    panel.innerHTML = `<div class="panel-loading">Fetching node details for <code>${escapeHtml(nodeId)}</code>...</div>`;

    try {
      const details = await api.getGraphNode(target, nodeId);
      renderNodeSidePanel(details, panel, onUseAsTarget);
    } catch (err) {
      panel.innerHTML = `<div class="error-banner">Error loading node details: ${escapeHtml(err.message)}</div>`;
    }
  });

  cyInstance.on('tap', 'edge', evt => {
    const edge = evt.target;
    renderEdgeSidePanel(edge.data(), panel);
  });
}

function renderNodeSidePanel(nodeDetails, panel, onUseAsTarget) {
  const callersHtml = (nodeDetails.callers || []).length > 0
    ? nodeDetails.callers.map(c => `<li><code>${escapeHtml(c)}</code></li>`).join('')
    : '<li class="empty-list">No callers recorded</li>';

  const calleesHtml = (nodeDetails.callees || []).length > 0
    ? nodeDetails.callees.map(c => `<li><code>${escapeHtml(c)}</code></li>`).join('')
    : '<li class="empty-list">No callees recorded</li>';

  const sourceHtml = nodeDetails.source
    ? `<pre class="node-source"><code>${escapeHtml(nodeDetails.source)}</code></pre>`
    : '<div class="empty-state">Source snippet not available</div>';

  const isTextOnly = nodeDetails.type === 'text_only' || (nodeDetails.note && nodeDetails.note.includes('text-only'));

  panel.innerHTML = `
    <div class="node-panel-card">
      <div class="np-header">
        <div class="np-title">
          <span class="badge badge-target">${escapeHtml(nodeDetails.type)}</span>
          <h4>${escapeHtml(nodeDetails.label || nodeDetails.id)}</h4>
        </div>
        ${!isTextOnly ? `
        <button id="btn-use-as-target" class="btn btn-sm btn-primary">
          &rarr; Use as target in Ask tab
        </button>
        ` : ''}
      </div>

      ${isTextOnly ? `
        <div class="info-box" style="margin: 8px 0; font-size: 12px; color: var(--text-secondary);">
          Text-only file (no call graph)
        </div>
      ` : ''}

      <div class="np-loc">
        <span>${escapeHtml(nodeDetails.file || '')}${nodeDetails.start_line ? `:${nodeDetails.start_line}-${nodeDetails.end_line}` : ''}</span>
      </div>

      ${!isTextOnly ? `
      <div class="np-relations">
        <details open>
          <summary>Callers (${(nodeDetails.callers || []).length})</summary>
          <ul>${callersHtml}</ul>
        </details>
        <details open>
          <summary>Callees (${(nodeDetails.callees || []).length})</summary>
          <ul>${calleesHtml}</ul>
        </details>
      </div>
      ` : ''}

      <div class="np-source-section">
        <h5>Definition Source</h5>
        ${sourceHtml}
      </div>
    </div>
  `;

  const btn = panel.querySelector('#btn-use-as-target');
  if (btn && onUseAsTarget) {
    btn.addEventListener('click', () => {
      onUseAsTarget(nodeDetails.label || nodeDetails.id);
    });
  }
}

function renderEdgeSidePanel(edgeData, panel) {
  panel.innerHTML = `
    <div class="edge-panel-card">
      <h4>Relationship Detail</h4>
      <div class="ep-meta">
        <div><span>Type:</span> <strong><span class="badge badge-subtle">${escapeHtml(edgeData.rel)}</span></strong></div>
        <div><span>Source:</span> <code>${escapeHtml(edgeData.source)}</code></div>
        <div><span>Target:</span> <code>${escapeHtml(edgeData.target)}</code></div>
        ${edgeData.confidence !== undefined ? `<div><span>Confidence:</span> <strong>${(edgeData.confidence * 100).toFixed(0)}%</strong></div>` : ''}
        ${edgeData.resolution ? `<div><span>Resolver Rule:</span> <code>${escapeHtml(edgeData.resolution)}</code></div>` : ''}
        ${edgeData.lineno ? `<div><span>Call Line Number:</span> Line ${edgeData.lineno}</div>` : ''}
      </div>
    </div>
  `;
}

function renderBanner(panel, message, isError = false) {
  panel.innerHTML = `
    <div class="${isError ? 'error-banner' : 'info-banner'}">
      ${escapeHtml(message)}
    </div>
  `;
}

function truncateLabel(str) {
  if (!str) return '';
  const parts = str.split('.');
  const shortName = parts[parts.length - 1];
  return shortName.length > 20 ? shortName.substring(0, 18) + '...' : shortName;
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
