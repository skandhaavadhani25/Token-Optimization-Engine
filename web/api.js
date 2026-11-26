import { keyManager } from './key_manager.js';

const API_BASE = '/api';

/**
 * Handle HTTP response and return JSON or throw descriptive error.
 */
async function handleResponse(res) {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    let errorCode = null;
    let requestId = res.headers ? res.headers.get('x-request-id') : null;
    try {
      const data = await res.json();
      if (data && data.error) {
        if (data.error.message) errorDetail = data.error.message;
        if (data.error.code) errorCode = data.error.code;
        if (data.error.request_id) requestId = data.error.request_id;
      } else if (data && data.detail) {
        errorDetail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
      }
    } catch (_) {
      // Non-JSON error body
    }
    // Never show raw key in error detail
    errorDetail = keyManager.redactText(errorDetail);
    const err = new Error(errorDetail);
    err.status = res.status;
    err.code = errorCode;
    err.requestId = requestId;
    throw err;
  }
  return await res.json();
}

export const api = {
  /**
   * Public service configuration
   */
  async getConfig() {
    const res = await fetch(`${API_BASE}/config`);
    return handleResponse(res);
  },

  /**
   * Health and readiness status
   */
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse(res);
  },

  /**
   * Available AI completion providers and status
   */
  async getProviders() {
    const res = await fetch(`${API_BASE}/providers`);
    return handleResponse(res);
  },

  /**
   * Repository index stats and staleness
   */
  async getStats(target) {
    let url = `${API_BASE}/stats`;
    if (typeof target === 'object' && target !== null) {
      if (target.workspaceId) url += `?workspace_id=${encodeURIComponent(target.workspaceId)}`;
      else if (target.repoPath) url += `?repo_path=${encodeURIComponent(target.repoPath)}`;
    } else if (target) {
      url += `?repo_path=${encodeURIComponent(target)}`;
    }
    const res = await fetch(url);
    return handleResponse(res);
  },

  /**
   * Full or forced repository indexing
   */
  async indexRepository(target, force = false) {
    const payload = { force };
    if (typeof target === 'object' && target !== null) {
      if (target.workspaceId) payload.workspace_id = target.workspaceId;
      if (target.repoPath) payload.repo_path = target.repoPath;
      if (target.force !== undefined) payload.force = target.force;
    } else {
      payload.repo_path = target;
    }
    const res = await fetch(`${API_BASE}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  /**
   * One-shot incremental indexing
   */
  async updateIndex(target, verify = false) {
    const payload = { verify };
    if (typeof target === 'object' && target !== null) {
      if (target.workspaceId) payload.workspace_id = target.workspaceId;
      if (target.repoPath) payload.repo_path = target.repoPath;
      if (target.verify !== undefined) payload.verify = target.verify;
    } else {
      payload.repo_path = target;
    }
    const res = await fetch(`${API_BASE}/index/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  /**
   * Hybrid retrieval and context building
   */
  async retrieveContext({
    repoPath = null,
    workspaceId = null,
    query,
    topK = 10,
    maxTokens = 4000,
    useSemantic = true,
    useKeyword = true,
    useGraph = true,
    explain = true,
  }) {
    const payload = {
      query,
      top_k: topK,
      max_tokens: maxTokens,
      use_semantic: useSemantic,
      use_keyword: useKeyword,
      use_graph: useGraph,
      explain,
    };
    if (workspaceId) payload.workspace_id = workspaceId;
    if (repoPath) payload.repo_path = repoPath;

    const res = await fetch(`${API_BASE}/retrieve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  /**
   * Validate provider API key with a cheap call (list models or 1 token).
   */
  async validateKey(provider, key, model = null) {
    const headers = {
      'Content-Type': 'application/json',
    };
    if (key) {
      headers['X-Provider-Key'] = key;
    }
    const res = await fetch(`${API_BASE}/providers/${encodeURIComponent(provider)}/validate`, {
      method: 'POST',
      headers,
      body: JSON.stringify(model ? { model } : {}),
    });
    return handleResponse(res);
  },

  /**
   * Fetch provider available models, optionally passing user key.
   */
  async getProviderModels(provider, key = null) {
    const headers = {};
    const effectiveKey = key || keyManager.getKey(provider);
    if (effectiveKey) {
      headers['X-Provider-Key'] = effectiveKey;
    }
    const res = await fetch(`${API_BASE}/providers/${encodeURIComponent(provider)}/models`, {
      headers,
    });
    return handleResponse(res);
  },

  /**
   * Run coding assistant with optional dry_run, provider, and model override
   */
  async ask({
    repoPath = null,
    workspaceId = null,
    query,
    maxTokens = 4000,
    dryRun = false,
    provider = null,
    model = null,
    key = null,
  }) {
    const payload = {
      query,
      max_tokens: maxTokens,
      dry_run: dryRun,
    };
    if (workspaceId) payload.workspace_id = workspaceId;
    if (repoPath) payload.repo_path = repoPath;
    if (provider) payload.provider = provider;
    if (model) payload.model = model;

    const headers = { 'Content-Type': 'application/json' };
    const effectiveKey = key || (provider ? keyManager.getKey(provider) : null);
    if (effectiveKey) {
      headers['X-Provider-Key'] = effectiveKey;
    }

    const res = await fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  /**
   * Fast single-query reduction vs whole_file baseline
   */
  async benchmarkOne({ repoPath = null, workspaceId = null, query, budget = 4000 }) {
    const payload = { query, budget };
    if (workspaceId) payload.workspace_id = workspaceId;
    if (repoPath) payload.repo_path = repoPath;

    const res = await fetch(`${API_BASE}/benchmark-one`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  /**
   * Subgraph topology for visualization
   */
  async getGraph({ repoPath = null, workspaceId = null, center = null, hops = 2, maxNodes = 150, rels = null }) {
    let url = `${API_BASE}/graph?hops=${hops}&max_nodes=${maxNodes}`;
    if (workspaceId) url += `&workspace_id=${encodeURIComponent(workspaceId)}`;
    else if (repoPath) url += `&repo_path=${encodeURIComponent(repoPath)}`;
    if (center) url += `&center=${encodeURIComponent(center)}`;
    if (rels) url += `&rels=${encodeURIComponent(rels)}`;
    const res = await fetch(url);
    return handleResponse(res);
  },

  /**
   * Single node details, callers, callees, dependencies, and source
   */
  async getGraphNode(targetOrPath, nodeId) {
    let url = `${API_BASE}/graph/node?id=${encodeURIComponent(nodeId)}`;
    if (typeof targetOrPath === 'object' && targetOrPath !== null) {
      if (targetOrPath.workspaceId) url += `&workspace_id=${encodeURIComponent(targetOrPath.workspaceId)}`;
      else if (targetOrPath.repoPath) url += `&repo_path=${encodeURIComponent(targetOrPath.repoPath)}`;
    } else if (targetOrPath) {
      url += `&repo_path=${encodeURIComponent(targetOrPath)}`;
    }
    const res = await fetch(url);
    return handleResponse(res);
  },

  /**
   * List available benchmark query set JSON files
   */
  async getBenchmarkQuerySets() {
    const res = await fetch(`${API_BASE}/benchmark/querysets`);
    return handleResponse(res);
  },

  /**
   * Run full multi-strategy benchmark
   */
  async runBenchmark({
    repoPath = null,
    workspaceId = null,
    queryFile = null,
    queries = null,
    budget = 4000,
    callAi = false,
    strategies = null,
  }) {
    const payload = {
      budget,
      call_ai: callAi,
    };
    if (workspaceId) payload.workspace_id = workspaceId;
    if (repoPath) payload.repo_path = repoPath;
    if (queryFile) payload.query_file = queryFile;
    if (queries) payload.queries = queries;
    if (strategies) payload.strategies = strategies;

    const res = await fetch(`${API_BASE}/benchmark`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return handleResponse(res);
  },

  // ==========================================
  // Workspaces API
  // ==========================================

  async listWorkspaces() {
    const res = await fetch(`${API_BASE}/workspaces`);
    return handleResponse(res);
  },

  async getWorkspace(id) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(id)}`);
    return handleResponse(res);
  },

  async createWorkspace(formDataOrJson) {
    const options = { method: 'POST' };
    if (formDataOrJson instanceof FormData) {
      options.body = formDataOrJson;
    } else {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(formDataOrJson);
    }
    const res = await fetch(`${API_BASE}/workspaces`, options);
    return handleResponse(res);
  },

  async addWorkspaceFiles(workspaceId, formData) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(workspaceId)}/files`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async deleteWorkspace(id) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(id)}`, { method: 'DELETE' });
    return handleResponse(res);
  },

  async reindexWorkspace(id) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(id)}/reindex`, { method: 'POST' });
    return handleResponse(res);
  },

  async getWorkspaceFile(workspaceId, path, startLine = null, endLine = null) {
    let url = `${API_BASE}/workspaces/${encodeURIComponent(workspaceId)}/file?path=${encodeURIComponent(path)}`;
    if (startLine) url += `&start_line=${startLine}`;
    if (endLine) url += `&end_line=${endLine}`;
    const res = await fetch(url);
    return handleResponse(res);
  },

  async applyEdit(workspaceId, editData) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(workspaceId)}/edits/apply`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(editData),
    });
    return handleResponse(res);
  },

  async undoEdit(workspaceId) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(workspaceId)}/edits/undo`, {
      method: 'POST',
    });
    return handleResponse(res);
  },

  async listEdits(workspaceId) {
    const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(workspaceId)}/edits/history`);
    return handleResponse(res);
  },

  // ==========================================
  // Conversations & Chat API
  // ==========================================

  async listConversations(workspaceId = null) {
    const url = workspaceId
      ? `${API_BASE}/conversations?workspace_id=${encodeURIComponent(workspaceId)}`
      : `${API_BASE}/conversations`;
    const res = await fetch(url);
    return handleResponse(res);
  },

  async createConversation(data) {
    const res = await fetch(`${API_BASE}/conversations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  async getConversation(id) {
    const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(id)}`);
    return handleResponse(res);
  },

  async deleteConversation(id) {
    const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    return handleResponse(res);
  },

  async sendMessage(conversationId, data, providerKey = null) {
    const headers = { 'Content-Type': 'application/json' };
    if (providerKey) {
      headers['X-Provider-Key'] = providerKey;
    }
    const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}/messages`, {
      method: 'POST',
      headers,
      body: JSON.stringify(data),
    });
    return handleResponse(res);
  },

  async uploadAttachment(conversationId, formData, provider = null, model = null) {
    let url = `${API_BASE}/conversations/${encodeURIComponent(conversationId)}/attachments`;
    const params = [];
    if (provider) params.push(`provider=${encodeURIComponent(provider)}`);
    if (model) params.push(`model=${encodeURIComponent(model)}`);
    if (params.length) url += `?${params.join('&')}`;

    const res = await fetch(url, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async listAttachments(conversationId) {
    const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}/attachments`);
    return handleResponse(res);
  },

  async deleteAttachment(conversationId, attachmentId) {
    const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}/attachments/${encodeURIComponent(attachmentId)}`, {
      method: 'DELETE',
    });
    return handleResponse(res);
  },

  async submitFeedback(conversationId, messageId, rating) {
    const res = await fetch(`${API_BASE}/conversations/${encodeURIComponent(conversationId)}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message_id: messageId, rating }),
    });
    return handleResponse(res);
  },
};

