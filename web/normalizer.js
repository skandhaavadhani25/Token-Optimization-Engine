/**
 * Normalization layer for Code Context Engine API responses.
 * Maps raw API payloads into typed view models with explicit safe defaults.
 * Components read ONLY from these view models, preventing runtime TypeErrors.
 */

export function normalizeContext(rawCtx) {
  const ctx = rawCtx || {};
  const tokens = typeof ctx.tokens === 'number' ? ctx.tokens : 0;
  const budget = typeof ctx.budget_total === 'number' ? ctx.budget_total : 4000;
  const utilization = typeof ctx.utilization === 'number' ? ctx.utilization : (budget > 0 ? tokens / budget : 0);

  const rawItems = Array.isArray(ctx.items) ? ctx.items : [];
  const items = rawItems.map(item => {
    const it = item || {};
    return {
      ref: String(it.ref || 'unknown'),
      file: String(it.file || ''),
      lines: String(it.lines || ''),
      level: String(it.level || 'REFERENCE').toUpperCase(),
      tokens: typeof it.tokens === 'number' ? it.tokens : 0,
      original_tokens: typeof it.original_tokens === 'number' ? it.original_tokens : 0,
      role: String(it.role || 'related'),
      score: typeof it.score === 'number' ? it.score : 0,
      reason: String(it.reason || ''),
    };
  });

  const levelCounts = { FULL: 0, TRIMMED: 0, SIGNATURE: 0, STUB: 0, REFERENCE: 0 };
  for (const it of items) {
    if (levelCounts[it.level] !== undefined) {
      levelCounts[it.level]++;
    } else {
      levelCounts.REFERENCE++;
    }
  }

  return {
    text: typeof ctx.text === 'string' ? ctx.text : '',
    tokens,
    budget_total: budget,
    utilization: Math.min(1.0, Math.max(0.0, utilization)),
    target: ctx.target ? String(ctx.target) : null,
    target_uncertain: Boolean(ctx.target_uncertain),
    items,
    omitted_items: Array.isArray(ctx.omitted_items) ? ctx.omitted_items : [],
    downgraded_count: typeof ctx.downgraded_count === 'number' ? ctx.downgraded_count : 0,
    files_included: Array.isArray(ctx.files_included) ? ctx.files_included.map(String) : [],
    counter_name: String(ctx.counter_name || 'tiktoken'),
    build_time_seconds: typeof ctx.build_time_seconds === 'number' ? ctx.build_time_seconds : 0,
    warnings: Array.isArray(ctx.warnings) ? ctx.warnings.map(String) : [],
    level_counts: levelCounts,
    raw: rawCtx,
  };
}

const num = (v, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? v : d);

export function normalizeTrace(rawTrace) {
  if (!rawTrace || typeof rawTrace !== 'object') return null;
  const t = rawTrace;
  const sel = t.selection || null;
  const sav = t.savings || null;

  return {
    intent: String(t.intent || 'general'),
    target: t.target ? String(t.target) : null,
    target_confidence: String(t.target_confidence || 'unknown'),
    target_reason: String(t.target_reason || ''),
    signals: Object.entries(t.signals || {}).map(([name, status]) => {
      const text = String(status);
      const state = text.startsWith('used') ? 'used' : (text.startsWith('failed') ? 'failed' : 'skipped');
      return { name, state, detail: text.replace(/^(used|skipped|failed):?\s*/, '') };
    }),
    hits: (Array.isArray(t.hits) ? t.hits : []).map(h => ({
      ref: String(h?.ref || ''),
      signal: String(h?.signal || ''),
      score: num(h?.score),
      file: String(h?.file || ''),
      note: String(h?.note || ''),
    })),
    graph_edges: (Array.isArray(t.graph_edges) ? t.graph_edges : []).map(e => ({
      source: String(e?.source || ''),
      target: String(e?.target || ''),
      rel: String(e?.rel || 'CALLS'),
      direction: String(e?.direction || 'callee'),
      depth: num(e?.depth, 1),
      confidence: String(e?.confidence || ''),
      outcome: String(e?.outcome || ''),
    })),
    graph_nodes: num(t.graph_nodes),
    graph_edges_total: num(t.graph_edges_total),
    candidates_considered: num(t.candidates_considered),
    candidates_returned: num(t.candidates_returned),
    operations: (Array.isArray(t.operations) ? t.operations : []).map(o => ({
      step: num(o?.step),
      name: String(o?.name || ''),
      status: String(o?.status || 'ran'),
      detail: String(o?.detail || ''),
      count: typeof o?.count === 'number' ? o.count : null,
      seconds: num(o?.seconds),
    })),
    selection: sel && {
      candidates_in: num(sel.candidates_in),
      included: (Array.isArray(sel.included) ? sel.included : []).map(i => ({
        ref: String(i?.ref || ''),
        file: String(i?.file || ''),
        lines: String(i?.lines || ''),
        level: String(i?.level || 'reference').toLowerCase(),
        tokens: num(i?.tokens),
        original_tokens: num(i?.original_tokens),
        role: String(i?.role || 'related'),
        score: num(i?.score),
        reason: String(i?.reason || ''),
      })),
      omitted: (Array.isArray(sel.omitted) ? sel.omitted : []).map(o => ({
        ref: String(o?.ref || ''),
        reason: String(o?.reason || ''),
      })),
      downgraded_count: num(sel.downgraded_count),
    },
    savings: sav && {
      counter: String(sav.counter || 'tiktoken'),
      is_estimate: Boolean(sav.is_estimate),
      context_tokens: num(sav.context_tokens),
      prompt_tokens: num(sav.prompt_tokens),
      provider_input_tokens: typeof sav.provider_input_tokens === 'number' ? sav.provider_input_tokens : null,
      naive_full_repo_tokens: num(sav.naive_full_repo_tokens),
      whole_files_tokens: num(sav.whole_files_tokens),
      repo_files: num(sav.repo_files),
      context_files: num(sav.context_files),
      saved_vs_full_repo_tokens: num(sav.saved_vs_full_repo_tokens),
      saved_vs_full_repo_pct: num(sav.saved_vs_full_repo_pct),
      saved_vs_whole_files_tokens: num(sav.saved_vs_whole_files_tokens),
      saved_vs_whole_files_pct: num(sav.saved_vs_whole_files_pct),
      compression_saved_tokens: num(sav.compression_saved_tokens),
      reduction_factor: num(sav.reduction_factor),
    },
    warnings: Array.isArray(t.warnings) ? t.warnings.map(String) : [],
    raw: rawTrace,
  };
}

export function normalizeCandidate(rawCand, targetRef = null) {
  const c = rawCand || {};
  const ref = String(c.ref || c.qualified_name || c.name || 'unknown');
  const file = String(c.file || c.file_path || '');
  const score = typeof c.score === 'number' ? c.score : (typeof c.final_score === 'number' ? c.final_score : 0);
  const matchType = String(c.match_type || (Array.isArray(c.sources) ? c.sources.join(', ') : 'semantic'));
  const isTarget = (targetRef && ref === targetRef) || c.role === 'TARGET' || Boolean(c.is_target);
  const isGraphDiscovery = matchType === 'graph' || (Array.isArray(c.sources) && c.sources.length === 1 && c.sources[0] === 'graph');

  // Derive sub-scores
  let semScore = typeof c.semantic_score === 'number' ? c.semantic_score : null;
  let kwScore = typeof c.keyword_score === 'number' ? c.keyword_score : null;
  let grScore = typeof c.graph_score === 'number' ? c.graph_score : null;

  if (semScore === null && kwScore === null && grScore === null) {
    if (matchType === 'semantic') {
      semScore = score; kwScore = 0; grScore = 0;
    } else if (matchType === 'keyword' || matchType === 'exact') {
      semScore = 0; kwScore = score; grScore = 0;
    } else if (matchType === 'graph') {
      semScore = 0; kwScore = 0; grScore = score;
    } else if (matchType === 'hybrid') {
      semScore = score * 0.7; kwScore = score * 0.5; grScore = score * 0.8;
    } else {
      semScore = score; kwScore = 0; grScore = 0;
    }
  }

  return {
    ref,
    file,
    entity_type: String(c.entity_type || c.type || 'function'),
    start_line: typeof c.start_line === 'number' ? c.start_line : null,
    end_line: typeof c.end_line === 'number' ? c.end_line : null,
    score: Math.max(0, score),
    match_type: matchType,
    matched_kinds: Array.isArray(c.matched_kinds) ? c.matched_kinds.map(String) : [],
    is_target: isTarget,
    is_graph_discovery: isGraphDiscovery,
    sem_score: Math.min(1.0, Math.max(0.0, semScore || 0)),
    kw_score: Math.min(1.0, Math.max(0.0, kwScore || 0)),
    gr_score: Math.min(1.0, Math.max(0.0, grScore || 0)),
  };
}

export function normalizeRetrievalResult(rawPayload) {
  // Can accept either full { retrieval: {...}, context: {...} } or just { query, target, ... }
  const root = rawPayload || {};
  const ret = root.retrieval || root;
  const ctx = root.context || null;

  const query = String(ret.query || '');
  const intent = String(ret.intent || (ret.analyzed_query && ret.analyzed_query.intent) || 'general');
  const targetRef = ret.target ? String(ret.target) : (ret.primary_target ? String(ret.primary_target.qualified_name || ret.primary_target.name || '') : null) || null;

  let targetConfidence = 'unknown';
  if (ret.target_confidence !== undefined && ret.target_confidence !== null) {
    targetConfidence = typeof ret.target_confidence === 'number'
      ? `${(ret.target_confidence * 100).toFixed(0)}%`
      : String(ret.target_confidence).toUpperCase();
  } else if (ret.primary_target && typeof ret.primary_target.confidence === 'number') {
    targetConfidence = `${(ret.primary_target.confidence * 100).toFixed(0)}%`;
  }

  const rawCandidates = Array.isArray(ret.candidates) ? ret.candidates : [];
  const candidates = rawCandidates.map(c => normalizeCandidate(c, targetRef));
  const targetCandidate = targetRef ? candidates.find(c => c.ref === targetRef) || null : null;

  const identifiers = ret.analyzed_query && Array.isArray(ret.analyzed_query.identifiers)
    ? ret.analyzed_query.identifiers.map(String)
    : [];
  const scope = ret.analyzed_query && ret.analyzed_query.scope ? String(ret.analyzed_query.scope) : null;

  return {
    query,
    intent,
    target: targetRef,
    target_confidence: targetConfidence,
    target_candidate: targetCandidate,
    identifiers,
    scope,
    candidates,
    context: ctx ? normalizeContext(ctx) : null,
    trace: normalizeTrace(ret.trace),
    raw: rawPayload,
  };
}

export function normalizeAskResult(rawPayload) {
  const root = rawPayload || {};
  const query = String(root.query || '');
  const dryRun = Boolean(root.dry_run);
  const systemPrompt = String(root.system_prompt || '');
  const messages = Array.isArray(root.messages)
    ? root.messages.map(m => ({ role: String(m?.role || 'user'), content: String(m?.content || '') }))
    : [];

  const tr = root.token_report || {};
  const isFreeTier = Boolean(tr.is_free_tier || (tr.cost_display && tr.cost_display.toLowerCase() === 'free tier'));
  let costDisplay = 'Unknown';
  if (isFreeTier) {
    costDisplay = 'free tier';
  } else if (tr.cost_display) {
    costDisplay = String(tr.cost_display);
  } else if (typeof tr.estimated_cost_usd === 'number') {
    costDisplay = `$${tr.estimated_cost_usd.toFixed(4)}`;
  }

  const tokenReport = {
    context_tokens: typeof tr.context_tokens === 'number' ? tr.context_tokens : 0,
    budget: typeof tr.budget === 'number' ? tr.budget : 4000,
    utilization: typeof tr.utilization === 'number' ? tr.utilization : 0,
    provider_input_tokens: typeof tr.provider_input_tokens === 'number' ? tr.provider_input_tokens : null,
    provider_output_tokens: typeof tr.provider_output_tokens === 'number' ? tr.provider_output_tokens : null,
    provider_total_tokens: typeof tr.provider_total_tokens === 'number' ? tr.provider_total_tokens : null,
    exact_count_tokens: typeof tr.exact_count_tokens === 'number' ? tr.exact_count_tokens : null,
    estimated_cost_usd: typeof tr.estimated_cost_usd === 'number' ? tr.estimated_cost_usd : null,
    is_free_tier: isFreeTier,
    cost_display: costDisplay,
    provider: tr.provider || root.provider || null,
    model: tr.model || root.model || null,
    usage_estimated: Boolean(tr.usage_estimated || root.usage_estimated),
    files_in_repo: typeof tr.files_in_repo === 'number' ? tr.files_in_repo : 0,
    files_in_context: typeof tr.files_in_context === 'number' ? tr.files_in_context : 0,
    retrieval_seconds: typeof tr.retrieval_seconds === 'number' ? tr.retrieval_seconds : 0,
    context_build_seconds: typeof tr.context_build_seconds === 'number' ? tr.context_build_seconds : 0,
    llm_seconds: typeof tr.llm_seconds === 'number' ? tr.llm_seconds : 0,
    total_seconds: typeof tr.total_seconds === 'number' ? tr.total_seconds : 0,
    key_source: tr.key_source || root.key_source || null,
  };

  let answer = null;
  if (root.answer) {
    const ans = root.answer;
    const rawChanges = Array.isArray(ans.changes) ? ans.changes : [];
    answer = {
      analysis: String(ans.analysis || ''),
      needs_context: Array.isArray(ans.needs_context) ? ans.needs_context.map(String) : [],
      changes: rawChanges.map(ch => ({
        qualified_name: String(ch?.qualified_name || 'Replacement Code'),
        file_path: String(ch?.file_path || ''),
        code: String(ch?.code || ''),
        syntax_valid: Boolean(ch?.syntax_valid),
        unknown_target: Boolean(ch?.unknown_target),
      })),
    };
  }

  let completionText = null;
  if (root.completion) {
    if (typeof root.completion === 'string') {
      completionText = root.completion;
    } else if (typeof root.completion === 'object' && typeof root.completion.text === 'string') {
      completionText = root.completion.text;
    } else if (typeof root.completion === 'object' && root.completion.text !== undefined) {
      completionText = String(root.completion.text || '');
    } else if (typeof root.completion === 'object') {
      completionText = JSON.stringify(root.completion, null, 2);
    } else {
      completionText = String(root.completion);
    }
  }

  return {
    query,
    dry_run: dryRun,
    provider: root.provider || tr.provider || null,
    model: root.model || tr.model || null,
    usage_estimated: Boolean(root.usage_estimated || tr.usage_estimated),
    system_prompt: systemPrompt,
    messages,
    context: root.context ? normalizeContext(root.context) : null,
    trace: normalizeTrace(root.trace),
    token_report: tokenReport,
    answer,
    completion: completionText,
    raw: rawPayload,
  };
}

export function normalizeBenchmarkOne(rawPayload) {
  const root = rawPayload || {};
  return {
    query: String(root.query || ''),
    whole_file_tokens: typeof root.whole_file_tokens === 'number' ? root.whole_file_tokens : 0,
    graph_tokens: typeof root.graph_tokens === 'number' ? root.graph_tokens : 0,
    reduction_pct: typeof root.reduction_pct === 'number' ? root.reduction_pct : 0,
    files_inspected: Array.isArray(root.files_inspected) ? root.files_inspected.map(String) : [],
    counter: String(root.counter || 'tiktoken'),
    raw: rawPayload,
  };
}

export function normalizeBenchmarkResult(rawPayload) {
  const root = rawPayload || {};
  const rawSummaries = Array.isArray(root.summaries)
    ? root.summaries
    : (root.summaries && typeof root.summaries === 'object' ? Object.values(root.summaries) : []);
  const summaries = rawSummaries.map(s => {
    const sm = s || {};
    const redMean = sm.mean_of_reductions_pct !== undefined ? sm.mean_of_reductions_pct : sm.reduction_pct_mean;
    const redAgg = sm.reduction_of_means_pct !== undefined ? sm.reduction_of_means_pct : sm.reduction_pct_aggregate;
    return {
      strategy_name: String(sm.strategy_name || 'unknown'),
      query_count: typeof sm.query_count === 'number' ? sm.query_count : 0,
      mean_context_tokens: typeof sm.mean_context_tokens === 'number' ? sm.mean_context_tokens : 0,
      median_context_tokens: typeof sm.median_context_tokens === 'number' ? sm.median_context_tokens : 0,
      mean_files_inspected: typeof sm.mean_files_inspected === 'number' ? sm.mean_files_inspected : 0,
      mean_build_latency_seconds: typeof sm.mean_build_latency_seconds === 'number' ? sm.mean_build_latency_seconds : 0,
      over_limit_count: typeof sm.over_limit_count === 'number' ? sm.over_limit_count : 0,
      mean_coverage_pct: typeof sm.mean_coverage_pct === 'number' ? sm.mean_coverage_pct : 0,
      mean_of_reductions_pct: typeof redMean === 'number' ? redMean : 0,
      reduction_of_means_pct: typeof redAgg === 'number' ? redAgg : 0,
      mean_real_input_tokens: typeof sm.mean_real_input_tokens === 'number' ? sm.mean_real_input_tokens : null,
      mean_real_output_tokens: typeof sm.mean_real_output_tokens === 'number' ? sm.mean_real_output_tokens : null,
      mean_ai_latency_seconds: typeof sm.mean_ai_latency_seconds === 'number' ? sm.mean_ai_latency_seconds : null,
      mean_cost_usd: typeof sm.mean_cost_usd === 'number' ? sm.mean_cost_usd : null,
      answered_rate: typeof sm.answered_rate === 'number' ? sm.answered_rate : null,
      needs_context_rate: typeof sm.needs_context_rate === 'number' ? sm.needs_context_rate : null,
    };
  });

  return {
    repo_path: String(root.repo_path || ''),
    repo_file_count: typeof root.repo_file_count === 'number' ? root.repo_file_count : 0,
    repo_total_tokens: typeof root.repo_total_tokens === 'number' ? root.repo_total_tokens : 0,
    model_name: String(root.model_name || ''),
    counter_used: String(root.counter_used || 'tiktoken'),
    budget: typeof root.budget === 'number' ? root.budget : 4000,
    call_ai: Boolean(root.call_ai),
    repeat: typeof root.repeat === 'number' ? root.repeat : 1,
    summaries,
    rows: Array.isArray(root.rows) ? root.rows : [],
    raw: rawPayload,
  };
}

export function normalizeStats(rawPayload) {
  const root = rawPayload || {};
  const db = root.database || {};
  const graph = root.graph || {};
  const emb = root.embeddings || {};
  return {
    repo_path: String(root.repo_path || ''),
    is_stale: Boolean(root.is_stale),
    stale_reasons: Array.isArray(root.stale_reasons) ? root.stale_reasons.map(String) : [],
    database: {
      files: typeof db.files === 'number' ? db.files : 0,
      functions: typeof db.functions === 'number' ? db.functions : 0,
      classes: typeof db.classes === 'number' ? db.classes : 0,
      methods: typeof db.methods === 'number' ? db.methods : 0,
      calls: typeof db.calls === 'number' ? db.calls : 0,
      files_with_errors: typeof db.files_with_errors === 'number' ? db.files_with_errors : 0,
    },
    graph: {
      nodes: typeof graph.nodes === 'number' ? graph.nodes : 0,
      edges: typeof graph.edges === 'number' ? graph.edges : 0,
      resolved_calls: typeof graph.resolved_calls === 'number' ? graph.resolved_calls : null,
      resolution_rate: typeof graph.resolution_rate === 'number' ? graph.resolution_rate : null,
    },
    embeddings: {
      total_items: typeof emb.total_items === 'number' ? emb.total_items : 0,
      model_name: String(emb.model_name || 'all-MiniLM-L6-v2'),
      dimension: typeof emb.dimension === 'number' ? emb.dimension : 384,
    },
    raw: rawPayload,
  };
}

export function normalizeGraphView(rawPayload) {
  const root = rawPayload || {};
  const rawNodes = Array.isArray(root.nodes) ? root.nodes : [];
  const rawEdges = Array.isArray(root.edges) ? root.edges : [];
  return {
    nodes: rawNodes.map(n => ({
      id: String(n?.id || ''),
      label: String(n?.label || n?.id || ''),
      type: String(n?.type || 'external'),
      file_path: String(n?.file_path || ''),
      start_line: typeof n?.start_line === 'number' ? n.start_line : null,
      end_line: typeof n?.end_line === 'number' ? n.end_line : null,
    })),
    edges: rawEdges.map(e => ({
      source: String(e?.source || ''),
      target: String(e?.target || ''),
      relationship: String(e?.relationship || 'DEPENDS_ON'),
      weight: typeof e?.weight === 'number' ? e.weight : 1.0,
    })),
    stats: {
      total_nodes: root.stats && typeof root.stats.total_nodes === 'number' ? root.stats.total_nodes : rawNodes.length,
      total_edges: root.stats && typeof root.stats.total_edges === 'number' ? root.stats.total_edges : rawEdges.length,
    },
    raw: rawPayload,
  };
}
