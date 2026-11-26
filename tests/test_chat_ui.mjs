/**
 * Frontend unit tests for chat UI components, markdown formatting,
 * file references, diff cards, token savings panel, and Phase 0 error states.
 * Executable directly via Node.js: node tests/test_chat_ui.mjs
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import {
  escapeHtml,
  renderAssistantContent,
  renderDiffCard,
  renderSavingsHtml,
} from '../web/chat.js';
import { formatErrorMessage } from '../web/app.js';

test('Chat UI: Markdown code block formatting and Copy button', () => {
  const msg = {
    id: 1,
    role: 'assistant',
    content: 'Here is the helper function:\n```python\ndef greet():\n    return "hello"\n```\nEnjoy!',
  };
  const html = renderAssistantContent(msg);

  assert.ok(html.includes('class="msg-code-block"'), 'Renders code block wrapper');
  assert.ok(html.includes('btn-copy-code'), 'Renders Copy button');
  assert.ok(html.includes('def greet():'), 'Preserves code content');
  assert.ok(html.includes('python'), 'Includes language badge');
});

test('Chat UI: File reference citations (path:line) converted to clickable spans', () => {
  const msg = {
    id: 2,
    role: 'assistant',
    content: 'Check the implementation in requests/sessions.py:15-30 and auth/login.py:42.',
  };
  const html = renderAssistantContent(msg);

  assert.ok(html.includes('class="file-ref-link"'), 'Transforms file:line into clickable link');
  assert.ok(html.includes('data-path="requests/sessions.py"'), 'Stores data-path attribute');
  assert.ok(html.includes('data-start="15"'), 'Stores data-start attribute');
  assert.ok(html.includes('data-end="30"'), 'Stores data-end attribute');
  assert.ok(html.includes('auth/login.py:42'), 'Matches second single-line reference');
});

test('Chat UI: NEEDS_CONTEXT chips rendered as interactive pin chips', () => {
  const msg = {
    id: 3,
    role: 'assistant',
    content: 'I need more information to answer.\nNEEDS_CONTEXT: requests.sessions.Session.send\nNEEDS_CONTEXT: requests.adapters.HTTPAdapter',
  };
  const html = renderAssistantContent(msg);

  assert.ok(html.includes('needs-context-chip-row'), 'Renders chip row container');
  assert.ok(html.includes('needs-context-chip'), 'Renders chip buttons');
  assert.ok(html.includes('data-target="requests.sessions.Session.send"'), 'Extracts target entity');
  assert.ok(html.includes('Pin requests.sessions.Session.send to context'), 'Chip label prompts user');
});

test('Chat UI: AI change proposals render unified diff cards', () => {
  const msg = {
    id: 4,
    role: 'assistant',
    content: 'Optimizing timeout check:\n### CHANGE: Session.send (requests/sessions.py)\n```python\ndef send(self, request, **kwargs):\n    timeout = kwargs.get("timeout", 30)\n    return super().send(request, **kwargs)\n```',
  };
  const html = renderAssistantContent(msg);

  assert.ok(html.includes('diff-card'), 'Renders diff card');
  assert.ok(html.includes('btn-apply-diff'), 'Renders Apply to Workspace Copy button');
  assert.ok(html.includes('data-symbol="Session.send"'), 'Tags target symbol');
  assert.ok(html.includes('data-path="requests/sessions.py"'), 'Tags target file path');
});

test('Chat UI: Diff card component direct generation', () => {
  const cardHtml = renderDiffCard('modify', 'login_user', 'auth/login.py', 'def login_user(): return True', 0, 99);
  assert.ok(cardHtml.includes('diff-card-99-0'), 'Generates card id');
  assert.ok(cardHtml.includes('auth/login.py'), 'Contains file path');
  assert.ok(cardHtml.includes('btn-apply-diff'), 'Has apply button');
  assert.ok(cardHtml.includes('btn-undo-diff'), 'Has undo button placeholder');
});

test('Chat UI: Savings panel renders correctly with complete metrics', () => {
  const tokens = {
    actual_total_sent: 2400,
    actual_context_tokens: 1800,
    budget: 4000,
    whole_files_baseline: 12500,
    whole_files_saved_pct: 81,
    conversation_baseline: 4200,
    full_repo_baseline: 45000,
    full_repo_saved_pct: 94.7,
  };
  const html = renderSavingsHtml(tokens);

  assert.ok(html.includes('2,400'), 'Displays formatted actual sent tokens');
  assert.ok(html.includes('1,800 repo context'), 'Displays context breakdown');
  assert.ok(html.includes('4,000 budget'), 'Displays budget');
  assert.ok(html.includes('94.7%'), 'Displays measured full-repo savings');
  assert.ok(html.includes('12,500 tokens') && html.includes('81% saved'), 'Displays whole-files baseline and savings');
  assert.ok(html.includes('45,000 tokens'), 'Displays full repo scale baseline');
  assert.ok(!html.includes('43%'), 'Never invents a conversation savings figure');
  assert.ok(html.includes('btn-retry-more-context'), 'Provides Retry with More Context action');
});

test('Chat UI: Savings panel handles empty/missing data gracefully without crashing', () => {
  const htmlNull = renderSavingsHtml(null);
  assert.ok(htmlNull.includes('No savings data recorded yet'), 'Gracefully handles null token report');

  const htmlPartial = renderSavingsHtml({});
  assert.ok(htmlPartial.includes('Tokens sent this turn'), 'Renders structure even with empty payload');
});

test('Chat UI: Phase 0 Error banner formatting ensures visible request id and error code', () => {
  const err = {
    code: 'NOT_FOUND',
    requestId: 'req-abc-123',
    message: 'Requested workspace copy does not exist',
  };
  const formatted = formatErrorMessage('Workspace lookup failed', err);
  assert.ok(formatted.includes('code: NOT_FOUND'), 'Shows error code');
  assert.ok(formatted.includes('request id: req-abc-123'), 'Shows request id');
  assert.ok(formatted.includes('Requested workspace copy does not exist'), 'Shows clear message');
});

test('Trace panel: explains graph walk, signals, operations and savings', async () => {
  const { normalizeTrace } = await import('../web/normalizer.js');
  const { renderTraceHtml, renderTraceSummaryLine } = await import('../web/components/trace_panel.js');

  const trace = normalizeTrace({
    intent: 'optimize',
    target: 'auth.login.login_user',
    target_confidence: 'high',
    target_reason: "'login_user' in the query is an exact function name",
    signals: { semantic: 'used', keyword: 'used', graph: 'used', other: 'skipped: disabled by request' },
    hits: [{ ref: 'utils.hashing.verify_password', signal: 'graph_callee', score: 0.85, file: 'utils/hashing.py', note: '' }],
    graph_edges: [{ source: 'auth.login.login_user', target: 'utils.hashing.verify_password', rel: 'CALLS', direction: 'callee', depth: 1, confidence: 'exact', outcome: 'added' }],
    graph_nodes: 14,
    graph_edges_total: 19,
    candidates_considered: 10,
    candidates_returned: 8,
    operations: [
      { step: 1, name: 'intent_detection', status: 'ran', detail: "Query classified as 'optimize'", seconds: 0 },
      { step: 2, name: 'graph_expansion', status: 'ran', detail: 'Walked 1 CALLS edges', count: 1, seconds: 0.0004 },
    ],
    selection: {
      candidates_in: 8,
      included: [{ ref: 'auth.login.login_user', file: 'auth/login.py', lines: '4-9', level: 'full', tokens: 65, original_tokens: 65, role: 'target', score: 1, reason: '' }],
      omitted: [{ ref: 'x.y.z', reason: 'budget_exceeded' }],
      downgraded_count: 0,
    },
    savings: {
      counter: 'tiktoken', is_estimate: false, context_tokens: 296, prompt_tokens: 301,
      naive_full_repo_tokens: 9000, whole_files_tokens: 1200, repo_files: 40, context_files: 3,
      saved_vs_full_repo_tokens: 8699, saved_vs_full_repo_pct: 96.7, saved_vs_whole_files_tokens: 899,
      saved_vs_whole_files_pct: 74.9, compression_saved_tokens: 0, reduction_factor: 29.9,
    },
    warnings: [],
  });

  const html = renderTraceHtml(trace);
  assert.ok(html.includes('96.7%'), 'Shows savings headline');
  assert.ok(html.includes('9,000'), 'Shows naive full-repo tokens');
  assert.ok(html.includes('auth.login.login_user'), 'Shows target');
  assert.ok(html.includes('CALLS') && html.includes('utils.hashing.verify_password'), 'Lists walked graph edges');
  assert.ok(html.includes('Code graph expansion'), 'Names operations in plain words');
  assert.ok(html.includes('did not fit the token budget'), 'Explains pruned items');
  assert.ok(html.includes('signal-skipped'), 'Marks skipped signals');

  const line = renderTraceSummaryLine(trace);
  assert.ok(line.includes('1 graph edge walked') && line.includes('96.7% fewer tokens'), 'Summary line digest');

  const tiny = normalizeTrace({ savings: { ...trace.savings, saved_vs_full_repo_pct: -20, naive_full_repo_tokens: 100 }, signals: {}, operations: [] });
  assert.ok(renderTraceHtml(tiny).includes('Savings grow with repository size'), 'Is honest when there are no savings');
});

test('benchmark normalizer accepts summaries keyed by strategy', async () => {
  const { normalizeBenchmarkResult } = await import('../web/normalizer.js');
  const vm = normalizeBenchmarkResult({ summaries: { graph: { strategy_name: 'graph', mean_context_tokens: 10 }, whole_repo: { strategy_name: 'whole_repo', mean_context_tokens: 100 } } });
  assert.equal(vm.summaries.length, 2);
  assert.equal(vm.summaries[0].strategy_name, 'graph');
});
