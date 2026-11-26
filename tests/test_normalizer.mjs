/**
 * Frontend unit tests for normalizer and UI error banner behavior.
 * Executable directly via Node.js: node tests/test_normalizer.mjs
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import {
  normalizeRetrievalResult,
  normalizeAskResult,
  normalizeContext,
  normalizeCandidate,
  normalizeBenchmarkOne,
  normalizeBenchmarkResult,
  normalizeStats,
} from '../web/normalizer.js';
import { formatErrorMessage, renderUiError } from '../web/app.js';

// Load shared contract paths specification
const contractPathsJson = JSON.parse(
  fs.readFileSync(path.resolve('web/contract_paths.json'), 'utf-8')
);

test('Normalizer: (a) full valid response', () => {
  const fullPayload = {
    retrieval: {
      query: 'optimize session',
      intent: 'optimize',
      target: 'src.requests.sessions.session',
      target_confidence: 'high',
      candidates: [
        {
          ref: 'src.requests.sessions.session',
          score: 0.95,
          entity_type: 'function',
          file: 'src/requests/sessions.py',
          start_line: 951,
          end_line: 958,
          matched_kinds: ['docstring'],
          match_type: 'keyword',
        },
      ],
    },
    context: {
      text: 'TARGET: src.requests.sessions.session',
      tokens: 1427,
      budget_total: 4000,
      utilization: 0.3568,
      items: [
        {
          ref: 'src.requests.sessions.session',
          file: 'src/requests/sessions.py',
          lines: '951-958',
          level: 'FULL',
          tokens: 120,
          role: 'target',
          score: 0.95,
          reason: 'Primary target',
        },
      ],
      files_included: ['src/requests/sessions.py'],
    },
  };

  const norm = normalizeRetrievalResult(fullPayload);
  assert.equal(norm.query, 'optimize session');
  assert.equal(norm.intent, 'optimize');
  assert.equal(norm.target, 'src.requests.sessions.session');
  assert.equal(norm.target_confidence, 'HIGH');
  assert.equal(norm.candidates.length, 1);
  assert.equal(norm.candidates[0].is_target, true);
  assert.equal(norm.candidates[0].ref, 'src.requests.sessions.session');
  assert.equal(norm.context.tokens, 1427);
  assert.equal(norm.context.level_counts.FULL, 1);
});

test('Normalizer: (b) response missing the analyzed-query object', () => {
  // Real API does not return analyzed_query; intent is at top level
  const payloadWithoutAnalyzedQuery = {
    retrieval: {
      query: 'optimize session',
      target: 'src.requests.sessions.session',
      target_confidence: 'medium',
      candidates: [
        {
          ref: 'src.requests.sessions.session',
          score: 0.85,
          entity_type: 'function',
          file: 'src/requests/sessions.py',
          match_type: 'semantic',
        },
      ],
      intent: 'optimize',
    },
  };

  // Must not throw, must produce default intent and identifiers
  const norm = normalizeRetrievalResult(payloadWithoutAnalyzedQuery);
  assert.equal(norm.intent, 'optimize');
  assert.deepEqual(norm.identifiers, []);
  assert.equal(norm.scope, null);
  assert.equal(norm.target, 'src.requests.sessions.session');
});

test('Normalizer: (c) structured error response passed directly', () => {
  const errorPayload = {
    error: {
      code: 'provider_not_configured',
      message: 'Provider is not configured with an API key',
      request_id: 'err-1234',
    },
  };

  // When error response is fed into normalizers, neither throws
  const normRet = normalizeRetrievalResult(errorPayload);
  assert.equal(normRet.intent, 'general');
  assert.equal(normRet.target, null);
  assert.deepEqual(normRet.candidates, []);

  const normAsk = normalizeAskResult(errorPayload);
  assert.equal(normAsk.dry_run, false);
  assert.equal(normAsk.answer, null);
  assert.equal(normAsk.token_report.context_tokens, 0);
});

test('Normalizer: (d) empty candidates list', () => {
  const emptyPayload = {
    retrieval: {
      query: 'unknown_symbol_xyz',
      candidates: [],
    },
    context: null,
  };

  const norm = normalizeRetrievalResult(emptyPayload);
  assert.equal(norm.candidates.length, 0);
  assert.equal(norm.target, null);
  assert.equal(norm.intent, 'general');
  assert.equal(norm.context, null);
});

test('Normalizer: (e) dry-run ask response with no completion', () => {
  const dryRunPayload = {
    query: 'optimize session',
    dry_run: true,
    system_prompt: 'System prompt content',
    messages: [{ role: 'user', content: 'User task' }],
    context: {
      text: 'Prompt context text',
      tokens: 950,
      budget_total: 4000,
    },
    completion: null,
    answer: null,
    token_report: {
      context_tokens: 950,
      budget: 4000,
      utilization: 0.2375,
      files_in_repo: 37,
      files_in_context: 2,
      total_seconds: 0.05,
    },
  };

  const norm = normalizeAskResult(dryRunPayload);
  assert.equal(norm.dry_run, true);
  assert.equal(norm.completion, null);
  assert.equal(norm.answer, null);
  assert.equal(norm.system_prompt, 'System prompt content');
  assert.equal(norm.token_report.context_tokens, 950);
  assert.equal(norm.context.tokens, 950);
});

test('Normalizer: (f) ask response with completion object extracts text string', () => {
  const askPayload = {
    query: 'explain auth',
    dry_run: false,
    completion: {
      text: '### ANALYSIS\nAuthentication is implemented in auth.py',
      usage: { input_tokens: 100, output_tokens: 50, total_tokens: 150 },
      model: 'gemini-2.5-flash',
    },
    answer: {
      analysis: 'Authentication is implemented in auth.py',
      changes: [],
    },
    token_report: {
      context_tokens: 200,
      budget: 4000,
      total_seconds: 1.2,
    },
  };

  const norm = normalizeAskResult(askPayload);
  assert.equal(norm.dry_run, false);
  assert.equal(norm.completion, '### ANALYSIS\nAuthentication is implemented in auth.py');
  assert.notEqual(norm.completion, '[object Object]');
  assert.equal(norm.answer.analysis, 'Authentication is implemented in auth.py');
});

test('Contract Path Verification: shared required paths', () => {
  // Confirm required contract paths are verified
  assert.ok(contractPathsJson.retrieve.includes('retrieval.intent'));
  assert.ok(contractPathsJson.retrieve.includes('retrieval.target'));
  assert.ok(contractPathsJson.retrieve.includes('context.tokens'));
  assert.ok(contractPathsJson.ask_dry_run.includes('token_report.context_tokens'));
});

test('Error Banner: shows request id and error code for structured error response', () => {
  const errorObj = {
    code: 'query_too_long',
    requestId: 'abc12345',
    message: 'Query exceeds maximum character budget',
  };

  const formatted = formatErrorMessage('API Error', errorObj);
  assert.ok(formatted.includes('code: query_too_long'), 'Must include error code');
  assert.ok(formatted.includes('request id: abc12345'), 'Must include request id');
  assert.ok(formatted.includes('Query exceeds maximum character budget'), 'Must include error message');
});

test('Component Boundary: shows "UI error" wording and debug section on rendering exception', () => {
  const mockContainer = { innerHTML: '' };
  const mockError = new TypeError("Cannot read properties of undefined (reading 'dummy')");
  const rawResponse = { status: 200, someKey: 'val' };

  renderUiError(mockContainer, 'ResultsTable', mockError, rawResponse);

  assert.ok(mockContainer.innerHTML.includes('Could not display results (UI error):'), 'Must state UI error wording');
  assert.ok(!mockContainer.innerHTML.includes('Retrieval failed:'), 'Must NOT use misleading "Retrieval failed" prefix');
  assert.ok(mockContainer.innerHTML.includes('Debug: raw response'), 'Must offer collapsible debug raw response');
  assert.ok(mockContainer.innerHTML.includes('someKey'), 'Must serialize raw response in debug pre');
});
