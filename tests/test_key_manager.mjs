/**
 * Frontend unit tests for key_manager.js Bring-Your-Own-Key behavior.
 * Executable directly via Node.js: node tests/test_key_manager.mjs
 */

import test from 'node:test';
import assert from 'node:assert/strict';

// Mock browser storage environments for Node.js
class MockStorage {
  constructor() {
    this.store = new Map();
  }
  getItem(key) {
    return this.store.has(key) ? this.store.get(key) : null;
  }
  setItem(key, value) {
    this.store.set(key, String(value));
  }
  removeItem(key) {
    this.store.delete(key);
  }
  clear() {
    this.store.clear();
  }
}

globalThis.sessionStorage = new MockStorage();
globalThis.localStorage = new MockStorage();

import { keyManager } from '../web/key_manager.js';

test('KeyManager: sessionStorage by default', () => {
  sessionStorage.clear();
  localStorage.clear();

  keyManager.setKey('gemini', 'AIzaSyTestGeminiKey12345', false);

  assert.equal(sessionStorage.getItem('cce_key_gemini'), 'AIzaSyTestGeminiKey12345');
  assert.equal(localStorage.getItem('cce_key_gemini'), null);
  assert.equal(keyManager.getKey('gemini'), 'AIzaSyTestGeminiKey12345');
  assert.equal(keyManager.isRemembered('gemini'), false);
});

test('KeyManager: localStorage only when remembered', () => {
  sessionStorage.clear();
  localStorage.clear();

  keyManager.setKey('claude', 'sk-ant-test-claude-key-9999', true);

  assert.equal(localStorage.getItem('cce_key_claude'), 'sk-ant-test-claude-key-9999');
  assert.equal(sessionStorage.getItem('cce_key_claude'), null);
  assert.equal(keyManager.getKey('claude'), 'sk-ant-test-claude-key-9999');
  assert.equal(keyManager.isRemembered('claude'), true);
});

test('KeyManager: Clear removes from both sessionStorage and localStorage', () => {
  sessionStorage.setItem('cce_key_gemini', 'temp_key');
  localStorage.setItem('cce_key_gemini', 'old_key');
  localStorage.setItem('cce_remember_gemini', 'true');

  keyManager.clearKey('gemini');

  assert.equal(sessionStorage.getItem('cce_key_gemini'), null);
  assert.equal(localStorage.getItem('cce_key_gemini'), null);
  assert.equal(localStorage.getItem('cce_remember_gemini'), null);
  assert.equal(keyManager.getKey('gemini'), null);
  assert.equal(keyManager.hasKey('gemini'), false);
});

test('KeyManager: Per-provider separation', () => {
  sessionStorage.clear();
  localStorage.clear();

  keyManager.setKey('gemini', 'gemini-key-111', false);
  keyManager.setKey('claude', 'claude-key-222', true);

  assert.equal(keyManager.getKey('gemini'), 'gemini-key-111');
  assert.equal(keyManager.getKey('claude'), 'claude-key-222');
  assert.equal(keyManager.getKey('groq'), null);

  keyManager.clearKey('gemini');
  assert.equal(keyManager.getKey('gemini'), null);
  assert.equal(keyManager.getKey('claude'), 'claude-key-222');
});

test('KeyManager: Masked display formatting', () => {
  const shortKey = '1234567';
  assert.equal(keyManager.maskKey(shortKey), '••••••••');

  const fullKey = 'AIzaSyBN1234567890abcdefXYZ9';
  const masked = keyManager.maskKey(fullKey);
  assert.equal(masked.startsWith('AIza'), true);
  assert.equal(masked.endsWith('XYZ9'), true);
  assert.equal(masked.includes('••••••••'), true);
  assert.equal(masked.includes('1234567890'), false); // middle characters masked
});

test('KeyManager: Redact sensitive text', () => {
  const rawSecret = 'sk-ant-api03-very-secret-token-abcdef1234';
  const sampleLog = `Request failed: Unauthorized for token ${rawSecret} in auth header`;
  const redacted = keyManager.redactText(sampleLog, rawSecret);

  assert.equal(redacted.includes(rawSecret), false);
  assert.equal(redacted.includes('[REDACTED_KEY]'), true);

  const geminiPattern = 'Call error with key AIzaSyD12345678901234567890123456789012 on endpoint';
  const redactedGemini = keyManager.redactText(geminiPattern);
  assert.equal(redactedGemini.includes('AIzaSyD'), false);
  assert.equal(redactedGemini.includes('[REDACTED_KEY]'), true);
});
