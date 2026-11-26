/**
 * Client-side Key Manager module for Bring-Your-Own-Key support.
 *
 * Security & Storage Rules:
 * 1. Default storage is sessionStorage (wiped on tab close).
 * 2. If "Remember on this device" is checked, localStorage is used.
 * 3. Keys are stored per-provider ('cce_key_<provider>').
 * 4. Never print raw keys to console or expose in DOM value attributes after masking.
 * 5. Full key is never repopulated into the DOM input field once saved/masked.
 */

const STORAGE_PREFIX = 'cce_key_';
const REMEMBER_PREFIX = 'cce_remember_';

export const keyManager = {
  /**
   * Check if "Remember on this device" is enabled for this provider.
   */
  isRemembered(provider) {
    if (!provider) return false;
    try {
      return localStorage.getItem(`${REMEMBER_PREFIX}${provider}`) === 'true';
    } catch (_) {
      return false;
    }
  },

  /**
   * Set remember preference for this provider.
   */
  setRemembered(provider, remember) {
    if (!provider) return;
    try {
      if (remember) {
        localStorage.setItem(`${REMEMBER_PREFIX}${provider}`, 'true');
      } else {
        localStorage.removeItem(`${REMEMBER_PREFIX}${provider}`);
      }
    } catch (_) {}
  },

  /**
   * Get the stored raw key for a provider.
   * Checks localStorage if remember is enabled, otherwise sessionStorage.
   */
  getKey(provider) {
    if (!provider) return null;
    try {
      if (this.isRemembered(provider)) {
        return localStorage.getItem(`${STORAGE_PREFIX}${provider}`) || null;
      }
      return sessionStorage.getItem(`${STORAGE_PREFIX}${provider}`) || null;
    } catch (_) {
      return null;
    }
  },

  /**
   * Store a key for a provider.
   * If remember is true: saves in localStorage and clears from sessionStorage.
   * If remember is false: saves in sessionStorage and clears from localStorage.
   */
  setKey(provider, key, remember = false) {
    if (!provider || !key) return;
    const cleanKey = String(key).trim();
    if (!cleanKey) return;

    this.setRemembered(provider, remember);

    try {
      if (remember) {
        localStorage.setItem(`${STORAGE_PREFIX}${provider}`, cleanKey);
        sessionStorage.removeItem(`${STORAGE_PREFIX}${provider}`);
      } else {
        sessionStorage.setItem(`${STORAGE_PREFIX}${provider}`, cleanKey);
        localStorage.removeItem(`${STORAGE_PREFIX}${provider}`);
      }
    } catch (_) {}
  },

  /**
   * Remove key from both sessionStorage and localStorage for a provider.
   */
  clearKey(provider) {
    if (!provider) return;
    try {
      sessionStorage.removeItem(`${STORAGE_PREFIX}${provider}`);
      localStorage.removeItem(`${STORAGE_PREFIX}${provider}`);
      localStorage.removeItem(`${REMEMBER_PREFIX}${provider}`);
    } catch (_) {}
  },

  /**
   * Return whether a key exists in storage for this provider.
   */
  hasKey(provider) {
    return Boolean(this.getKey(provider));
  },

  /**
   * Return a masked display version of a key (at most first 4 and last 4 characters).
   * Example: 'AIza••••••••1234' or '••••••••'
   */
  maskKey(key) {
    if (!key) return '';
    const str = String(key).trim();
    if (str.length <= 8) {
      return '••••••••';
    }
    const prefix = str.slice(0, 4);
    const suffix = str.slice(-4);
    return `${prefix}••••••••${suffix}`;
  },

  /**
   * Redact key strings and key patterns from arbitrary text before display.
   */
  redactText(text, currentKey = null) {
    if (!text || typeof text !== 'string') return text;
    let out = text;
    if (currentKey && currentKey.length >= 4 && out.includes(currentKey)) {
      out = out.split(currentKey).join('[REDACTED_KEY]');
    }
    // Redact common key patterns
    out = out.replace(/sk-[a-zA-Z0-9_\-]{14,}/gi, '[REDACTED_KEY]');
    out = out.replace(/AIza[0-9A-Za-z\-_]{35}/g, '[REDACTED_KEY]');
    out = out.replace(/AQ\.[0-9A-Za-z\-_]{20,}/g, '[REDACTED_KEY]');
    out = out.replace(/gsk_[a-zA-Z0-9_\-]{14,}/g, '[REDACTED_KEY]');
    return out;
  },
};
