/**
 * Non-blocking notifications and plain-language error messages.
 */

const HINTS = {
  storage_unavailable: 'The database is not reachable. Check MONGODB_URI in your .env file and that MongoDB is running, or remove MONGODB_URI to use temporary in-memory storage.',
  api_key_required: 'This needs an AI provider key. Open "Model & Key" at the top and paste one, or turn on "Preview only" in Retrieval Settings.',
  api_key_invalid: 'The API key was rejected by the provider. Open "Model & Key" and check it.',
  prerequisite_missing: 'This repository has not been indexed yet. Click "Update Index" or "Rebuild" first.',
  indexing_in_progress: 'Indexing is still running. Give it a moment and try again.',
  rate_limit_exceeded: 'Too many requests in a short time. Wait a minute and try again.',
  QUERY_TOO_LONG: 'That message is too long. Shorten it and try again.',
  BUDGET_TOO_SMALL: 'The token budget is too small for this request. Raise it in Retrieval Settings.',
};

/** Turn an API error into a short sentence a non-developer can act on. */
export function friendlyError(err) {
  if (!err) return 'Something went wrong.';
  const raw = err.message || String(err);
  if (/Failed to fetch|NetworkError|Load failed/i.test(raw)) {
    return 'Cannot reach the server. Make sure the app is still running, then retry.';
  }
  const hint = err.code && HINTS[err.code];
  return hint ? `${hint} (${raw})` : raw;
}

function classify(message) {
  if (/fail|cannot|could not|unable|error|not reachable/i.test(message)) return 'error';
  if (/success|complete|done|reindexed|uploaded/i.test(message) && !/warning|secret/i.test(message)) return 'success';
  if (/warning|secret|please|only|first/i.test(message)) return 'warn';
  return 'info';
}

function host() {
  let el = document.getElementById('toast-host');
  if (!el) {
    el = document.createElement('div');
    el.id = 'toast-host';
    el.className = 'toast-host';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    document.body.appendChild(el);
  }
  return el;
}

/** Show a dismissible toast. The kind is inferred from the text when omitted. */
export function notify(message, kind) {
  const text = String(message);
  const toast = document.createElement('div');
  toast.className = `toast toast-${kind || classify(text)}`;
  const body = document.createElement('div');
  body.className = 'toast-body';
  body.textContent = text;
  const close = document.createElement('button');
  close.className = 'toast-close';
  close.setAttribute('aria-label', 'Dismiss');
  close.textContent = '×';
  const remove = () => toast.remove();
  close.addEventListener('click', remove);
  toast.append(body, close);
  host().appendChild(toast);
  setTimeout(remove, toast.classList.contains('toast-error') ? 12000 : 6000);
}
