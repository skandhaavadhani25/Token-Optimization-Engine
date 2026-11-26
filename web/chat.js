/**
 * Chat Experience Controller: Workspaces, Conversations, Attachments,
 * Upload pipeline, AI Diffs, Undo, and Token Savings Panel.
 */

import { api } from './api.js';
import { notify, friendlyError } from './toast.js';
import { keyManager } from './key_manager.js';
import { normalizeTrace } from './normalizer.js';
import { renderTraceHtml, renderTraceSummaryLine } from './components/trace_panel.js';

const SAMPLE_PROMPTS = [
  'Give me an overview of how this project is structured',
  'Where is user login handled and what does it call?',
  'Find possible bugs in the main entry point',
  'Add input validation to the most important function',
];

function welcomeHtml() {
  const chips = SAMPLE_PROMPTS.map(p => `<button type="button" class="prompt-chip" data-prompt="${escapeHtml(p)}">${escapeHtml(p)}</button>`).join('');
  return `
    <div class="chat-welcome">
      <span class="welcome-eyebrow">ContextForge &middot; Graph-based context, fewer tokens</span>
      <h3>Ask your codebase anything</h3>
      <p>The app finds only the code that matters for your question, so the model reads far fewer tokens than the whole repo.</p>
      <div class="welcome-steps">
        <div class="welcome-step"><span class="welcome-step-num">1</span><div><strong>Add your code</strong><span>Upload a .zip of your project or open a folder on this computer.</span><button type="button" class="btn btn-primary btn-sm welcome-cta" data-action="open-upload">Open or upload code</button></div></div>
        <div class="welcome-step"><span class="welcome-step-num">2</span><div><strong>Ask a question or request a change</strong><span>Try one of these, or type your own below.</span><div class="prompt-chips">${chips}</div></div></div>
        <div class="welcome-step"><span class="welcome-step-num">3</span><div><strong>See how it worked</strong><span>Click "How?" under any answer to see which functions were chosen and how many tokens were saved.</span></div></div>
      </div>
    </div>`;
}

export function initChat(appState, setStatusCallback, onWorkspaceChanged) {
  const chatState = {
    workspaces: [],
    activeWorkspaceId: null,
    conversations: [],
    activeConversationId: null,
    mode: 'balanced',
    pendingAttachments: [],
    pendingUploadFiles: null,
    currentProposals: [],
    lastTurnTokens: null,
    messages: [],
  };

  // Try restoring active workspace id from localStorage
  try {
    const savedId = localStorage.getItem('cce_active_workspace_id');
    if (savedId) chatState.activeWorkspaceId = savedId;
  } catch (_) {}

  // DOM Elements
  const selectWs = document.getElementById('select-active-workspace');
  const wsCard = document.getElementById('workspace-info-card');
  const wsCountsText = document.getElementById('ws-counts-text');
  const btnWsZip = document.getElementById('btn-ws-zip');
  const btnWsPatch = document.getElementById('btn-ws-patch');
  const btnWsReindex = document.getElementById('btn-ws-reindex');
  const btnWsDelete = document.getElementById('btn-ws-delete');

  const convListEl = document.getElementById('conversation-list');
  const btnNewChat = document.getElementById('btn-new-chat');
  const threadTitleEl = document.getElementById('thread-title');
  const activeRepoBadge = document.getElementById('active-repo-badge');
  const badgeActiveModel = document.getElementById('badge-active-model');

  const messagesScrollEl = document.getElementById('chat-messages-scroll');
  const messagesContainer = document.getElementById('chat-messages-container');
  if (messagesContainer && messagesContainer.querySelector('.chat-welcome')) {
    messagesContainer.innerHTML = welcomeHtml();
  }
  messagesContainer?.addEventListener('click', (e) => {
    const chip = e.target.closest('[data-prompt]');
    if (chip) {
      const input = document.getElementById('chat-input-textarea');
      if (input) {
        input.value = chip.dataset.prompt;
        input.focus();
      }
      if (!appState.activeWorkspaceId) notify('Add your code first (step 1), then send this question.', 'info');
      return;
    }
    if (e.target.closest('[data-action="open-upload"]')) {
      document.getElementById('btn-open-workspace-modal')?.click();
      document.getElementById('tab-ws-upload')?.click();
    }
  });
  const chatMainEl = document.querySelector('.chat-main');
  const chatInputAreaEl = document.querySelector('.chat-input-area');
  const chatInput = document.getElementById('chat-input-textarea');
  const btnSend = document.getElementById('btn-chat-send');

  const btnAttach = document.getElementById('btn-attach-file');
  const btnAttachFolder = document.getElementById('btn-attach-folder');
  const fileInput = document.getElementById('input-chat-file');
  const folderInput = document.getElementById('input-chat-folder');
  const attachmentTray = document.getElementById('attachment-tray');

  const btnToggleSettings = document.getElementById('btn-toggle-chat-settings');
  const settingsPopover = document.getElementById('chat-settings-popover');
  const btnToggleSavings = document.getElementById('btn-toggle-savings-panel');
  const savingsPanel = document.getElementById('chat-panel-savings');
  const btnCloseSavings = document.getElementById('btn-close-savings');
  const savingsContent = document.getElementById('savings-panel-content');

  // Modals
  const modalWs = document.getElementById('modal-workspace');
  const btnOpenWsModal = document.getElementById('btn-open-workspace-modal');
  const btnSidebarNewWs = document.getElementById('btn-sidebar-new-ws');
  const btnCloseWsModal = document.getElementById('btn-close-modal-ws');

  const modalModel = document.getElementById('modal-model');
  const btnOpenModelModal = document.getElementById('btn-open-model-modal');
  const btnCloseModelModal = document.getElementById('btn-close-modal-model');

  const modalViewer = document.getElementById('modal-file-viewer');
  const btnCloseViewer = document.getElementById('btn-close-viewer');
  const viewerTitle = document.getElementById('viewer-file-path');
  const viewerCode = document.getElementById('viewer-code-content');

  // Upload Choice Modal
  const modalUploadChoice = document.getElementById('modal-upload-choice');
  const btnCloseUploadChoice = document.getElementById('btn-close-upload-choice');
  const btnCancelUploadChoice = document.getElementById('btn-cancel-upload-choice');
  const btnConfirmUploadChoice = document.getElementById('btn-confirm-upload-choice');
  const uploadChoiceDesc = document.getElementById('upload-choice-desc');
  const choiceCurrentWsName = document.getElementById('choice-current-ws-name');
  const choiceStartNew = document.getElementById('choice-start-new');
  const choiceAddCurrent = document.getElementById('choice-add-current');
  const choiceAttachChat = document.getElementById('choice-attach-chat');

  // =========================================================================
  // 1. Workspaces Lifecycle
  // =========================================================================

  async function loadWorkspaces() {
    try {
      const list = await api.listWorkspaces();
      chatState.workspaces = Array.isArray(list) ? list : [];
      renderWorkspacesList();
    } catch (err) {
      console.warn('Failed to load workspaces:', err);
    }
  }

  function renderWorkspacesList() {
    if (selectWs) {
      selectWs.innerHTML = '<option value="">(No project open)</option>';
      chatState.workspaces.forEach(ws => {
        const opt = document.createElement('option');
        opt.value = ws.id;
        opt.textContent = `${ws.name} (${ws.source})`;
        selectWs.appendChild(opt);
      });
    }

    // Verify activeWorkspaceId actually exists
    if (chatState.activeWorkspaceId) {
      const match = chatState.workspaces.find(w => w.id === chatState.activeWorkspaceId);
      if (match) {
        if (selectWs) selectWs.value = chatState.activeWorkspaceId;
        updateActiveWorkspaceCard();
        return;
      }
    }

    // No valid saved workspace
    chatState.activeWorkspaceId = null;
    if (selectWs) selectWs.value = '';
    updateActiveWorkspaceCard();
  }

  function updateActiveWorkspaceCard() {
    const ws = chatState.workspaces.find(w => w.id === chatState.activeWorkspaceId);
    if (!ws) {
      if (wsCard) wsCard.style.display = 'none';
      if (activeRepoBadge) activeRepoBadge.textContent = 'No project';
      appState.activeWorkspaceId = null;
      appState.activeWorkspace = null;
      appState.repoPath = null;
      const repoInput = document.getElementById('input-repo-path');
      if (repoInput) repoInput.value = '';
      setStatusCallback('no-repo', 'no project open', 'No project is open.');
      if (onWorkspaceChanged) onWorkspaceChanged(null);
      return;
    }

    if (wsCard) wsCard.style.display = 'block';
    if (activeRepoBadge) activeRepoBadge.textContent = ws.name;

    appState.activeWorkspaceId = ws.id;
    appState.activeWorkspace = ws;
    appState.repoPath = ws.working_path;
    const repoInput = document.getElementById('input-repo-path');
    if (repoInput) repoInput.value = ws.working_path;

    const counts = ws.file_counts || {};
    const py = counts.python || 0;
    const txt = counts.text_only || 0;
    const skip = counts.skipped || 0;
    if (wsCountsText) {
      wsCountsText.textContent = `${py} Python files · ${txt} text-only · ${skip} skipped`;
    }

    setStatusCallback('ready', 'ready', `Active workspace: ${ws.name}`);
    if (onWorkspaceChanged) onWorkspaceChanged(ws);
  }

  function switchWorkspace(id) {
    chatState.activeWorkspaceId = id || null;
    try {
      if (chatState.activeWorkspaceId) {
        localStorage.setItem('cce_active_workspace_id', chatState.activeWorkspaceId);
      } else {
        localStorage.removeItem('cce_active_workspace_id');
      }
    } catch (_) {}

    if (selectWs) selectWs.value = chatState.activeWorkspaceId || '';
    updateActiveWorkspaceCard();
    chatState.activeConversationId = null;
    loadConversations();
  }

  selectWs?.addEventListener('change', () => {
    switchWorkspace(selectWs.value || null);
  });

  // Workspace Actions
  btnWsZip?.addEventListener('click', () => {
    if (!chatState.activeWorkspaceId) return;
    window.open(`/api/workspaces/${encodeURIComponent(chatState.activeWorkspaceId)}/download`, '_blank');
  });

  btnWsPatch?.addEventListener('click', () => {
    if (!chatState.activeWorkspaceId) return;
    window.open(`/api/workspaces/${encodeURIComponent(chatState.activeWorkspaceId)}/patch`, '_blank');
  });

  btnWsReindex?.addEventListener('click', async () => {
    if (!chatState.activeWorkspaceId) return;
    btnWsReindex.disabled = true;
    btnWsReindex.textContent = 'Reindexing...';
    try {
      await api.reindexWorkspace(chatState.activeWorkspaceId);
      await loadWorkspaces();
      notify('Workspace reindexed successfully.');
    } catch (e) {
      notify(`Reindexing failed: ${friendlyError(e)}`);
    } finally {
      btnWsReindex.disabled = false;
      btnWsReindex.textContent = '🔄 Reindex';
    }
  });

  btnWsDelete?.addEventListener('click', async () => {
    if (!chatState.activeWorkspaceId) return;
    const ws = chatState.workspaces.find(w => w.id === chatState.activeWorkspaceId);
    const confirmName = prompt(`Are you sure? Type '${ws?.name}' to delete workspace and all data:`);
    if (confirmName === ws?.name) {
      await api.deleteWorkspace(chatState.activeWorkspaceId);
      switchWorkspace(null);
      await loadWorkspaces();
      await loadConversations();
    }
  });

  // Modal Workspace Openers
  const openModalWs = () => { if (modalWs) modalWs.style.display = 'flex'; };
  const closeModalWs = () => { if (modalWs) modalWs.style.display = 'none'; };
  btnOpenWsModal?.addEventListener('click', openModalWs);
  btnSidebarNewWs?.addEventListener('click', openModalWs);
  btnCloseWsModal?.addEventListener('click', closeModalWs);

  // Modal Tabs
  const tabLocal = document.getElementById('tab-ws-local');
  const tabUpload = document.getElementById('tab-ws-upload');
  const tabEmpty = document.getElementById('tab-ws-empty');
  const secLocal = document.getElementById('section-ws-local');
  const secUpload = document.getElementById('section-ws-upload');
  const secEmpty = document.getElementById('section-ws-empty');

  function setModalTab(activeTab, activeSec) {
    [tabLocal, tabUpload, tabEmpty].forEach(t => t?.classList.remove('active'));
    [secLocal, secUpload, secEmpty].forEach(s => { if (s) s.style.display = 'none'; });
    activeTab?.classList.add('active');
    if (activeSec) activeSec.style.display = 'block';
  }

  tabLocal?.addEventListener('click', () => setModalTab(tabLocal, secLocal));
  tabUpload?.addEventListener('click', () => setModalTab(tabUpload, secUpload));
  tabEmpty?.addEventListener('click', () => setModalTab(tabEmpty, secEmpty));

  // Submit Open Local Path
  document.getElementById('btn-submit-open-local')?.addEventListener('click', async () => {
    const pathInput = document.getElementById('modal-input-local-path');
    const inPlaceCheck = document.getElementById('modal-check-edit-in-place');
    const path = pathInput ? pathInput.value.trim() : '';
    if (!path) {
      notify('Please enter a repository path.');
      return;
    }
    const editInPlace = inPlaceCheck ? inPlaceCheck.checked : false;

    try {
      const res = await api.createWorkspace({ local_path: path, edit_in_place: editInPlace });
      if (res && res.workspace) {
        closeModalWs();
        await loadWorkspaces();
        switchWorkspace(res.workspace.id);
        if (res.secret_warnings && res.secret_warnings.length > 0) {
          notify(`Warning: Secrets detected and excluded:\n${res.secret_warnings.join('\n')}`);
        }
      }
    } catch (err) {
      notify(`Failed to open workspace: ${friendlyError(err)}`);
    }
  });

  // Submit Upload Zip from modal
  const dropZone = document.getElementById('drop-zone-upload');
  const uploadFileInput = document.getElementById('modal-input-upload-file');
  const uploadStatus = document.getElementById('upload-status-msg');

  dropZone?.addEventListener('click', () => uploadFileInput?.click());
  dropZone?.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--accent-color)'; });
  dropZone?.addEventListener('dragleave', () => { dropZone.style.borderColor = 'var(--border-color)'; });
  dropZone?.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--border-color)';
    if (e.dataTransfer.files.length > 0) {
      handleModalZipUpload(e.dataTransfer.files[0]);
    }
  });

  uploadFileInput?.addEventListener('change', () => {
    if (uploadFileInput.files.length > 0) {
      handleModalZipUpload(uploadFileInput.files[0]);
    }
  });

  async function handleModalZipUpload(file) {
    if (!file.name.toLowerCase().endsWith('.zip')) {
      notify('Only .zip archives are supported for workspace upload.');
      return;
    }
    if (uploadStatus) { uploadStatus.classList.remove('is-error'); uploadStatus.innerHTML = '<span class="spinner"></span> Uploading, scanning and indexing your code. This can take a few seconds...'; }
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await api.createWorkspace(fd);
      if (res && res.workspace) {
        closeModalWs();
        if (uploadStatus) uploadStatus.textContent = '';
        await loadWorkspaces();
        switchWorkspace(res.workspace.id);
        if (res.secret_warnings && res.secret_warnings.length > 0) {
          notify(`Uploaded with secret warnings:\n${res.secret_warnings.join('\n')}`);
        }
      }
    } catch (err) {
      if (uploadStatus) { uploadStatus.textContent = `Upload failed: ${friendlyError(err)}`; uploadStatus.classList.add('is-error'); }
    }
  }

  // Submit Chat Only
  document.getElementById('btn-submit-chat-only')?.addEventListener('click', async () => {
    try {
      const res = await api.createWorkspace({});
      if (res && res.workspace) {
        closeModalWs();
        await loadWorkspaces();
        switchWorkspace(res.workspace.id);
      }
    } catch (err) {
      notify(`Failed to create chat workspace: ${friendlyError(err)}`);
    }
  });

  // =========================================================================
  // 2. Chat Box Upload & Drag-and-Drop Pipeline
  // =========================================================================

  btnAttach?.addEventListener('click', () => fileInput?.click());
  btnAttachFolder?.addEventListener('click', () => folderInput?.click());

  fileInput?.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleIncomingFiles(Array.from(fileInput.files));
      fileInput.value = '';
    }
  });

  folderInput?.addEventListener('change', () => {
    if (folderInput.files.length > 0) {
      handleIncomingFiles(Array.from(folderInput.files));
      folderInput.value = '';
    }
  });

  // Drag and drop onto chat main area
  [chatMainEl, messagesScrollEl, chatInputAreaEl].forEach(el => {
    if (!el) return;
    el.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (chatInputAreaEl) chatInputAreaEl.style.borderColor = 'var(--accent-color)';
    });
    el.addEventListener('dragleave', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (chatInputAreaEl) chatInputAreaEl.style.borderColor = '';
    });
    el.addEventListener('drop', (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (chatInputAreaEl) chatInputAreaEl.style.borderColor = '';
      if (e.dataTransfer && e.dataTransfer.files.length > 0) {
        handleIncomingFiles(Array.from(e.dataTransfer.files));
      }
    });
  });

  // Paste support on chat textarea
  chatInput?.addEventListener('paste', (e) => {
    if (e.clipboardData && e.clipboardData.files.length > 0) {
      e.preventDefault();
      handleIncomingFiles(Array.from(e.clipboardData.files));
    }
  });

  /**
   * Main router for files incoming through chat box:
   * a. If NO workspace is active -> create new workspace automatically and index it
   * b. If a workspace IS active -> prompt with inline 3-way choice
   */
  async function handleIncomingFiles(files) {
    if (!files || files.length === 0) return;

    if (!chatState.activeWorkspaceId) {
      // Flow (a): Auto-create new workspace from upload
      await executeCreateWorkspaceFromUpload(files);
    } else {
      // Flow (b): Show inline choice modal
      promptUploadActionChoice(files);
    }
  }

  function promptUploadActionChoice(files) {
    chatState.pendingUploadFiles = files;
    const currentWs = chatState.workspaces.find(w => w.id === chatState.activeWorkspaceId);

    if (choiceCurrentWsName) {
      choiceCurrentWsName.textContent = currentWs ? currentWs.name : 'Active';
    }

    if (uploadChoiceDesc) {
      const names = files.slice(0, 3).map(f => f.name).join(', ');
      const more = files.length > 3 ? ` +${files.length - 3} more` : '';
      uploadChoiceDesc.textContent = `Attached ${files.length} item(s): ${names}${more}. Choose how to process this upload:`;
    }

    // Default pre-selection rule:
    // "default selection is 'Start a new workspace' when the upload contains more than 1 file or a zip,
    // and 'Add to current workspace' for a single file only if the user picks it."
    if (choiceStartNew) choiceStartNew.checked = true;
    if (choiceAddCurrent) choiceAddCurrent.checked = false;
    if (choiceAttachChat) choiceAttachChat.checked = false;

    if (modalUploadChoice) modalUploadChoice.style.display = 'flex';
  }

  const closeUploadChoiceModal = () => {
    if (modalUploadChoice) modalUploadChoice.style.display = 'none';
    chatState.pendingUploadFiles = null;
  };

  btnCloseUploadChoice?.addEventListener('click', closeUploadChoiceModal);
  btnCancelUploadChoice?.addEventListener('click', closeUploadChoiceModal);

  btnConfirmUploadChoice?.addEventListener('click', async () => {
    const files = chatState.pendingUploadFiles;
    if (!files || files.length === 0) {
      closeUploadChoiceModal();
      return;
    }

    const isNew = choiceStartNew && choiceStartNew.checked;
    const isAdd = choiceAddCurrent && choiceAddCurrent.checked;
    const isAttach = choiceAttachChat && choiceAttachChat.checked;

    closeUploadChoiceModal();

    if (isNew) {
      await executeCreateWorkspaceFromUpload(files);
    } else if (isAdd) {
      await executeAddToActiveWorkspace(files);
    } else if (isAttach) {
      await executeAttachToCurrentChat(files);
    }
  });

  async function executeCreateWorkspaceFromUpload(files) {
    const isSingleZip = files.length === 1 && files[0].name.toLowerCase().endsWith('.zip');
    let wsName = 'Uploaded Code';

    if (isSingleZip) {
      wsName = files[0].name.replace(/\.zip$/i, '');
    } else if (files.length === 1) {
      wsName = files[0].name;
    } else {
      const firstRel = files[0].webkitRelativePath || files[0].name;
      const parts = firstRel.split('/');
      wsName = parts.length > 1 ? parts[0] : (files[0].name.split('.')[0] + ' project');
    }

    // Render inline progress indicator
    const progressBubble = appendSystemNotice(`🔄 Scanning, Parsing, Building graph, Embedding: <strong>${escapeHtml(wsName)}</strong>...`);

    const fd = new FormData();
    if (isSingleZip) {
      fd.append('file', files[0]);
      fd.append('name', wsName);
    } else {
      files.forEach(f => {
        const rel = f.webkitRelativePath || f.name;
        fd.append('files', f);
        fd.append('relative_paths', rel);
      });
      fd.append('name', wsName);
    }

    try {
      const res = await api.createWorkspace(fd);
      if (res && res.workspace) {
        await loadWorkspaces();
        switchWorkspace(res.workspace.id);

        const py = res.workspace.file_counts?.python || 0;
        const txt = res.workspace.file_counts?.text_only || 0;
        let note = `✓ Created workspace <strong>${escapeHtml(res.workspace.name)}</strong> (${py} Python, ${txt} text-only). Code graph & stats ready!`;

        if (res.secret_warnings && res.secret_warnings.length > 0) {
          note += `<br>⚠️ Secret warnings: ${escapeHtml(res.secret_warnings.join(', '))}`;
        }
        if (res.skip_report) {
          const skippedList = [];
          for (const [k, v] of Object.entries(res.skip_report)) {
            if (Array.isArray(v) && v.length > 0) skippedList.push(`${k}: ${v.length}`);
          }
          if (skippedList.length > 0) {
            note += `<br><span style="font-size: 11px; color: var(--text-secondary);">Skipped: ${escapeHtml(skippedList.join(', '))}</span>`;
          }
        }

        updateSystemNotice(progressBubble, note, 'success');
      }
    } catch (err) {
      updateSystemNotice(progressBubble, `❌ Failed to create workspace: ${escapeHtml(err.message)}`, 'error');
    }
  }

  async function executeAddToActiveWorkspace(files) {
    const currentWs = chatState.workspaces.find(w => w.id === chatState.activeWorkspaceId);
    const wsName = currentWs ? currentWs.name : 'workspace';

    const progressBubble = appendSystemNotice(`🔄 Incrementally updating <strong>${escapeHtml(wsName)}</strong>: hashing, parsing, updating graph & embeddings...`);

    const fd = new FormData();
    files.forEach(f => {
      const rel = f.webkitRelativePath || f.name;
      fd.append('files', f);
      fd.append('relative_paths', rel);
    });

    try {
      const res = await api.addWorkspaceFiles(chatState.activeWorkspaceId, fd);
      if (res) {
        await loadWorkspaces();
        updateActiveWorkspaceCard();

        const added = res.report?.added?.length || 0;
        const mod = res.report?.modified?.length || 0;
        let note = `✓ Incremental update finished for <strong>${escapeHtml(wsName)}</strong> (+${added} added, ~${mod} modified). Graph refreshed!`;

        if (res.secret_warnings && res.secret_warnings.length > 0) {
          note += `<br>⚠️ Secret warnings: ${escapeHtml(res.secret_warnings.join(', '))}`;
        }
        updateSystemNotice(progressBubble, note, 'success');
      }
    } catch (err) {
      updateSystemNotice(progressBubble, `❌ Incremental update failed: ${escapeHtml(err.message)}`, 'error');
    }
  }

  async function executeAttachToCurrentChat(files) {
    for (const f of files) {
      await uploadChatAttachment(f);
    }
    appendSystemNotice(`📎 Attached ${files.length} item(s) to this chat only (not indexed into the code graph).`);
  }

  async function uploadChatAttachment(file) {
    if (!chatState.activeConversationId) {
      const newConv = await api.createConversation({
        workspace_id: chatState.activeWorkspaceId,
        title: `Chat with ${file.name}`,
        mode: chatState.mode,
      });
      chatState.activeConversationId = newConv.id;
      await loadConversations();
    }

    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await api.uploadAttachment(
        chatState.activeConversationId,
        fd,
        appState.selectedProvider,
        appState.selectedModel,
      );
      if (res && res.attachment) {
        if (res.secret_warnings && res.secret_warnings.length > 0) {
          notify(`Attachment warning:\n${res.secret_warnings.join('\n')}`);
        }
        await refreshAttachments();
      }
    } catch (err) {
      notify(`Attachment failed: ${friendlyError(err)}`);
    }
  }

  function appendSystemNotice(html) {
    if (!messagesContainer) return null;
    const row = document.createElement('div');
    row.className = 'msg-row msg-system';
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble system-notice';
    bubble.innerHTML = html;
    row.appendChild(bubble);
    messagesContainer.appendChild(row);
    scrollToBottom();
    return bubble;
  }

  // Animated "thinking" bubble; the next renderMessages() call clears it.
  function showTypingIndicator() {
    if (!messagesContainer) return;
    const row = document.createElement('div');
    row.className = 'msg-row msg-assistant msg-typing';
    row.innerHTML = '<div class="msg-bubble typing-bubble" role="status"><span class="typing-dots" aria-hidden="true"><span></span><span></span><span></span></span>Reading your code and writing an answer...</div>';
    messagesContainer.appendChild(row);
    scrollToBottom();
  }

  function updateSystemNotice(bubble, html, type = 'info') {
    if (!bubble) return;
    bubble.innerHTML = html;
    bubble.className = `msg-bubble system-notice notice-${type}`;
    scrollToBottom();
  }

  async function refreshAttachments() {
    if (!chatState.activeConversationId) return;
    try {
      const atts = await api.listAttachments(chatState.activeConversationId);
      renderAttachments(atts);
    } catch (_) {}
  }

  function renderAttachments(atts) {
    if (!attachmentTray) return;
    if (!Array.isArray(atts) || atts.length === 0) {
      attachmentTray.style.display = 'none';
      attachmentTray.innerHTML = '';
      return;
    }
    attachmentTray.style.display = 'flex';
    attachmentTray.innerHTML = '';

    atts.forEach(a => {
      const chip = document.createElement('div');
      chip.className = 'att-chip';
      chip.innerHTML = `
        <span>📎 ${escapeHtml(a.filename)}</span>
        <button class="att-chip-remove" title="Remove attachment">&times;</button>
      `;
      chip.querySelector('.att-chip-remove').addEventListener('click', async () => {
        await api.deleteAttachment(chatState.activeConversationId, a.id);
        await refreshAttachments();
      });
      attachmentTray.appendChild(chip);
    });
  }

  // =========================================================================
  // 3. Conversations Management
  // =========================================================================

  async function loadConversations() {
    try {
      const list = await api.listConversations(chatState.activeWorkspaceId);
      chatState.conversations = Array.isArray(list) ? list : [];
      renderConversationList();
    } catch (err) {
      console.warn('Failed to load conversations:', err);
    }
  }

  function renderConversationList() {
    if (!convListEl) return;
    if (chatState.conversations.length === 0) {
      convListEl.innerHTML = '<div class="empty-list-msg">No chats yet. Type a question to start one.</div>';
      renderMessages([]);
      if (threadTitleEl) threadTitleEl.textContent = 'New chat';
      return;
    }

    convListEl.innerHTML = '';
    chatState.conversations.forEach(c => {
      const item = document.createElement('div');
      item.className = `conv-item ${c.id === chatState.activeConversationId ? 'active' : ''}`;
      item.innerHTML = `
        <span class="conv-title" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</span>
        <button class="btn-del-conv" title="Delete conversation">&times;</button>
      `;

      item.querySelector('.conv-title').addEventListener('click', () => {
        selectConversation(c.id);
      });

      item.querySelector('.btn-del-conv').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (confirm(`Delete conversation "${c.title}"?`)) {
          await api.deleteConversation(c.id);
          if (chatState.activeConversationId === c.id) {
            chatState.activeConversationId = null;
          }
          await loadConversations();
        }
      });

      convListEl.appendChild(item);
    });

    if (!chatState.activeConversationId && chatState.conversations.length > 0) {
      selectConversation(chatState.conversations[0].id);
    }
  }

  async function selectConversation(id) {
    chatState.activeConversationId = id;
    renderConversationList();

    try {
      const data = await api.getConversation(id);
      if (data && data.conversation) {
        if (threadTitleEl) threadTitleEl.textContent = data.conversation.title;
        setModePill(data.conversation.mode || 'balanced');
        renderMessages(data.messages || []);
        renderAttachments(data.attachments || []);
      }
    } catch (err) {
      console.warn('Failed to load conversation details:', err);
    }
  }

  btnNewChat?.addEventListener('click', async () => {
    try {
      const newConv = await api.createConversation({
        workspace_id: chatState.activeWorkspaceId,
        title: 'New chat',
        mode: chatState.mode,
      });
      chatState.activeConversationId = newConv.id;
      await loadConversations();
      selectConversation(newConv.id);
    } catch (err) {
      notify(`Failed to start new chat: ${friendlyError(err)}`);
    }
  });

  // Mode Pill Controls
  document.querySelectorAll('.mode-pill').forEach(btn => {
    btn.addEventListener('click', () => {
      setModePill(btn.dataset.mode);
    });
  });

  function setModePill(mode) {
    chatState.mode = mode;
    document.querySelectorAll('.mode-pill').forEach(b => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    const budgetSlider = document.getElementById('input-budget');
    const budgetVal = document.getElementById('budget-val');
    const budgetMap = { economy: 2000, balanced: 4000, quality: 8000 };
    if (budgetSlider && budgetMap[mode]) {
      budgetSlider.value = budgetMap[mode];
      if (budgetVal) budgetVal.textContent = budgetMap[mode].toLocaleString();
      appState.budget = budgetMap[mode];
    }
  }

  // =========================================================================
  // 4. Model & Key Settings Modal
  // =========================================================================

  btnOpenModelModal?.addEventListener('click', () => {
    if (modalModel) modalModel.style.display = 'flex';
  });
  btnCloseModelModal?.addEventListener('click', () => {
    if (modalModel) modalModel.style.display = 'none';
  });

  function updateActiveModelBadge() {
    if (!badgeActiveModel) return;
    const prov = appState.selectedProvider || 'None';
    const model = appState.selectedModel || 'Default';
    badgeActiveModel.textContent = `${prov} / ${model}`;
  }

  // =========================================================================
  // 5. Message Thread & Citations Rendering
  // =========================================================================

  function renderMessages(messages) {
    chatState.messages = Array.isArray(messages) ? messages.slice() : [];
    if (!messagesContainer) return;
    messagesContainer.innerHTML = '';

    if (!messages || messages.length === 0) {
      messagesContainer.innerHTML = welcomeHtml();
      return;
    }

    messages.forEach(m => {
      const row = document.createElement('div');
      row.className = `msg-row msg-${m.role}`;

      const bubble = document.createElement('div');
      bubble.className = 'msg-bubble';

      if (m.role === 'user') {
        bubble.textContent = m.content;
      } else {
        bubble.innerHTML = renderAssistantContent(m);
        attachAssistantInteractivity(bubble, m);
      }

      const meta = document.createElement('div');
      meta.className = 'msg-meta';
      const timeStr = new Date(m.created_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      meta.textContent = `${m.role === 'user' ? 'You' : (m.provider || 'AI')} · ${timeStr}`;

      row.appendChild(bubble);

      const trace = m.role === 'assistant' ? normalizeTrace(m.context_manifest && m.context_manifest.trace) : null;
      if (trace && trace.savings) {
        const holder = document.createElement('div');
        holder.className = 'trace-summary-holder';
        holder.innerHTML = renderTraceSummaryLine(trace);
        holder.querySelector('.trace-summary-line')?.addEventListener('click', () => {
          renderSavingsPanel(m.raw_tokens || (m.context_manifest && m.context_manifest.tokens), trace);
        });
        row.appendChild(holder);
      }

      row.appendChild(meta);
      messagesContainer.appendChild(row);
    });

    scrollToBottom();
  }

  function attachAssistantInteractivity(container, msg) {
    container.querySelectorAll('.btn-copy-code').forEach(btn => {
      btn.addEventListener('click', () => {
        navigator.clipboard.writeText(btn.dataset.code);
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
      });
    });

    container.querySelectorAll('.file-ref-link').forEach(link => {
      link.addEventListener('click', () => {
        openFileViewer(link.dataset.path, parseInt(link.dataset.start, 10), parseInt(link.dataset.end, 10));
      });
    });

    container.querySelectorAll('.needs-context-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        if (chatInput) {
          chatInput.value = `Explain and show context for ${chip.dataset.target}`;
          handleSendMessage();
        }
      });
    });

    container.querySelectorAll('.btn-feedback').forEach(fbBtn => {
      fbBtn.addEventListener('click', async () => {
        await api.submitFeedback(chatState.activeConversationId, fbBtn.dataset.msg, fbBtn.dataset.rating);
        fbBtn.style.color = fbBtn.dataset.rating === 'up' ? 'var(--success-text)' : 'var(--danger-text)';
      });
    });

    container.querySelectorAll('.btn-apply-diff').forEach(applyBtn => {
      applyBtn.addEventListener('click', async () => {
        if (!chatState.activeWorkspaceId) {
          notify('Cannot apply changes without an open workspace.');
          return;
        }
        applyBtn.disabled = true;
        applyBtn.textContent = 'Applying...';
        try {
          const res = await api.applyEdit(chatState.activeWorkspaceId, {
            change_type: applyBtn.dataset.type,
            target_symbol: applyBtn.dataset.symbol,
            file_path: applyBtn.dataset.path,
            new_content: applyBtn.dataset.code,
            message_id: applyBtn.dataset.msg,
          });
          if (res && res.status === 'applied') {
            applyBtn.textContent = '✓ Applied';
            applyBtn.classList.remove('btn-secondary');
            applyBtn.classList.add('btn-ghost');

            const undoBtn = applyBtn.parentElement.querySelector('.btn-undo-diff');
            if (undoBtn) {
              undoBtn.style.display = 'inline-block';
              undoBtn.onclick = async () => {
                await api.undoEdit(chatState.activeWorkspaceId);
                undoBtn.style.display = 'none';
                applyBtn.disabled = false;
                applyBtn.textContent = 'Apply to Workspace Copy';
              };
            }
          }
        } catch (err) {
          notify(`Failed to apply edit: ${friendlyError(err)}`);
          applyBtn.disabled = false;
          applyBtn.textContent = 'Apply to Workspace Copy';
        }
      });
    });
  }

  async function openFileViewer(path, startLine, endLine) {
    if (!chatState.activeWorkspaceId) {
      notify('Upload code or open a workspace to view repository files.');
      return;
    }
    try {
      const data = await api.getWorkspaceFile(chatState.activeWorkspaceId, path, startLine, endLine);
      if (viewerTitle) viewerTitle.textContent = `${path}${startLine ? ` (lines ${startLine}-${endLine})` : ''}`;
      if (viewerCode) viewerCode.textContent = data.content;
      if (modalViewer) modalViewer.style.display = 'flex';
    } catch (err) {
      notify(`Could not open file: ${friendlyError(err)}`);
    }
  }

  btnCloseViewer?.addEventListener('click', () => {
    if (modalViewer) modalViewer.style.display = 'none';
  });

  function scrollToBottom() {
    if (messagesScrollEl) {
      messagesScrollEl.scrollTop = messagesScrollEl.scrollHeight;
    }
  }

  // =========================================================================
  // 6. Send Message Handler
  // =========================================================================

  btnSend?.addEventListener('click', handleSendMessage);
  chatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  async function handleSendMessage() {
    const text = chatInput ? chatInput.value.trim() : '';
    if (!text) return;

    // Check if code exists in workspace or attachments exist
    if (!chatState.activeWorkspaceId && (!chatState.pendingAttachments || chatState.pendingAttachments.length === 0)) {
      // Append user message
      const userMsg = { role: 'user', content: text, created_at: Date.now() / 1000 };
      const assistantNotice = {
        role: 'assistant',
        content: 'Upload code first to ask questions about a repository or inspect files.',
        created_at: Date.now() / 1000 + 0.1,
      };
      renderMessages([...chatState.messages, userMsg, assistantNotice]);
      chatInput.value = '';
      return;
    }

    if (!chatState.activeConversationId) {
      const newConv = await api.createConversation({
        workspace_id: chatState.activeWorkspaceId,
        title: text.slice(0, 30),
        mode: chatState.mode,
      });
      chatState.activeConversationId = newConv.id;
      if (threadTitleEl) threadTitleEl.textContent = newConv.title;
      await loadConversations();
    }

    const userMsgObj = { role: 'user', content: text, created_at: Date.now() / 1000 };
    renderMessages([...chatState.messages, userMsgObj]);
    chatInput.value = '';

    btnSend.disabled = true;
    btnSend.textContent = 'Thinking...';
    showTypingIndicator();

    const prov = appState.selectedProvider;
    const model = appState.selectedModel;
    const rawKey = prov ? keyManager.getKey(prov) : null;
    const isDryRun = Boolean(document.getElementById('toggle-dry-run')?.checked);

    try {
      const res = await api.sendMessage(
        chatState.activeConversationId,
        {
          content: text,
          provider: prov,
          model: model,
          mode: chatState.mode,
          budget: appState.budget,
          dry_run: isDryRun,
        },
        rawKey,
      );

      if (res && res.assistant_message) {
        renderMessages([...chatState.messages, res.assistant_message]);
        if (res.turn_tokens) {
          chatState.lastTurnTokens = res.turn_tokens;
          renderSavingsPanel(res.turn_tokens, normalizeTrace(res.trace));
        }
      }
    } catch (err) {
      const errMsg = {
        role: 'assistant',
        content: `Could not complete that request. ${friendlyError(err)}${err.requestId ? ` (Request ID: ${err.requestId})` : ''}`,
        created_at: Date.now() / 1000,
      };
      renderMessages([...chatState.messages, errMsg]);
    } finally {
      btnSend.disabled = false;
      btnSend.textContent = 'Send';
      scrollToBottom();
    }
  }

  // =========================================================================
  // 7. Token Savings Panel
  // =========================================================================

  btnToggleSavings?.addEventListener('click', () => {
    if (!savingsPanel) return;
    const isHidden = savingsPanel.style.display === 'none' || getComputedStyle(savingsPanel).display === 'none';
    savingsPanel.style.display = isHidden ? 'flex' : 'none';
  });

  btnCloseSavings?.addEventListener('click', () => {
    if (savingsPanel) savingsPanel.style.display = 'none';
  });

  function renderSavingsPanel(tokens, trace = null) {
    if (!savingsContent) return;
    // With a trace, its hero card already shows the savings; keep only the retry action from the basic card.
    savingsContent.innerHTML = trace
      ? renderTraceHtml(trace) + '<div class="savings-actions"><button id="btn-retry-more-context" class="btn btn-secondary btn-xs" title="Retry this turn with a larger context mode">Retry with more context</button></div>'
      : renderSavingsHtml(tokens);
    if (savingsPanel) savingsPanel.style.display = 'flex';

    savingsContent.querySelectorAll('.trace-focus-graph').forEach(btn => {
      btn.addEventListener('click', () => {
        window.dispatchEvent(new CustomEvent('cce:focus-graph', { detail: { ref: btn.dataset.ref } }));
      });
    });

    document.getElementById('btn-retry-more-context')?.addEventListener('click', () => {
      const nextMode = chatState.mode === 'economy' ? 'balanced' : 'quality';
      setModePill(nextMode);
      const lastUser = [...chatState.messages].reverse().find(m => m.role === 'user');
      if (lastUser && chatInput) {
        chatInput.value = lastUser.content;
        handleSendMessage();
      }
    });
  }

  btnToggleSettings?.addEventListener('click', () => {
    if (settingsPopover) {
      const isHidden = settingsPopover.style.display === 'none';
      settingsPopover.style.display = isHidden ? 'block' : 'none';
    }
  });

  // Initial Load
  loadWorkspaces();
  loadConversations();
  updateActiveModelBadge();

  return {
    updateActiveModelBadge,
    loadWorkspaces,
    switchWorkspace,
    getWorkspaces: () => chatState.workspaces,
    getActiveWorkspaceId: () => chatState.activeWorkspaceId,
  };
}

// =========================================================================
// Pure Component Renderers (Exported for Unit Tests)
// =========================================================================

export function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

export function renderDiffCard(type, symbol, fpath, code, idx = 0, msgId = '0') {
  return `
    <div class="diff-card" id="diff-card-${msgId}-${idx}">
      <div class="diff-header">
        <span class="diff-title">${type === 'new_file' ? 'New File' : 'Change'}: ${escapeHtml(symbol)} (${escapeHtml(fpath)})</span>
        <div class="diff-status">
          <span class="status-valid">✓ Proposed replacement</span>
        </div>
      </div>
      <div class="diff-viewer">
        <pre><code>${renderDiffLines(code)}</code></pre>
      </div>
      <div class="diff-actions">
        <button class="btn btn-secondary btn-sm btn-apply-diff"
          data-type="${escapeHtml(type)}"
          data-symbol="${escapeHtml(symbol)}"
          data-path="${escapeHtml(fpath)}"
          data-code="${escapeHtml(code)}"
          data-msg="${escapeHtml(msgId)}">
          Apply to Workspace Copy
        </button>
        <button class="btn btn-ghost btn-sm btn-undo-diff" style="display: none;">
          ↩ Undo Edit
        </button>
      </div>
    </div>
  `;
}

export function renderDiffLines(code) {
  if (!code) return '';
  const lines = code.split('\n');
  return lines.map(l => {
    let cls = 'diff-ctx';
    if (l.startsWith('+')) cls = 'diff-add';
    else if (l.startsWith('-')) cls = 'diff-del';
    return `<div class="${cls}">${escapeHtml(l)}</div>`;
  }).join('');
}

export function renderSavingsHtml(tokens) {
  if (!tokens) return '<div class="empty-state">No savings data recorded yet.</div>';

  const actualSent = tokens.actual_total_sent !== undefined
    ? tokens.actual_total_sent
    : (tokens.total_tokens || tokens.prompt_tokens || 0);

  const ctxTokens = tokens.actual_context_tokens !== undefined
    ? tokens.actual_context_tokens
    : (tokens.prompt_tokens || 0);

  const budget = tokens.budget || 4000;
  const wholeFilesBase = tokens.whole_files_baseline || 0;
  const fullRepoBase = tokens.full_repo_baseline || 0;
  const wholeFilesSavedPct = tokens.whole_files_saved_pct || 0;
  const fullRepoSavedPct = tokens.full_repo_saved_pct || 0;
  const fmt = n => Math.round(n).toLocaleString();
  const baselineRow = (label, base, pct) => base > 0
    ? `<div class="metric-row"><span>${label}</span><strong>${fmt(base)} tokens${pct > 0 ? ` <span class="trace-saved">(${pct}% saved)</span>` : ''}</strong></div>`
    : '';

  return `
    <div class="savings-card">
      <div class="savings-metric-big">
        <span class="savings-pct">${fullRepoBase > 0 ? `${fullRepoSavedPct}%` : '&ndash;'}</span>
        <span class="savings-label">fewer tokens than sending the full repo</span>
      </div>
      <div class="savings-section">
        <div class="metric-row">
          <span>Tokens sent this turn</span>
          <strong>${fmt(actualSent)}</strong>
        </div>
        <div class="metric-sub">${fmt(ctxTokens)} repo context &middot; ${fmt(budget)} budget</div>
      </div>
      <div class="savings-details">
        ${baselineRow('Whole files used', wholeFilesBase, wholeFilesSavedPct)}
        ${baselineRow('Full repo pasted', fullRepoBase, fullRepoSavedPct)}
      </div>
      <div class="savings-actions">
        <button id="btn-retry-more-context" class="btn btn-secondary btn-xs" title="Retry this turn with a larger context mode">
          Retry with more context
        </button>
      </div>
    </div>
  `;
}

export function renderAssistantContent(msg) {
  let body = msg.content || '';
  const parts = [];

  // 1. Detect and parse CHANGE proposals: ### CHANGE: symbol (path)\n```lang\ncode\n```
  const changeRegex = /### CHANGE:\s*([^\s(]+)\s*\(([^)]+)\)\s*```[a-zA-Z0-9_-]*\n([\s\S]*?)```/g;
  let match;
  let lastIdx = 0;
  let diffCount = 0;

  while ((match = changeRegex.exec(body)) !== null) {
    const textBefore = body.slice(lastIdx, match.index);
    if (textBefore.trim()) {
      parts.push(`<div class="msg-text">${renderMarkdownFormatting(textBefore)}</div>`);
    }
    const symbol = match[1];
    const fpath = match[2];
    const code = match[3].trim();
    parts.push(renderDiffCard('modify', symbol, fpath, code, diffCount++, String(msg.id || msg.created_at || '0')));
    lastIdx = changeRegex.lastIndex;
  }

  // 2. Detect and parse diff blocks: ```diff ... ```
  let remaining = body.slice(lastIdx);
  lastIdx = 0;
  const diffRegex = /```diff([\s\S]*?)```/g;
  while ((match = diffRegex.exec(remaining)) !== null) {
    const textBefore = remaining.slice(lastIdx, match.index);
    if (textBefore.trim()) {
      parts.push(`<div class="msg-text">${renderMarkdownFormatting(textBefore)}</div>`);
    }
    const diffContent = match[1].trim();
    parts.push(renderDiffCard('change', 'Assistant Edit', 'workspace', diffContent, diffCount++, String(msg.id || msg.created_at || '0')));
    lastIdx = diffRegex.lastIndex;
  }

  const leftover = remaining.slice(lastIdx);
  if (leftover.trim()) {
    parts.push(`<div class="msg-text">${renderMarkdownFormatting(leftover)}</div>`);
  }

  // 3. Parse NEEDS_CONTEXT chips
  const needsContextMatches = [...body.matchAll(/NEEDS_CONTEXT:\s*([a-zA-Z0-9_.-]+)/g)];
  if (needsContextMatches.length > 0) {
    const chipsHtml = needsContextMatches.map(m => `
      <button class="needs-context-chip" data-target="${escapeHtml(m[1])}">
        Pin ${escapeHtml(m[1])} to context
      </button>
    `).join('');
    parts.push(`<div class="needs-context-chip-row">${chipsHtml}</div>`);
  }

  // 4. Feedback actions bar
  const feedbackBar = `
    <div class="msg-actions-bar">
      <button class="btn-icon-xs btn-feedback" data-msg="${escapeHtml(String(msg.id || ''))}" data-rating="up" title="Helpful answer">👍</button>
      <button class="btn-icon-xs btn-feedback" data-msg="${escapeHtml(String(msg.id || ''))}" data-rating="down" title="Unhelpful answer">👎</button>
    </div>
  `;
  parts.push(feedbackBar);

  return parts.join('');
}

export function renderMarkdownFormatting(text) {
  if (!text) return '';

  // Match file citations: both [path:lines] and path:lines
  let formatted = text.replace(/(?:\[)?([a-zA-Z0-9_\-./\\]+\.py):(?:L)?(\d+)(?:-(?:L)?(\d+))?(?:\])?/g, (match, path, start, end) => {
    const endLine = end || start;
    return `<a href="javascript:void(0)" class="file-ref-link" data-path="${escapeHtml(path)}" data-start="${start}" data-end="${endLine}">${escapeHtml(match)}</a>`;
  });

  // Code blocks: msg-code-block wrapper, code-lang badge, copy button
  formatted = formatted.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (match, lang, code) => {
    return `
      <div class="msg-code-block">
        <div class="code-block-header">
          <span class="code-lang">${escapeHtml(lang || 'code')}</span>
          <button class="btn-copy-code" data-code="${escapeHtml(code.trim())}">Copy</button>
        </div>
        <pre><code>${escapeHtml(code.trim())}</code></pre>
      </div>
    `;
  });

  formatted = formatted.replace(/`([^`]+)`/g, '<code>$1</code>');
  return formatted;
}
