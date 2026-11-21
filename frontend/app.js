/**
 * TechText AI - Frontend Client Logic
 * Handles interactive text generation, parameter sliders, clipboard, and history management.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const promptInput = document.getElementById("promptInput");
  const charCount = document.getElementById("charCount");
  const btnClearPrompt = document.getElementById("btnClearPrompt");
  const btnGenerate = document.getElementById("btnGenerate");
  const genSpinner = document.getElementById("genSpinner");
  const modelSelect = document.getElementById("modelSelect");

  // Sliders & Value Displays
  const sliderTemp = document.getElementById("sliderTemperature");
  const valTemp = document.getElementById("valTemperature");
  const sliderTopK = document.getElementById("sliderTopK");
  const valTopK = document.getElementById("valTopK");
  const sliderMaxTokens = document.getElementById("sliderMaxTokens");
  const valMaxTokens = document.getElementById("valMaxTokens");

  // Output elements
  const outputContainer = document.getElementById("outputContainer");
  const emptyState = document.getElementById("emptyState");
  const generatedContent = document.getElementById("generatedContent");
  const displayPrompt = document.getElementById("displayPrompt");
  const displayContinuation = document.getElementById("displayContinuation");
  const btnCopy = document.getElementById("btnCopy");
  const copyText = document.getElementById("copyText");

  // Metrics elements
  const metricsBar = document.getElementById("metricsBar");
  const metricTokens = document.getElementById("metricTokens");
  const metricLatency = document.getElementById("metricLatency");
  const metricModel = document.getElementById("metricModel");
  const metricSampling = document.getElementById("metricSampling");

  // History elements
  const historyList = document.getElementById("historyList");
  const historyCount = document.getElementById("historyCount");
  const btnClearHistory = document.getElementById("btnClearHistory");

  // Modal elements
  const btnInspectModel = document.getElementById("btnInspectModel");
  const archModal = document.getElementById("archModal");
  const btnCloseModal = document.getElementById("btnCloseModal");

  // Status pill
  const statusPill = document.getElementById("statusPill");
  const statusText = document.getElementById("statusText");

  // Chip buttons
  const chips = document.querySelectorAll(".chip");

  // 1. Slider Events
  sliderTemp.addEventListener("input", (e) => {
    valTemp.textContent = parseFloat(e.target.value).toFixed(2);
  });
  sliderTopK.addEventListener("input", (e) => {
    valTopK.textContent = e.target.value;
  });
  sliderMaxTokens.addEventListener("input", (e) => {
    valMaxTokens.textContent = e.target.value;
  });

  // 2. Character counter
  promptInput.addEventListener("input", () => {
    const len = promptInput.value.length;
    charCount.textContent = `${len} character${len === 1 ? "" : "s"}`;
  });

  // 3. Clear prompt
  btnClearPrompt.addEventListener("click", () => {
    promptInput.value = "";
    promptInput.focus();
    charCount.textContent = "0 characters";
  });

  // 4. Quick prompt chips
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      promptInput.value = chip.getAttribute("data-prompt");
      promptInput.dispatchEvent(new Event("input"));
      promptInput.focus();
    });
  });

  // 5. Modal Toggle
  btnInspectModel.addEventListener("click", () => {
    archModal.style.display = "flex";
  });
  btnCloseModal.addEventListener("click", () => {
    archModal.style.display = "none";
  });
  archModal.addEventListener("click", (e) => {
    if (e.target === archModal) {
      archModal.style.display = "none";
    }
  });

  // 6. Copy to Clipboard
  btnCopy.addEventListener("click", async () => {
    const textToCopy = (displayPrompt.textContent + displayContinuation.textContent).trim();
    if (!textToCopy) return;

    try {
      await navigator.clipboard.writeText(textToCopy);
      copyText.textContent = "Copied!";
      btnCopy.style.borderColor = "var(--accent-emerald)";
      setTimeout(() => {
        copyText.textContent = "Copy";
        btnCopy.style.borderColor = "";
      }, 2000);
    } catch (err) {
      console.error("Clipboard copy failed:", err);
    }
  });

  // 7. Check System Health & Load Models
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        statusPill.classList.remove("offline");
        statusPill.classList.add("online");
        statusText.textContent = `Model Ready (${data.device.toUpperCase()})`;
      } else {
        throw new Error("Health check non-200");
      }
    } catch (err) {
      statusPill.classList.remove("online");
      statusPill.classList.add("offline");
      statusText.textContent = "API Offline";
    }
  }

  async function loadModels() {
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const models = await res.json();
      if (models.length > 0) {
        modelSelect.innerHTML = models
          .map((m) => {
            const paramStr = m.parameters >= 1e6 ? `${(m.parameters / 1e6).toFixed(1)}M params` : `${Math.round(m.parameters / 1000)}K params`;
            return `<option value="${m.id}">${m.name} (${paramStr})</option>`;
          })
          .join("");
      }
    } catch (e) {
      console.warn("Could not load dynamic models list:", e);
    }
  }


  // 8. Generate AI Text
  btnGenerate.addEventListener("click", async () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
      promptInput.focus();
      promptInput.placeholder = "Please enter a prompt first!";
      return;
    }

    // Set UI to loading state
    btnGenerate.disabled = true;
    genSpinner.style.display = "inline-block";
    btnGenerate.querySelector(".btn-text").textContent = "Generating...";

    const startTime = performance.now();

    try {
      const payload = {
        prompt: prompt,
        model_name: modelSelect.value,
        max_tokens: parseInt(sliderMaxTokens.value, 10),
        temperature: parseFloat(sliderTemp.value),
        top_k: parseInt(sliderTopK.value, 10),
      };

      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);

      // Render output
      emptyState.style.display = "none";
      generatedContent.style.display = "block";
      displayPrompt.textContent = data.prompt + " ";
      displayContinuation.textContent = data.continuation;

      // Render metrics
      metricsBar.style.display = "grid";
      metricTokens.textContent = data.tokens_generated;
      metricLatency.textContent = `${elapsed}s`;
      metricModel.textContent = data.model_name.toUpperCase();
      metricSampling.textContent = `${data.temperature} / ${data.top_k}`;

      // Refresh history
      loadHistory();
    } catch (err) {
      alert(`Text Generation Error: ${err.message}`);
    } finally {
      btnGenerate.disabled = false;
      genSpinner.style.display = "none";
      btnGenerate.querySelector(".btn-text").textContent = "Generate AI Text";
    }
  });

  // 9. Load History
  async function loadHistory() {
    try {
      const res = await fetch("/api/history?limit=30");
      if (!res.ok) return;

      const items = await res.json();
      historyCount.textContent = `${items.length} ${items.length === 1 ? "entry" : "entries"}`;

      if (items.length === 0) {
        historyList.innerHTML = `<div class="history-empty">No generation history yet. Try generating your first text above!</div>`;
        return;
      }

      historyList.innerHTML = items
        .map((item) => {
          const dateStr = item.created_at ? new Date(item.created_at).toLocaleTimeString() : "";
          return `
            <div class="history-item" data-id="${item.id}">
              <div class="history-item-top">
                <div class="history-tags">
                  <span class="history-tag">${item.model_name.toUpperCase()}</span>
                  <span>Tokens: ${item.tokens_generated}</span>
                  <span>Temp: ${item.temperature}</span>
                </div>
                <span>${dateStr}</span>
              </div>
              <div class="history-text">
                <strong>${escapeHtml(item.prompt)}</strong> ${escapeHtml(item.continuation)}
              </div>
              <div class="history-actions">
                <button class="btn btn-secondary btn-sm btn-reuse" data-prompt="${escapeHtml(item.prompt)}">Reuse Prompt</button>
                <button class="btn btn-danger-outline btn-sm btn-delete-item" data-id="${item.id}">Delete</button>
              </div>
            </div>
          `;
        })
        .join("");

      // Bind reuse buttons
      document.querySelectorAll(".btn-reuse").forEach((btn) => {
        btn.addEventListener("click", () => {
          promptInput.value = btn.getAttribute("data-prompt");
          promptInput.dispatchEvent(new Event("input"));
          window.scrollTo({ top: 0, behavior: "smooth" });
        });
      });

      // Bind delete buttons
      document.querySelectorAll(".btn-delete-item").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const id = btn.getAttribute("data-id");
          await deleteHistoryItem(id);
        });
      });
    } catch (err) {
      console.error("Failed to load history:", err);
    }
  }

  // 10. Delete Single History Item
  async function deleteHistoryItem(id) {
    try {
      const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
      if (res.ok) {
        loadHistory();
      }
    } catch (err) {
      console.error("Failed to delete history item:", err);
    }
  }

  // 11. Clear All History
  btnClearHistory.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to clear all generation history?")) return;
    try {
      const res = await fetch("/api/history", { method: "DELETE" });
      if (res.ok) {
        loadHistory();
      }
    } catch (err) {
      console.error("Failed to clear history:", err);
    }
  });

  // Helper
  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  // Initial runs
  checkHealth();
  loadModels();
  loadHistory();
});

