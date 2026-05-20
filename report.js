async function main() {
  const data = await fetch("report-data.json").then((r) => r.json());
  const variants = Object.fromEntries(data.variants.map((v) => [v.id, v]));
  const fmt = (x, n = 4) => (x == null ? "-" : Number(x).toFixed(n));
  const pct = (x) => `${(100 * Number(x || 0)).toFixed(1)}%`;
  const esc = (s) =>
    String(s ?? "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    })[c]);
  document.head.insertAdjacentHTML(
    "beforeend",
    "<style>.charts{grid-template-columns:repeat(4,minmax(220px,1fr))}.chart{min-width:0}.score-note{margin-top:8px}.finetuned-label{color:#cf222e;font-weight:750}</style>",
  );
  const shortModelName = (name) =>
    String(name || "")
      .replace("Qwen3-VL-Embedding-", "Qwen-")
      .replace("Qwen/Qwen3-VL-Embedding-", "Qwen-");
  const shortVariantLabel = (variantOrId) => {
    const variant = typeof variantOrId === "string" ? variants[variantOrId] : variantOrId;
    if (!variant) return "-";
    const model = data.models.find((m) => m.id === variant.model_id);
    const name = shortModelName(model?.name || variant.label);
    if (model?.is_baseline) return name;
    return variant.epoch === 0 ? `${name} Base` : `${name} Epoch ${variant.epoch}`;
  };
  const modelById = (id) => data.models.find((m) => m.id === id);
  const variantIdFor = (modelId, kind) => {
    const model = modelById(modelId);
    return model ? model[kind] : null;
  };
  const orderedModelIds = () => [
    "qwen8b",
    "qwen2b",
    "hpsv2_1",
    "clip_vit_l14_openai",
  ].filter((id) => modelById(id));
  const orderedModels = () => orderedModelIds().map(modelById);
  const orderedComparisonVariantIds = () => [
    variantIdFor("qwen8b", "best_variant"),
    variantIdFor("qwen2b", "best_variant"),
    variantIdFor("hpsv2_1", "best_variant"),
    variantIdFor("clip_vit_l14_openai", "best_variant"),
    variantIdFor("qwen8b", "base_variant"),
    variantIdFor("qwen2b", "base_variant"),
  ].filter(Boolean);
  const orderedBestVariantIds = () => [
    variantIdFor("qwen8b", "best_variant"),
    variantIdFor("qwen2b", "best_variant"),
    variantIdFor("hpsv2_1", "best_variant"),
    variantIdFor("clip_vit_l14_openai", "best_variant"),
  ].filter(Boolean);
  const isFineTunedVariant = (variantOrId) => {
    const variant = typeof variantOrId === "string" ? variants[variantOrId] : variantOrId;
    const model = variant ? modelById(variant.model_id) : null;
    return Boolean(variant && model && !model.is_baseline && variant.epoch > 0);
  };
  const variantLabelHtml = (variantOrId) => {
    const label = shortVariantLabel(variantOrId);
    return isFineTunedVariant(variantOrId) ? `<span class="finetuned-label">${label}</span>` : label;
  };
  const stat = (value, label) =>
    `<div class="card stat"><div class="value">${value}</div><div class="label">${label}</div></div>`;

  function lineChart(title, rows, series, yLabel) {
    const w = 680;
    const h = 270;
    const m = { l: 52, r: 18, t: 22, b: 42 };
    const xs = rows.map((r) => r.epoch);
    const vals = [];
    series.forEach((s) => rows.forEach((r) => r[s.key] != null && vals.push(r[s.key])));
    const minY = Math.min(...vals);
    const maxY = Math.max(...vals);
    const pad = (maxY - minY || 1) * 0.12;
    const y0 = minY - pad;
    const y1 = maxY + pad;
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const x = (ep) => m.l + ((ep - minX) / (maxX - minX || 1)) * (w - m.l - m.r);
    const y = (v) => h - m.b - ((v - y0) / (y1 - y0)) * (h - m.t - m.b);
    const grid = [0, 0.25, 0.5, 0.75, 1].map((t) => {
      const v = y0 + t * (y1 - y0);
      return `<g><line x1="${m.l}" x2="${w - m.r}" y1="${y(v)}" y2="${y(v)}" stroke="#eceff3"/><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="#656d76">${fmt(v, 2)}</text></g>`;
    }).join("");
    const paths = series.map((s) => {
      const pts = rows.filter((r) => r[s.key] != null).map((r) => `${x(r.epoch)},${y(r[s.key])}`).join(" ");
      const dots = rows.filter((r) => r[s.key] != null).map((r) => `<circle cx="${x(r.epoch)}" cy="${y(r[s.key])}" r="3" fill="${s.color}"><title>${s.label} epoch ${r.epoch}: ${fmt(r[s.key], 5)}</title></circle>`).join("");
      return `<polyline fill="none" stroke="${s.color}" stroke-width="2.5" points="${pts}"/>${dots}`;
    }).join("");
    const xt = xs.map((ep) => `<text x="${x(ep)}" y="${h - 16}" text-anchor="middle" font-size="11" fill="#656d76">${ep}</text>`).join("");
    return `<div class="card chart"><h3>${title}</h3><svg viewBox="0 0 ${w} ${h}"><line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${h - m.b}" stroke="#9da7b1"/><line x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}" stroke="#9da7b1"/>${grid}${paths}${xt}<text x="${w / 2}" y="${h - 2}" text-anchor="middle" font-size="11" fill="#656d76">Epoch, 0 = base model</text><text transform="translate(13 ${h / 2}) rotate(-90)" text-anchor="middle" font-size="11" fill="#656d76">${yLabel}</text></svg><div class="legend">${series.map((s) => `<span style="color:${s.color}">${s.label}</span>`).join("")}</div></div>`;
  }

  function renderOverview() {
    const cards = orderedModels().map((model) => {
      const base = variants[model.base_variant];
      const best = variants[model.best_variant];
      const modelName = shortModelName(model.name);
      if (model.is_baseline) {
        return [
          stat(pct(best.summary.accuracy), `${modelName} GT-vs-AI accuracy`),
        ].join("");
      }
      return [
        stat(pct(best.summary.accuracy), `${modelName} best GT-vs-AI accuracy`),
        stat(pct(base.summary.accuracy), `${modelName} base GT-vs-AI accuracy`),
      ].join("");
    }).join("");
    document.getElementById("overview").innerHTML = `<div class="grid stats">${cards}</div>`;
  }

  function renderMetrics() {
    const trainedModels = orderedModels().filter((model) => !model.is_baseline);
    document.getElementById("metrics").innerHTML = `<h2>Training And Validation Curves</h2><p class="muted">Each model is trained on the same vtracer-free Text-to-SVG split. Epoch 0 is the untrained base embedding model with cosine scoring.</p><div class="grid charts">${trainedModels.map((model) => `${lineChart(`${shortModelName(model.name)}: Loss`, model.epoch_metrics, [{ key: "train_loss", label: "train loss", color: "#0969da" }, { key: "validation_loss", label: "validation loss", color: "#cf222e" }], "Loss")}${lineChart(`${shortModelName(model.name)}: Ranking Metrics`, model.epoch_metrics, [{ key: "pairwise_accuracy", label: "pairwise accuracy", color: "#0969da" }, { key: "hit_at_1", label: "hit@1", color: "#1a7f37" }, { key: "mrr", label: "MRR", color: "#9a6700" }], "Metric")}`).join("")}</div>`;
  }

  function renderModelCompare() {
    const rows = orderedModels().map((model) => {
      const base = variants[model.base_variant];
      const best = variants[model.best_variant];
      if (model.is_baseline) {
        return `<tr><td>${shortModelName(model.name)}</td><td>-</td><td>-</td><td><b>${pct(best.summary.accuracy)}</b></td><td><b>${fmt(best.metrics.mrr, 3)}</b></td><td><b>${fmt(best.metrics.hit_at_1, 3)}</b></td><td><b>${fmt(best.metrics.pairwise_accuracy, 3)}</b></td></tr>`;
      }
      return `<tr><td>${shortModelName(model.name)}</td><td>${model.embedding_dim}</td><td>${model.best_epoch}</td><td>${pct(base.summary.accuracy)} → <b>${pct(best.summary.accuracy)}</b></td><td>${fmt(base.metrics.mrr, 3)} → <b>${fmt(best.metrics.mrr, 3)}</b></td><td>${fmt(base.metrics.hit_at_1, 3)} → <b>${fmt(best.metrics.hit_at_1, 3)}</b></td><td>${fmt(base.metrics.pairwise_accuracy, 3)} → <b>${fmt(best.metrics.pairwise_accuracy, 3)}</b></td></tr>`;
    }).join("");
    document.getElementById("modelcompare").innerHTML = `<h2>Lica vs CLIP / HPSv2 Summary</h2><table><tbody><tr><th>Model</th><th>Embedding dim</th><th>Best epoch</th><th>GT-vs-AI accuracy</th><th>MRR</th><th>Hit@1</th><th>Pairwise acc.</th></tr>${rows}</tbody></table>`;
  }

  function confusionCard(variant) {
    return `<div class="card"><h3>${variantLabelHtml(variant)}</h3><table><tbody><tr><th>Actual winner</th><th>Predicted GT</th><th>Predicted AI</th><th>Accuracy</th></tr><tr><td>GT</td><td class="good">${variant.summary.confusion.true_gt_pred_gt}</td><td class="${variant.summary.confusion.true_gt_pred_ai ? "bad" : "good"}">${variant.summary.confusion.true_gt_pred_ai}</td><td>${pct(variant.summary.accuracy)}</td></tr></tbody></table><h3 style="margin-top:14px">By AI source</h3><table><tbody>${Object.entries(variant.summary.by_source).map(([src, v]) => `<tr><td>${src}</td><td>${v.correct} / ${v.total}</td><td>${pct(v.accuracy)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  function renderGtAi() {
    const ids = orderedComparisonVariantIds();
    document.getElementById("gtai").innerHTML = `<h2>GT vs AI Accuracy And Confusion Matrix</h2><p class="muted">GT is assumed winner; Claude/Gemini/GPT-5.2 are assumed losers.</p><div class="confusion">${ids.map((id) => confusionCard(variants[id])).join("")}</div>`;
  }

  function renderGallery() {
    const displayIds = orderedComparisonVariantIds();
    document.getElementById("gallery").innerHTML = `<h2>Validation Render Gallery</h2><p class="muted">15 validation groups. Scores are scaled cosine logits, not 0-1 probabilities.</p>${data.groups.map((g) => `<div class="card group-card"><h3>${g.group_id} <span class="tag">${g.bucket}</span></h3><div class="prompt">${esc(g.prompt)}</div><div class="small muted">${displayIds.map((id) => `${variantLabelHtml(id)}: <b>${g.winners[id]}</b>`).join(" · ")}</div><div class="renders">${g.entries.map((e) => `<div class="render"><img src="${e.image}" alt="${e.source} render"><div class="body"><div class="source">${e.source}</div>${displayIds.map((id) => `<div class="score-row"><span>${variantLabelHtml(id)}</span><b>${fmt(e.scores[id], 3)}</b></div>`).join("")}</div></div>`).join("")}</div><div class="small muted score-note">Score = learned logit_scale × cosine_similarity(prompt, render). Higher is better; values can exceed 1.</div></div>`).join("")}`;
  }

  function renderAiVai() {
    const displayIds = orderedBestVariantIds();
    const winCounts = Object.fromEntries(displayIds.map((id) => [id, {}]));
    data.ai_comparisons.forEach((c) => displayIds.forEach((id) => {
      winCounts[id][c.winners[id]] = (winCounts[id][c.winners[id]] || 0) + 1;
    }));
    document.getElementById("aivai").innerHTML = `<h2>AI Generated Results Compared Against Each Other</h2><p class="muted">Pairwise winners among GPT-5.2, Claude, and Gemini for each best ranker.</p><div class="grid stats">${displayIds.map((id) => stat(Object.entries(winCounts[id]).map(([k, v]) => `${k}: ${v}`).join("<br>"), `${variantLabelHtml(id)} AI-vs-AI wins`)).join("")}</div><div class="controls"><label>Filter pair <select id="pairFilter"><option value="all">all</option><option>gpt-5.2 vs claude</option><option>claude vs gemini</option><option>gpt-5.2 vs gemini</option></select></label></div><div id="aiCompareGrid" class="compare-grid"></div>`;
    const draw = () => {
      const f = document.getElementById("pairFilter").value;
      const rows = data.ai_comparisons.filter((c) => f === "all" || c.pair === f);
      document.getElementById("aiCompareGrid").innerHTML = rows.map((c) => `<div class="card"><h3>${c.pair} <span class="tag">${c.bucket}</span></h3><div class="small muted">${c.group_id}</div><div class="compare-images"><img src="${c.left.image}" alt="${c.left.source}"><img src="${c.right.image}" alt="${c.right.source}"></div><table><tbody><tr><th>model</th><th>${c.left.source}</th><th>${c.right.source}</th><th>winner</th></tr>${displayIds.map((id) => `<tr><td>${variantLabelHtml(id)}</td><td>${fmt(c.left.scores[id], 3)}</td><td>${fmt(c.right.scores[id], 3)}</td><td><b>${c.winners[id]}</b></td></tr>`).join("")}</tbody></table></div>`).join("");
    };
    draw();
    document.getElementById("pairFilter").addEventListener("change", draw);
  }

  renderOverview();
  renderMetrics();
  renderModelCompare();
  renderGtAi();
  renderGallery();
  renderAiVai();
}

main().catch((err) => {
  document.body.innerHTML = `<pre style="padding:24px;color:#cf222e;white-space:pre-wrap">${String(err.stack || err)}</pre>`;
  console.error(err);
});
