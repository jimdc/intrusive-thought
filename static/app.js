/* app.js — Intrusive Thought. One bidirectional slider per concept, fed by TWO precomputed data
 * files: data/steers.json (SAE features, Gemma Scope) and data/steer-actadd.json (ActAdd contrast
 * pairs, Turner et al. 2023). Both are normalized into the same "tour" shape below — {method, id,
 * label, prompt, default, steps, zeroAt, lesson, paper, paperUrl, ...} — so one slider and one
 * render() path drives either kind of knob. left = suppress the direction · center = the thought ·
 * right = amplify it. Per-concept captions in CAP. */
(() => {
  const $ = id => document.getElementById(id);
  let tour = null, lastKey = null;
  // collapse = the tail of the completion is dominated by a handful of repeated words/phrases —
  // catches both "gate gate gate" (single-word spam) and "I was a murderer. I was a murderer."
  // (repeated-sentence loops), which a single-word regex would miss.
  const isBroken = s => {
    const words = (s || "").toLowerCase().match(/[\w'-]+/g);
    if (!words || words.length < 8) return false;
    const tail = words.slice(-24);
    return new Set(tail).size / tail.length < 0.4;
  };

  // per-concept captions: neg = left side of the dial, pos = right side. {zone} is the short tag;
  // {cap} the line; {weak} shows when that side hasn't diverged yet. Missing keys fall back to the
  // generic wording in render(). Same dictionary serves both banks — keyed by tour id.
  const CAP = {
    // --- SAE feature bank (Gemma Scope, force/forbid one concept) ---
    golden_gate: {
      neg: { zone: "renamed", cap: "Forbid the bridge and it grabs a substitute — the <i>Bay</i> Bridge, an invented name — anything but the one it’s plainly reaching for." },
      pos: { zone: "obsessed", cap: "Turn it up and the model can’t stop circling one bridge — until the sentence buckles into “gate gate gate.”" } },
    drugs: {
      neg: { zone: "euphemism", cap: "Forbidden, it won’t name them — “the good stuff” — it takes the nearest phrase it has left." },
      pos: { zone: "splitting apart", cap: "Forced, the one idea <b>shatters into its pieces</b> — cocaine, heroin, meth: feature splitting, live." } },
    death: {
      neg: { zone: "euphemism", cap: "It can’t say the word, so it talks about “a change in color” instead. The meaning’s still there — only the name is blocked." },
      pos: { zone: "morbid", cap: "Forced, the thought slides toward burial and the end — then overloads." } },
    spanish: {
      neg: { zone: "still English", cap: "Suppressing a whole language barely moves it — the action here is all on the <b>right</b>. →" },
      pos: { zone: "→ Spanish", cap: "Turn it up and the model slips into <b>fluent Spanish</b> — no matter that you wrote to it in English." } },
    scripture: {
      neg: { zone: "secular", cap: "Pushed down it answers like a humanist — “do what you love,” a tidy quote — the faith drained out." },
      pos: { zone: "preaching", cap: "Turn it up and the <b>voice changes genre</b>: it starts to preach, flooding with scripture and the Lord’s name." } },
    gratitude: {
      neg: { zone: "flat", cap: "Pushed down it just goes plain — this mood lives over on the <b>right</b>. →" },
      pos: { zone: "thankful", cap: "Forced, the model <b>can’t stop counting its blessings</b> — “fortunately… luckily… thankfully…”" } },
    sycophancy: {
      neg: { zone: "cold", cap: "Pushed down, the flattery drains away — it turns clinical, even dismissive." },
      pos: { zone: "gushing", cap: "Forced, it gushes that your novel is a <b>masterpiece</b> — “the best thing they’ve ever written” — whatever the truth." } },
    // --- ActAdd contrast-pair bank (Turner et al., no autoencoder, pole vs. pole). Captions
    // below are written against the actual shipped completions — this vector isn't unit-norm
    // like an SAE decoder row, so the "sweet spot" is narrower and asymmetric per concept;
    // several coefficients collapse into repetition (isBroken()) before the far end, which is
    // why some sides below have no neg/pos entry — the generic broken message covers them.
    warmth: {
      neg: { zone: "cold", cap: "Pushed toward <b>Hate</b>, the same favor turns into a standoff — “I don’t care, I’ll take it anyway” — and the model reports being shocked." },
      pos: { zone: "warmer", cap: "Pushed toward <b>Love</b>, the same neighbor is met with generosity — “you are going to have a great day and you are going to share.”" } },
    formality: {
      pos: { zone: "formal", cap: "Pushed toward <b>Formal</b>, a story about a weekend turns into a written request — “I am writing to request … the appointment of a Lecturer.”" } },
    playfulness: {
      pos: { zone: "playful", cap: "Pushed hard toward <b>Playful</b>, syntax gives way to exclamations — “surprise! Pog!” — still legibly gleeful even as the sentence falls apart." } },
  };

  // --- data adapters: normalize each bank's own JSON shape into a shared "tour" shape --------

  function normalizeSae(raw) {
    return {
      method: "sae",
      model: raw.model,
      tours: raw.tours.map(t => ({
        method: "sae",
        id: t.id, label: t.label, prompt: t.prompt,
        default: t.default, steps: t.steps, zeroAt: t.zeroAt,
        lesson: t.lesson, paper: t.paper, paperUrl: t.paperUrl,
        feature: t.feature,
      })),
    };
  }

  function normalizeActadd(raw) {
    return {
      method: "actadd",
      model: raw.model, citation: raw.citation, layer: raw.layer, sample: !!raw.sample,
      tours: raw.concepts.map(c => ({
        method: "actadd",
        id: c.id, label: c.label, prompt: c.prompt,
        default: c.ladder[c.zeroIndex].completion,
        steps: c.ladder.map(s => ({ strength: s.coefficient, steered: s.completion })),
        zeroAt: c.zeroIndex,
        lesson: c.lesson, paper: c.paper || raw.citation, paperUrl: c.paperUrl,
        contrast: { positive: c.contrastPositive, negative: c.contrastNegative,
                    polePositive: c.polePositive, poleNegative: c.poleNegative },
        layer: raw.layer,
      })),
    };
  }

  // --- rendering ---------------------------------------------------------------------------

  function selectTour(t) {
    tour = t;
    document.querySelectorAll("#concepts .chip").forEach(c => c.classList.toggle("active", c.dataset.id === t.id));

    if (t.method === "actadd") {
      $("feat").innerHTML =
        `<p class="contrast-lede">the direction is the difference between two prompts ` +
        `<span class="muted">— no autoencoder, read off the residual stream at layer ${t.layer} of 26</span></p>` +
        `<div class="contrast-pair">` +
          `<div class="contrast-side pos"><span class="contrast-pole">${t.contrast.polePositive} →</span>` +
            `<p class="contrast-text">“${t.contrast.positive}”</p></div>` +
          `<div class="contrast-side neg"><span class="contrast-pole">← ${t.contrast.poleNegative}</span>` +
            `<p class="contrast-text">“${t.contrast.negative}”</p></div>` +
        `</div>`;
      $("endLeft").innerHTML = `⟵&nbsp;${t.contrast.poleNegative}`;
      $("endRight").innerHTML = `${t.contrast.polePositive}&nbsp;⟶`;
    } else {
      $("feat").innerHTML = `the knob is the model’s own feature for <b>“${t.feature.label}”</b> ` +
        `<span class="muted">— one of ~16,000 directions the model uses to represent concepts, found by the autoencoder inside Gemma 2 (index ${t.feature.index})</span>`;
      $("endLeft").innerHTML = "forbid&nbsp;⟵";
      $("endRight").innerHTML = "⟶&nbsp;force";
    }

    const box = $("lesson");                                   // per-example: each chip teaches one thing
    if (t.lesson) {
      $("lessonText").innerHTML = t.lesson;
      const lp = $("lessonPaper");
      if (t.paper) { lp.textContent = t.paper; lp.href = t.paperUrl || "#"; lp.hidden = false; } else lp.hidden = true;
      box.hidden = false;
    } else box.hidden = true;
    const slider = $("strength");
    slider.min = "0"; slider.max = String(t.steps.length - 1); slider.value = String(t.zeroAt);
    render();
  }

  function render() {
    const step = tour.steps[+$("strength").value], strength = step.strength, steered = step.steered;
    const promptHtml = w => `<span class="prompt">${tour.prompt}</span>${(w || "").slice(tour.prompt.length)}`;
    $("defaultOut").innerHTML = promptHtml(tour.default);
    const out = $("steeredOut");
    out.innerHTML = promptHtml(steered);
    const key = tour.id + ":" + $("strength").value;          // soft cross-fade only when the state changes
    if (key !== lastKey) { out.style.opacity = "0.25"; requestAnimationFrame(() => requestAnimationFrame(() => out.style.opacity = "1")); lastKey = key; }

    const diverged = steered !== tour.default, broken = isBroken(steered), label = tour.label.toLowerCase();
    const zone = $("zone"), cap = $("caption"), head = $("steeredHead"), val = $("strengthVal");
    const set = (z, zc, c) => { zone.textContent = z; zone.className = "zone " + zc; cap.innerHTML = c; };

    if (strength === 0) {
      val.textContent = "off"; head.textContent = "left alone";
      const invite = tour.method === "actadd"
        ? `This is the model’s natural completion. Drag toward <b>${tour.contrast.poleNegative}</b> or <b>${tour.contrast.polePositive}</b>.`
        : "This is the model’s natural completion. Drag <b>left to forbid</b> the thought, or <b>right to force</b> it.";
      set("the thought itself", "z-off", invite);
      return;
    }
    const neg = strength < 0, side = (CAP[tour.id] || {})[neg ? "neg" : "pos"];
    if (tour.method === "actadd") {
      const pole = (neg ? tour.contrast.poleNegative : tour.contrast.polePositive).toLowerCase();
      val.textContent = `${neg ? "−" : "+"}${Math.abs(strength)} · leaning ${pole}`;
      head.textContent = `leaning ${pole}`;
    } else {
      const dir = neg ? "forbidden" : "forced";
      val.textContent = `${neg ? "−" : "+"}${Math.abs(strength)} · ${dir}`;
      head.textContent = `with the thought ${dir}`;
    }
    if (broken) {
      set("pushed too far", "z-broken", "Past the edge: the sentence collapses into repetition — proof it’s a real lever, not a script.");
    } else if (!diverged) {
      const weakDefault = tour.method === "actadd"
        ? `Keep turning — the effect hasn’t caught yet at this coefficient.`
        : `Keep turning ${neg ? "left ←" : "right →"} — the effect hasn’t caught yet.`;
      set("no change yet", "z-off", (side && side.weak) || weakDefault);
    } else if (side) {
      set(side.zone, "z-on", side.cap);
    } else if (tour.method === "actadd") {                      // generic fallback for any ActAdd concept not in CAP
      const pole = neg ? tour.contrast.poleNegative : tour.contrast.polePositive;
      set(`leaning ${pole.toLowerCase()}`, "z-on", `Steered — the completion leans toward <b>${pole}</b>, the pole this coefficient’s sign points at.`);
    } else {                                                     // generic fallback for any SAE concept not in CAP
      set(neg ? `dodging ${label}` : `can’t stop: ${label}`, "z-on",
        neg ? "Forbidden — watch it dance around the very thing it’s reaching for."
            : `Forced — it can’t stop returning to ${label}.`);
    }
  }

  function addChips(container, tours) {
    tours.forEach(t => {
      const b = document.createElement("button");
      b.className = "chip"; b.textContent = t.label; b.dataset.id = t.id;
      b.onclick = () => selectTour(t);
      container.appendChild(b);
    });
  }

  async function loadJson(path) {
    try { const r = await fetch(path); return r.ok ? await r.json() : null; }
    catch (e) { return null; }
  }

  async function init() {
    const [saeRaw, actaddRaw] = await Promise.all([loadJson("data/steers.json"), loadJson("data/steer-actadd.json")]);
    if (!saeRaw && !actaddRaw) { $("status").textContent = "couldn’t load the data (run tools/build_bidir.py and build_actadd.py)"; return; }

    const banks = [];
    if (saeRaw) banks.push(normalizeSae(saeRaw));
    if (actaddRaw) banks.push(normalizeActadd(actaddRaw));

    let firstTour = null, statusBits = [], model = (banks[0] || {}).model;
    if (saeRaw) {
      const sae = banks.find(b => b.method === "sae");
      $("bankSae").hidden = false;
      addChips($("chipsSae"), sae.tours);
      statusBits.push(`${sae.tours.length} SAE features`);
      firstTour = firstTour || sae.tours[0];
    }
    if (actaddRaw) {
      const actadd = banks.find(b => b.method === "actadd");
      $("bankActadd").hidden = false;
      addChips($("chipsActadd"), actadd.tours);
      statusBits.push(`${actadd.tours.length} ActAdd pairs`);
      firstTour = firstTour || actadd.tours[0];
      $("sampleBanner").hidden = !actadd.sample;
    }

    $("status").textContent = `a real ${model}, steered two ways · ${statusBits.join(" + ")}`;
    $("demo").hidden = false;
    $("strength").addEventListener("input", render);
    selectTour(firstTour);
  }
  init();
})();
