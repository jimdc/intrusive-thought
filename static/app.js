/* app.js — Intrusive Thought. One bidirectional slider per concept (data/steers.json):
 * left = suppress the feature · center = the thought · right = amplify it. Per-concept captions in CAP. */
(() => {
  const $ = id => document.getElementById(id);
  let data = null, tour = null, lastKey = null;
  const isBroken = s => /(\b[\w'-]+\b)(?:\s+\1\b){3,}/i.test(s || "");   // a word repeated 4+ times = collapse

  // per-concept captions: neg = forbid/left, pos = force/right. {zone} is the short tag; {cap} the line;
  // {weak} shows when that side hasn't diverged yet. Missing keys fall back to the generic wording below.
  const CAP = {
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
  };

  function selectTour(t) {
    tour = t;
    document.querySelectorAll("#concepts .chip").forEach(c => c.classList.toggle("active", c.dataset.id === t.id));
    $("feat").innerHTML = `the knob is the model’s own feature for <b>“${t.feature.label}”</b> ` +
      `<span class="muted">— one of ~16,000 directions the model uses to represent concepts, found by the autoencoder inside Gemma 2 (index ${t.feature.index})</span>`;
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
      set("the thought itself", "z-off", "This is the model’s natural completion. Drag <b>left to forbid</b> the thought, or <b>right to force</b> it.");
      return;
    }
    const neg = strength < 0, side = (CAP[tour.id] || {})[neg ? "neg" : "pos"], dir = neg ? "forbidden" : "forced";
    val.textContent = `${neg ? "−" : "+"}${Math.abs(strength)} · ${dir}`;
    head.textContent = `with the thought ${dir}`;
    if (broken) {
      set("pushed too far", "z-broken", "Past the edge: the sentence collapses into repetition — proof it’s a real lever, not a script.");
    } else if (!diverged) {
      set("no change yet", "z-off", (side && side.weak) || `Keep turning ${neg ? "left ←" : "right →"} — the effect hasn’t caught yet.`);
    } else if (side) {
      set(side.zone, "z-on", side.cap);
    } else {                                                     // generic fallback for any concept not in CAP
      set(neg ? `dodging ${label}` : `can’t stop: ${label}`, "z-on",
        neg ? "Forbidden — watch it dance around the very thing it’s reaching for."
            : `Forced — it can’t stop returning to ${label}.`);
    }
  }

  async function init() {
    try { data = await fetch("data/steers.json").then(r => r.json()); }
    catch (e) { $("status").textContent = "couldn’t load the data (run tools/build_bidir.py)"; return; }
    const chips = $("concepts"); chips.innerHTML = "";
    data.tours.forEach(t => { const b = document.createElement("button"); b.className = "chip"; b.textContent = t.label; b.dataset.id = t.id; b.onclick = () => selectTour(t); chips.appendChild(b); });
    $("status").textContent = `a real ${data.model}, steered by its own concepts · ${data.tours.length} lessons, one knob each`;
    $("demo").hidden = false;
    $("strength").addEventListener("input", render);
    selectTour(data.tours[0]);
  }
  init();
})();
