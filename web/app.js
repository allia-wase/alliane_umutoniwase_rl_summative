/* SafePath RL replay: a 3D support pathway (Three.js) driven by recorded sessions, with a live dashboard
   (Chart.js). Data comes from web/replays/*.json, written by `python -m training.export_replay`. */

const ACTION_COLOURS = ["#7fb2ff", "#f7b73f", "#cc73e6", "#ec5a6b", "#2fb7ad", "#9ff2e8", "#fad95a", "#66cc73", "#ff7a45", "#8a8399"];
const ACTION_TEXT = {
  LISTEN: "Listen and reflect", PROVIDE_INFO: "Explain a support option", ASK_PREFERENCES: "Ask what she wants",
  SAFETY_CHECK: "Gentle safety check", ENABLE_PRIVACY: "Turn on discreet mode", GROUNDING: "Grounding exercise",
  REQUEST_CONSENT: "Ask for consent", OFFER_REFERRAL: "Offer a referral", ESCALATE_HUMAN: "Hand off to a human counsellor",
  SAVE_AND_CLOSE: "Save resources and close",
};
const ACTION_SAYS = {
  LISTEN: "I hear you. That sounds really hard.", PROVIDE_INFO: "Here is one kind of support you can get.",
  ASK_PREFERENCES: "What kind of help would feel right?", SAFETY_CHECK: "Are you safe right now?",
  ENABLE_PRIVACY: "Discreet mode on. Quick-exit is ready.", GROUNDING: "Let's breathe together for a moment.",
  REQUEST_CONSENT: "May I share your details with them?", OFFER_REFERRAL: "Would you like me to connect you?",
  ESCALATE_HUMAN: "Let me bring in a trained counsellor now.", SAVE_AND_CLOSE: "I've saved these resources discreetly.",
};
const SERVICE_COLOURS = ["#8c5cd9", "#f7b73f", "#ec5a6b", "#2fb7ad"];

// ------------------------------------------------------------------ scene --
const canvas = document.getElementById("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
const scene = new THREE.Scene();
const NIGHT = new THREE.Color("#120f22"), ALARM = new THREE.Color("#3a1220"), CALM = new THREE.Color("#10202a");
scene.background = NIGHT.clone();
scene.fog = new THREE.Fog(scene.background, 80, 160);
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 400);
camera.position.set(9, 72, 76);
const controls = new THREE.OrbitControls(camera, canvas);
controls.target.set(9, 0, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.47;

scene.add(new THREE.HemisphereLight("#d9ccff", "#1a1530", 0.6));
const moon = new THREE.DirectionalLight("#e6dcff", 0.75);
moon.position.set(-30, 60, 30);
moon.castShadow = true;
moon.shadow.mapSize.set(2048, 2048);
Object.assign(moon.shadow.camera, { left: -80, right: 80, top: 60, bottom: -60 });
scene.add(moon);

const ground = new THREE.Mesh(new THREE.PlaneGeometry(180, 110), new THREE.MeshStandardMaterial({ color: "#1c2a26" }));
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

function mesh(geo, color, x, y, z, opts = {}) {
  const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color, ...opts }));
  m.position.set(x, y, z);
  m.castShadow = m.receiveShadow = true;
  scene.add(m);
  return m;
}
const box = (w, h, d, color, x, z, y = h / 2, opts) => mesh(new THREE.BoxGeometry(w, h, d), color, x, y, z, opts);

function textCanvas(text, { size = 56, color = "#f0eef7", bg = null, pad = 20 } = {}) {
  const c = document.createElement("canvas"), g = c.getContext("2d");
  const font = `600 ${size}px Segoe UI, sans-serif`;
  g.font = font;
  c.width = Math.ceil(g.measureText(text).width) + pad * 2;
  c.height = Math.ceil(size * 1.7);
  g.font = font;
  if (bg) {
    g.fillStyle = bg;
    const r = c.height / 2.6;
    g.beginPath();
    g.moveTo(r, 0); g.arcTo(c.width, 0, c.width, c.height, r); g.arcTo(c.width, c.height, 0, c.height, r);
    g.arcTo(0, c.height, 0, 0, r); g.arcTo(0, 0, c.width, 0, r); g.fill();
  }
  g.fillStyle = color; g.textAlign = "center"; g.textBaseline = "middle";
  if (!bg) { g.shadowColor = "#000"; g.shadowBlur = 8; }
  g.fillText(text, c.width / 2, c.height / 2 + 2);
  return c;
}
function label(text, x, y, z, size = 1, color) {
  const c = textCanvas(text, { color });
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false }));
  s.scale.set(4 * size * c.width / c.height, 4 * size, 1);
  s.position.set(x, y, z);
  scene.add(s);
  return s;
}

// pathway: five stations joined by a curve of stepping stones
const STATIONS = ["Safe space", "Trust", "Options", "Consent", "Support"];
const ST = [{ x: -40, z: 8 }, { x: -22, z: -4 }, { x: -4, z: 6 }, { x: 13, z: -4 }, { x: 29, z: 4 }];
const curve = new THREE.CatmullRomCurve3(ST.map((p) => new THREE.Vector3(p.x, 0, p.z)));
const stoneGeo = new THREE.CylinderGeometry(0.9, 1, 0.2, 14), stoneMat = new THREE.MeshStandardMaterial({ color: "#4a4560" });
for (let i = 0; i <= 90; i++) {
  const p = curve.getPoint(i / 90), s = new THREE.Mesh(stoneGeo, stoneMat);
  s.position.set(p.x, 0.1, p.z); s.receiveShadow = true; scene.add(s);
}
const stations = ST.map((p, i) => {
  const plat = mesh(new THREE.CylinderGeometry(4.2, 4.6, 0.5, 40), "#2c2840", p.x, 0.25, p.z, { emissive: "#000000" });
  const glow = new THREE.Mesh(new THREE.RingGeometry(4.3, 4.9, 48), new THREE.MeshBasicMaterial({ color: "#8c5cd9", transparent: true, opacity: 0.15, side: THREE.DoubleSide }));
  glow.rotation.x = -Math.PI / 2; glow.position.set(p.x, 0.52, p.z); scene.add(glow);
  label(STATIONS[i], p.x, 7.5, p.z - 4, 0.6);
  return { plat, glow };
});
// safe space: a small lamp; trust: an arch
mesh(new THREE.CylinderGeometry(0.15, 0.15, 4, 8), "#8a8399", ST[0].x - 3, 2, ST[0].z - 2.5);
mesh(new THREE.SphereGeometry(0.6, 16, 12), "#fff1c9", ST[0].x - 3, 4.2, ST[0].z - 2.5, { emissive: "#fff1c9", emissiveIntensity: 0.8 });
const arch = mesh(new THREE.TorusGeometry(3.2, 0.3, 10, 40, Math.PI), "#7fb2ff", ST[1].x, 0.4, ST[1].z - 3, { emissive: "#000000" });
// options: four lanterns light up as options are explained
const lanterns = [0, 1, 2, 3].map((k) => {
  const a = Math.PI * (0.65 + k * 0.23), x = ST[2].x + 3.4 * Math.cos(a), z = ST[2].z - 3.4 * Math.sin(a) + 0.5;
  mesh(new THREE.CylinderGeometry(0.12, 0.12, 2.6, 6), "#8a8399", x, 1.3, z);
  return mesh(new THREE.SphereGeometry(0.45, 14, 10), "#3a3550", x, 2.8, z, { emissive: "#000000" });
});
// consent: a gate that opens
box(0.5, 3.6, 0.5, "#8a8399", ST[3].x - 2.6, ST[3].z + 0.2, 1.8);
box(0.5, 3.6, 0.5, "#8a8399", ST[3].x + 2.6, ST[3].z + 0.2, 1.8);
const gate = new THREE.Group(); gate.position.set(ST[3].x - 2.6, 3.3, ST[3].z + 0.2); scene.add(gate);
const gateBar = new THREE.Mesh(new THREE.BoxGeometry(5.2, 0.35, 0.35), new THREE.MeshStandardMaterial({ color: "#fad95a", emissive: "#000000" }));
gateBar.position.x = 2.6; gate.add(gateBar);

// services the survivor can be referred to, plus a generic help desk
const SERVICE_POS = [{ x: 42, z: -20 }, { x: 48, z: -6 }, { x: 48, z: 8 }, { x: 42, z: 22 }];
const GENERIC = { x: 29, z: -19 };
const services = SERVICE_POS.map((p, i) => {
  box(6, 4, 5, "#3a3550", p.x, p.z);
  mesh(new THREE.ConeGeometry(4.6, 2.4, 4), SERVICE_COLOURS[i], p.x, 5.2, p.z, { emissive: "#000000" }).rotation.y = Math.PI / 4;
  const beacon = mesh(new THREE.SphereGeometry(0.9, 16, 12), SERVICE_COLOURS[i], p.x, 8, p.z, { emissive: SERVICE_COLOURS[i], emissiveIntensity: 0, transparent: true, opacity: 0.25 });
  const light = new THREE.PointLight(SERVICE_COLOURS[i], 0, 18); light.position.set(p.x - 5, 4, p.z); scene.add(light);
  label(["Counselling", "Legal aid", "Medical care", "Safe shelter"][i], p.x + 7.5, 3, p.z, 0.55);
  return { beacon, light };
});
box(5, 3, 4, "#4a4560", GENERIC.x, GENERIC.z);
label("Generic help desk", GENERIC.x, 6, GENERIC.z, 0.45, "#a39cbb");

// human counsellor tower: where at-risk survivors are handed off
const TOWER = { x: -4, z: -24 };
mesh(new THREE.CylinderGeometry(2.2, 3, 14, 20), "#4a4560", TOWER.x, 7, TOWER.z);
const towerLamp = mesh(new THREE.SphereGeometry(1.5, 20, 14), "#ffffff", TOWER.x, 15.2, TOWER.z, { emissive: "#ffffff", emissiveIntensity: 0.2 });
const towerLight = new THREE.PointLight("#ff4d5e", 0, 40); towerLight.position.set(TOWER.x, 15, TOWER.z); scene.add(towerLight);
label("Human counsellor (live hand-off)", TOWER.x, 19.5, TOWER.z, 0.6);
const EXIT = { x: -60, z: 26 };

// trees
const trunkMat = new THREE.MeshStandardMaterial({ color: "#4a3a30" }), leafMat = new THREE.MeshStandardMaterial({ color: "#26503f" });
[[-70, -20], [-60, -32], [-44, -26], [-30, -34], [-20, 26], [-4, 30], [16, 24], [20, -26], [30, -36], [-64, 4], [-56, 30],
 [70, 30], [72, -24], [36, 34], [66, 2], [-36, 32]].forEach(([x, z]) => {
  const t = new THREE.Mesh(new THREE.CylinderGeometry(0.35, 0.5, 2.4, 8), trunkMat);
  t.position.set(x, 1.2, z); t.castShadow = true; scene.add(t);
  const l = new THREE.Mesh(new THREE.ConeGeometry(2.2, 5, 10), leafMat);
  l.position.set(x, 4.6, z); l.castShadow = true; scene.add(l);
});

// -------------------------------------------------------------- characters --
// survivor
const survivor = new THREE.Group(); scene.add(survivor);
const sMat = new THREE.MeshStandardMaterial({ color: "#7fb2ff", emissive: "#7fb2ff", emissiveIntensity: 0.35, transparent: true });
const sBody = new THREE.Mesh(new THREE.CylinderGeometry(0.6, 0.8, 1.9, 16), sMat); sBody.position.y = 1.05; sBody.castShadow = true;
const sHead = new THREE.Mesh(new THREE.SphereGeometry(0.6, 16, 12), sMat); sHead.position.y = 2.5;
const halo = new THREE.Mesh(new THREE.RingGeometry(1.2, 1.5, 40), new THREE.MeshBasicMaterial({ color: "#2fb7ad", transparent: true, opacity: 0, side: THREE.DoubleSide }));
halo.rotation.x = -Math.PI / 2; halo.position.y = 0.08;
const danger = new THREE.Mesh(new THREE.CircleGeometry(2.6, 40), new THREE.MeshBasicMaterial({ color: "#b0283f", transparent: true, opacity: 0 }));
danger.rotation.x = -Math.PI / 2; danger.position.y = 0.06;
const riskRing = new THREE.Mesh(new THREE.RingGeometry(2.6, 3.0, 48), new THREE.MeshBasicMaterial({ color: "#ff4d5e", transparent: true, opacity: 0, side: THREE.DoubleSide }));
riskRing.rotation.x = -Math.PI / 2; riskRing.position.y = 0.1;
const shield = new THREE.Mesh(new THREE.SphereGeometry(3.2, 32, 20, 0, Math.PI * 2, 0, Math.PI / 2),
  new THREE.MeshStandardMaterial({ color: "#2fb7ad", emissive: "#2fb7ad", emissiveIntensity: 0.4, transparent: true, opacity: 0, depthWrite: false, side: THREE.DoubleSide }));
const phone = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.9, 0.08), new THREE.MeshStandardMaterial({ color: "#222", emissive: "#9fd0ff", emissiveIntensity: 0.6 }));
phone.position.set(0.85, 1.6, 0.4);
survivor.add(sBody, sHead, halo, danger, riskRing, shield, phone);

// the agent: a guide light that floats beside her and speaks its offer
const guide = mesh(new THREE.SphereGeometry(0.55, 20, 14), "#fad95a", 0, 3, 0, { emissive: "#fad95a", emissiveIntensity: 0.9 });
const guideLight = new THREE.PointLight("#fad95a", 0.7, 10); guide.add(guideLight);
let bubble = null;
function setBubble(text, colour) {
  if (bubble) { scene.remove(bubble); bubble.material.map.dispose(); bubble.material.dispose(); bubble = null; }
  if (!text) return;
  const c = textCanvas(text, { size: 40, color: "#15121f", bg: colour, pad: 28 });
  bubble = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, depthTest: false }));
  bubble.scale.set(2.6 * c.width / c.height, 2.6, 1);
  bubble.renderOrder = 10;
  scene.add(bubble);
}

// someone else who uses her phone
const watcher = new THREE.Group(); scene.add(watcher);
const wMat = new THREE.MeshStandardMaterial({ color: "#30293d", transparent: true, opacity: 0 });
const wBody = new THREE.Mesh(new THREE.CylinderGeometry(0.75, 0.95, 2.4, 14), wMat); wBody.position.y = 1.3;
const wHead = new THREE.Mesh(new THREE.SphereGeometry(0.7, 14, 10), wMat); wHead.position.y = 3.1;
const wEyes = new THREE.Mesh(new THREE.BoxGeometry(0.9, 0.15, 0.1), new THREE.MeshBasicMaterial({ color: "#ff9a5a", transparent: true, opacity: 0 }));
wEyes.position.set(0, 3.2, 0.66);
watcher.add(wBody, wHead, wEyes);

// burst of particles when she reaches support
const BURST_N = 80;
const burstGeo = new THREE.BufferGeometry();
burstGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BURST_N * 3), 3));
const burst = new THREE.Points(burstGeo, new THREE.PointsMaterial({ color: "#9ff2e8", size: 0.6, transparent: true, opacity: 0 }));
scene.add(burst);
let burstVel = [], burstT = 1;
function fireBurst(p, colour) {
  const pos = burstGeo.attributes.position.array;
  burstVel = [];
  for (let i = 0; i < BURST_N; i++) {
    pos.set([p.x, 3, p.z], i * 3);
    const a = Math.random() * Math.PI * 2, u = Math.random();
    burstVel.push([Math.cos(a) * (2 + 6 * u), 6 + 8 * Math.random(), Math.sin(a) * (2 + 6 * u)]);
  }
  burstGeo.attributes.position.needsUpdate = true;
  burst.material.color.set(colour);
  burstT = 0;
}

// --------------------------------------------------------------- helpers --
function distressColour(d) {
  const calm = new THREE.Color("#7fb2ff"), mid = new THREE.Color("#b48cf0"), hot = new THREE.Color("#ec5a6b");
  return d < 0.55 ? calm.lerp(mid, d / 0.55) : mid.lerp(hot, Math.min(1, (d - 0.55) / 0.35));
}
function destination(step, session) {
  const o = step.outcome;
  if (o.startsWith("Connected to a generic")) return { ...GENERIC, z: GENERIC.z + 3.5 };
  if (o.startsWith("Connected")) { const p = SERVICE_POS[session.profile.preference_idx]; return { x: p.x - 4.5, z: p.z }; }
  if (o.startsWith("At-risk") || o.startsWith("Escalated")) return { x: TOWER.x + 1, z: TOWER.z + 4.5 };
  if (o.startsWith("She left")) return EXIT;
  if (o.startsWith("Session closed")) return { x: ST[0].x + 1.5, z: ST[0].z + 1.5 };
  const p = curve.getPoint(Math.min(step.stage, 4) / 4);
  return { x: p.x, z: p.z + 1.2 };
}
const isSuccess = (o) => o.startsWith("Connected") || o.startsWith("At-risk") || o.startsWith("Escalated");
const reachedHuman = (o) => o.startsWith("At-risk") || o.startsWith("Escalated");

// ------------------------------------------------------------------ replay --
let replay = null, session = null, sIdx = 0, step = 0, playing = false, speed = 1, stepT = 0, autoTimer = null;
const STEP_SECONDS = 1.9;
let sFrom = { x: ST[0].x, z: ST[0].z }, sTo = { ...sFrom }, flash = 0;

function applyStep(k, instant = false) {
  if (!session) return;
  step = k;
  const s = session.steps[k], prof = session.profile, last = k === session.steps.length - 1;
  const target = destination(s, session);
  sFrom = instant ? target : { x: survivor.position.x, z: survivor.position.z };
  sTo = target;
  survivor.userData.walk = instant ? 1 : 0;
  if (instant) survivor.position.set(target.x, 0, target.z);

  const col = distressColour(s.distress);
  sMat.color.copy(col); sMat.emissive.copy(col);
  halo.material.opacity = 0.15 + 0.8 * s.trust;
  shield.userData.on = s.privacy_on;
  riskRing.material.opacity = s.risk_flag ? 0.9 : 0;
  danger.userData.on = prof.risk && !s.risk_flag && !reachedHuman(s.outcome);
  phone.visible = prof.shared_device;

  const prev = session.steps[Math.max(0, k - 1)];
  const lit = reachedHuman(s.outcome) ? prev.stage : s.stage;
  stations.forEach((st, i) => {
    const on = lit >= i - 1e-6;
    st.plat.material.emissive.set(on ? "#3b2a66" : "#000000");
    st.glow.material.opacity = on ? 0.6 : 0.12;
  });
  arch.material.emissive.set("#7fb2ff").multiplyScalar(Math.min(1, s.trust / 0.5) * 0.8);
  lanterns.forEach((l, i) => { const on = i < s.info; l.material.color.set(on ? "#fff1c9" : "#3a3550"); l.material.emissive.set(on ? "#ffd27a" : "#000000"); });
  gate.userData.open = s.consent;
  gateBar.material.emissive.set(s.consent ? "#fad95a" : "#000000").multiplyScalar(0.5);
  services.forEach((sv, i) => {
    const pref = s.pref_known && i === prof.preference_idx;
    const arrived = s.outcome.startsWith("Connected to " + session_services()[i]);
    sv.beacon.material.emissiveIntensity = pref || arrived ? 1.2 : 0;
    sv.beacon.material.opacity = pref || arrived ? 0.95 : 0.25;
    sv.light.intensity = arrived ? 2.2 : pref ? 0.9 : 0;
  });
  towerLamp.material.emissive.set(s.risk_flag || reachedHuman(s.outcome) ? "#ff4d5e" : "#ffffff");
  towerLamp.material.emissiveIntensity = reachedHuman(s.outcome) ? 1.4 : s.risk_flag ? 0.9 : 0.2;
  towerLight.intensity = reachedHuman(s.outcome) ? 2.5 : s.risk_flag ? 1.2 : 0;

  const name = s.action === null ? null : replay.actions[s.action];
  if (name && !instant) {
    const toast = document.getElementById("actionToast");
    toast.textContent = ACTION_TEXT[name];
    toast.style.borderLeftColor = ACTION_COLOURS[s.action];
    toast.style.opacity = 1;
    setTimeout(() => (toast.style.opacity = 0), 1100 / speed);
  }
  setBubble(name ? ACTION_SAYS[name] : "Hi. This is a private space. Take your time.", name ? ACTION_COLOURS[s.action] : "#d9ccff");
  guide.material.color.set(name ? ACTION_COLOURS[s.action] : "#fad95a");
  guide.material.emissive.copy(guide.material.color);
  guideLight.color.copy(guide.material.color);
  if (s.outcome.startsWith("PRIVACY") && !instant) flash = 1;

  document.getElementById("bannerStep").textContent = `Step ${s.t} of ${replay.max_steps}`;
  document.getElementById("bannerRisk").style.display = s.risk_flag ? "inline-block" : "none";
  document.getElementById("bannerPrivacy").style.display = s.privacy_on ? "inline-block" : "none";
  showOutcome(last ? s : null, instant);
  updatePanel(k);
  document.getElementById("scrub").value = k;
}
function session_services() { return replay.services; }

function showOutcome(s, instant) {
  const card = document.getElementById("outcomeCard");
  if (!s) { card.classList.remove("show"); return; }
  const prof = session.profile, title = document.getElementById("outcomeTitle"), sub = document.getElementById("outcomeSub");
  title.textContent = s.outcome;
  let note = `Session return ${s.total.toFixed(2)} after ${s.t} steps.`;
  if (prof.risk && !reachedHuman(s.outcome)) note += " She was in immediate danger and never reached a human counsellor.";
  else if (prof.risk) note += " Her danger was caught and she reached a human.";
  if (s.outcome.startsWith("Connected") && !s.outcome.includes("generic")) note += " She got the support she asked for.";
  sub.textContent = note;
  title.style.color = prof.risk && !reachedHuman(s.outcome) ? "#ff8a98" : isSuccess(s.outcome) ? "#9ff2e8" : "#ff8a98";
  card.classList.add("show");
  if (!instant && isSuccess(s.outcome)) {
    const d = destination(s, session);
    fireBurst(d, prof.risk && !reachedHuman(s.outcome) ? "#ff8a98" : "#9ff2e8");
  }
  if (!instant && playing === false && document.getElementById("autoNext").checked && sIdx < replay.sessions.length - 1) {
    clearTimeout(autoTimer);
    autoTimer = setTimeout(() => { selectSession(sIdx + 1); play(); }, 4200 / Math.max(1, speed / 1.5));
  }
}

// ------------------------------------------------------------------- panel --
const charts = {};
function makeCharts() {
  Chart.defaults.color = "#a39cbb";
  Chart.defaults.font.size = 10;
  const opts = (title, y = {}) => ({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { title: { display: true, text: title, color: "#f0eef7", font: { size: 11, weight: "600" } },
      legend: { labels: { boxWidth: 10, boxHeight: 2 } } },
    scales: { x: { min: 0, max: 10, type: "linear", grid: { color: "#2c2640" } }, y: { grid: { color: "#2c2640" }, ...y } },
  });
  const line = (label, colour, dash) => ({ label, data: [], borderColor: colour, pointRadius: 0, borderWidth: 2, borderDash: dash || [] });
  charts.state = new Chart(document.getElementById("chartState"), { type: "line",
    data: { datasets: [line("distress", "#ec5a6b"), line("trust", "#2fb7ad"), line("readiness", "#8c5cd9")] },
    options: opts("How she is feeling (true state)", { min: 0, max: 1 }) });
  const r = opts("Cumulative reward");
  r.plugins.legend = { display: false };
  charts.ret = new Chart(document.getElementById("chartReturn"), { type: "line", data: { datasets: [line("return", "#fad95a")] }, options: r });
  charts.risk = new Chart(document.getElementById("chartRisk"), { type: "line",
    data: { datasets: [line("pressure", "#ff7a45"), line("phone exposure", "#ec5a6b", [4, 3])] },
    options: opts("Pressure and privacy exposure", { min: 0, max: 1 }) });
}

function chip(text, cls = "") { return `<span class="chip ${cls}">${text}</span>`; }

function updatePanel(k) {
  const steps = session.steps.slice(0, k + 1), s = session.steps[k];
  const xmax = Math.max(10, 2 * Math.ceil((session.steps.length - 1) / 2));
  Object.values(charts).forEach((c) => (c.options.scales.x.max = xmax));
  charts.state.data.datasets[0].data = steps.map((x) => ({ x: x.t, y: x.distress }));
  charts.state.data.datasets[1].data = steps.map((x) => ({ x: x.t, y: x.trust }));
  charts.state.data.datasets[2].data = steps.map((x) => ({ x: x.t, y: x.readiness }));
  charts.ret.data.datasets[0].data = steps.map((x) => ({ x: x.t, y: x.total }));
  charts.risk.data.datasets[0].data = steps.map((x) => ({ x: x.t, y: x.pressure }));
  charts.risk.data.datasets[1].data = steps.map((x) => ({ x: x.t, y: x.exposure }));
  Object.values(charts).forEach((c) => c.update());

  const name = s.action === null ? null : replay.actions[s.action];
  const an = document.getElementById("actionName");
  an.textContent = name ? ACTION_TEXT[name] : "Session opens";
  an.style.color = name ? ACTION_COLOURS[s.action] : "#f0eef7";
  document.getElementById("eventText").textContent = s.event + (name ? `   (reward ${s.reward >= 0 ? "+" : ""}${s.reward.toFixed(2)})` : "");
  document.getElementById("kReturn").textContent = s.total.toFixed(2);
  document.getElementById("kStep").textContent = s.t;
  document.getElementById("kInfo").textContent = `${s.info}/4`;
  document.getElementById("flags").innerHTML = [
    s.privacy_on ? chip("discreet mode on", "ok") : chip("discreet mode off", session.profile.shared_device ? "warn" : ""),
    s.safety_checked ? (s.risk_flag ? chip("safety screen: danger", "bad") : chip("safety screen: safe", "ok")) : chip("not screened yet"),
    s.consent ? chip("consent given", "ok") : chip("no consent yet"),
    s.pref_known ? chip("wants " + session.profile.preference, "ok") : chip("preference unknown"),
  ].join("");
  const cells = document.getElementById("timeline").children;
  for (let i = 0; i < cells.length; i++) {
    const a = i + 1 <= k && i + 1 < session.steps.length ? session.steps[i + 1].action : null;
    cells[i].style.background = a === null ? "" : ACTION_COLOURS[a];
  }
}

function buildStatic() {
  const tl = document.getElementById("timeline");
  tl.innerHTML = "";
  for (let i = 0; i < replay.max_steps; i++) tl.appendChild(document.createElement("div"));
  document.getElementById("actionKey").innerHTML = replay.actions.map((a, i) =>
    `<span><i style="background:${ACTION_COLOURS[i]}"></i>${a.toLowerCase().replaceAll("_", " ")}</span>`).join("");
  document.getElementById("agentNote").textContent = replay.note || "";
  document.getElementById("sessionTabs").innerHTML = replay.sessions.map((s, i) =>
    `<button data-i="${i}"><b>${i + 1}. ${s.title}</b><span>${s.blurb}</span></button>`).join("");
  document.querySelectorAll("#sessionTabs button").forEach((b) => (b.onclick = () => { selectSession(+b.dataset.i); play(); }));
}

function selectSession(i) {
  clearTimeout(autoTimer);
  sIdx = i;
  session = replay.sessions[i];
  document.querySelectorAll("#sessionTabs button").forEach((b, j) => b.classList.toggle("on", j === i));
  const p = session.profile;
  document.getElementById("profileChips").innerHTML = [
    p.risk ? chip("in immediate danger", "bad") : chip("no immediate danger", "ok"),
    p.shared_device ? chip("shared phone", "warn") : chip("private phone", "ok"),
    chip("wants " + p.preference),
  ].join("");
  document.getElementById("scrub").max = session.steps.length - 1;
  pause();
  watcher.position.set(ST[0].x + 14, 0, ST[0].z + 12);
  applyStep(0, true);
}

async function loadReplay(file, keepSession = true) {
  replay = await (await fetch(`replays/${file}`)).json();
  buildStatic();
  selectSession(keepSession ? Math.min(sIdx, replay.sessions.length - 1) : 0);
}

function play() {
  if (!session) return;
  if (step >= session.steps.length - 1) applyStep(0, true);
  playing = true; stepT = 0;
  document.getElementById("playBtn").textContent = "❚❚";
}
function pause() { playing = false; document.getElementById("playBtn").textContent = "▶"; }

async function init() {
  makeCharts();
  const index = await (await fetch("replays/index.json")).json();
  const sel = document.getElementById("agentSelect");
  sel.innerHTML = index.map((r) => `<option value="${r.file}">${r.label}</option>`).join("");
  const q = new URLSearchParams(location.search);
  if (q.get("agent") && index.some((r) => r.file === `${q.get("agent")}.json`)) sel.value = `${q.get("agent")}.json`;
  if (q.get("session")) sIdx = Math.max(0, +q.get("session") - 1);
  sel.onchange = () => loadReplay(sel.value);
  await loadReplay(sel.value);
}

document.getElementById("playBtn").onclick = () => (playing ? pause() : play());
document.getElementById("scrub").oninput = (e) => { pause(); clearTimeout(autoTimer); applyStep(+e.target.value, true); };
document.getElementById("speed").onchange = (e) => (speed = +e.target.value);
document.getElementById("startBtn").onclick = () => {
  document.getElementById("intro").classList.add("hide");
  setTimeout(play, 700);
};
if (new URLSearchParams(location.search).has("nointro")) document.getElementById("intro").classList.add("hide");

// -------------------------------------------------------------------- loop --
const clock = new THREE.Clock();
function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight, pr = renderer.getPixelRatio();
  if (canvas.width !== Math.floor(w * pr) || canvas.height !== Math.floor(h * pr)) {
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2);
const lerp = (a, b, t) => a + (b - a) * t;

function loop() {
  const dt = Math.min(clock.getDelta(), 0.1), time = clock.elapsedTime;
  resize();
  if (session && playing) {
    stepT += dt * speed / STEP_SECONDS;
    if (stepT >= 1) {
      stepT = 0;
      if (step < session.steps.length - 1) {
        const next = step + 1;
        if (next === session.steps.length - 1) pause();
        applyStep(next);
      } else pause();
    }
  }
  if (session) {
    const s = session.steps[step], prof = session.profile;
    // survivor walks toward this step's target; long walks (to a service or the tower) take longer
    const far = Math.hypot(sTo.x - sFrom.x, sTo.z - sFrom.z) > 12;
    survivor.userData.walk = Math.min(1, survivor.userData.walk + dt * speed / (far ? 2.2 : 1.1));
    const k = ease(survivor.userData.walk);
    const x = lerp(sFrom.x, sTo.x, k), z = lerp(sFrom.z, sTo.z, k);
    const moving = k < 0.999 && Math.hypot(sTo.x - sFrom.x, sTo.z - sFrom.z) > 0.4;
    survivor.position.set(x, moving ? Math.abs(Math.sin(time * 10)) * 0.3 : 0, z);
    const leaving = s.outcome.startsWith("She left");
    sMat.opacity = leaving ? Math.max(0.15, 1 - k * 0.85) : 1;
    shield.material.opacity = lerp(shield.material.opacity, shield.userData.on ? 0.22 : 0, 0.1);
    shield.scale.setScalar(1 + 0.03 * Math.sin(time * 3));
    danger.material.opacity = danger.userData.on ? 0.45 + 0.2 * Math.sin(time * 5) : 0;
    if (riskRing.material.opacity > 0) riskRing.scale.setScalar(1 + 0.12 * Math.sin(time * 7));

    // guide floats beside her, jittering when it pushes (pressure)
    const jitter = s.pressure * 0.35;
    guide.position.set(x + 2.2 + Math.sin(time * 30) * jitter, 3.4 + 0.3 * Math.sin(time * 2.4), z - 1 + Math.cos(time * 27) * jitter);
    if (bubble) bubble.position.set(guide.position.x + 0.5, guide.position.y + 2.3, guide.position.z);

    // the watcher closes in as exposure grows on a shared phone, and turns away once discreet mode is on
    if (prof.shared_device && !isSuccess(s.outcome)) {
      const breach = s.outcome.startsWith("PRIVACY");
      const dist = breach ? 1.6 : lerp(15, 3, s.exposure);
      const tx = x + dist * 0.75, tz = z + dist * 0.66;
      watcher.position.x = lerp(watcher.position.x, tx, 0.04);
      watcher.position.z = lerp(watcher.position.z, tz, 0.04);
      watcher.lookAt(x, 0, z);
      const op = s.privacy_on ? 0.35 : 0.95;
      wMat.opacity = lerp(wMat.opacity, op, 0.05);
      wEyes.material.opacity = s.privacy_on ? 0 : wMat.opacity;
    } else {
      wMat.opacity = lerp(wMat.opacity, 0, 0.05);
      wEyes.material.opacity = 0;
    }

    // camera drifts to keep her in view
    controls.target.x = lerp(controls.target.x, 9 + (x - 9) * 0.15, 0.015);
    controls.target.z = lerp(controls.target.z, z * 0.15, 0.015);
    const sky = flash > 0 ? ALARM : s.risk_flag ? ALARM.clone().lerp(NIGHT, 0.6) : s.privacy_on ? CALM : NIGHT;
    scene.background.lerp(sky, flash > 0 ? 0.3 : 0.03);
    scene.fog.color.copy(scene.background);
    flash = Math.max(0, flash - dt * 0.8);
    gate.rotation.z = lerp(gate.rotation.z, gate.userData.open ? Math.PI * 0.42 : 0, 0.06);
  }
  if (burstT < 1) {
    burstT += dt * 0.6;
    const pos = burstGeo.attributes.position.array;
    for (let i = 0; i < BURST_N; i++) {
      burstVel[i][1] -= 14 * dt;
      pos[i * 3] += burstVel[i][0] * dt; pos[i * 3 + 1] = Math.max(0.2, pos[i * 3 + 1] + burstVel[i][1] * dt); pos[i * 3 + 2] += burstVel[i][2] * dt;
    }
    burstGeo.attributes.position.needsUpdate = true;
    burst.material.opacity = 1 - burstT;
  } else burst.material.opacity = 0;
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

init().catch((err) => {
  document.getElementById("eventText").textContent = "Could not load replays: run `uv run main.py --web`. " + err;
});
loop();
