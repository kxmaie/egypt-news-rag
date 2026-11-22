/* Egypt News Intelligence — Interactive 3D Holographic Map of Egypt + GSAP Animations */

// Map coordinate projection: lon (24.8..36.8) -> X (-95..95), lat (22.0..31.7) -> Y (-85..85)
function lonLatToMapXY(lon, lat) {
  const x = (lon - 30.8) * 15.5;
  const y = (lat - 26.85) * 17.2;
  return { x, y };
}

// Well-spaced visual coordinates (lon, lat) inside Egypt's map outline so all 27 governorates
// sit in their true geographic zones while Delta neighbors remain crystal-clear & non-overlapping
const EGYPT_MAP_GOVS = {
  "القاهرة":       { lon: 31.35, lat: 30.02 },
  "الجيزة":        { lon: 30.65, lat: 29.85 },
  "القليوبية":     { lon: 31.38, lat: 30.42 },
  "المنوفية":      { lon: 30.78, lat: 30.52 },
  "الغربية":       { lon: 30.98, lat: 30.88 },
  "الدقهلية":      { lon: 31.62, lat: 31.05 },
  "كفر الشيخ":     { lon: 30.82, lat: 31.42 },
  "دمياط":         { lon: 31.88, lat: 31.48 },
  "البحيرة":       { lon: 30.18, lat: 30.78 },
  "الإسكندرية":    { lon: 29.68, lat: 31.18 },
  "مطروح":         { lon: 27.15, lat: 31.15 },
  "الشرقية":       { lon: 31.82, lat: 30.62 },
  "الإسماعيلية":   { lon: 32.38, lat: 30.55 },
  "بورسعيد":       { lon: 32.38, lat: 31.28 },
  "السويس":        { lon: 32.52, lat: 29.92 },
  "شمال سيناء":    { lon: 33.75, lat: 30.85 },
  "جنوب سيناء":    { lon: 33.95, lat: 28.65 },
  "الفيوم":        { lon: 30.42, lat: 29.30 },
  "بني سويف":      { lon: 31.05, lat: 28.95 },
  "المنيا":        { lon: 30.68, lat: 28.10 },
  "أسيوط":         { lon: 31.18, lat: 27.22 },
  "سوهاج":         { lon: 31.70, lat: 26.55 },
  "قنا":           { lon: 32.75, lat: 26.18 },
  "الأقصر":        { lon: 32.62, lat: 25.55 },
  "أسوان":         { lon: 32.90, lat: 24.05 },
  "البحر الأحمر":  { lon: 34.15, lat: 26.55 },
  "الوادي الجديد": { lon: 28.20, lat: 25.35 },
};

// Accurate geographic outline of Egypt (lon, lat)
const EGYPT_OUTLINE_COORDS = [
  [25.0, 31.55], [26.2, 31.45], [27.2, 31.35], [28.5, 31.05], [29.85, 31.20],
  [30.4, 31.45], [31.0, 31.58], [31.85, 31.52], [32.35, 31.26], [33.2, 31.15],
  [34.22, 31.28], [34.88, 29.52], [34.70, 28.70], [34.30, 27.75], [33.72, 28.35],
  [32.55, 29.92], [32.68, 29.20], [33.15, 28.15], [33.82, 27.20], [34.45, 26.10],
  [34.95, 25.00], [35.55, 23.95], [36.65, 22.00], [32.00, 22.00], [28.50, 22.00],
  [25.00, 22.00], [25.00, 25.50], [24.75, 29.20], [25.00, 31.55],
];

// Point-in-polygon check so holographic grid dots fill ONLY Egypt's real territory
function isInsideEgypt(lon, lat) {
  let inside = false;
  for (let i = 0, j = EGYPT_OUTLINE_COORDS.length - 1; i < EGYPT_OUTLINE_COORDS.length; j = i++) {
    const xi = EGYPT_OUTLINE_COORDS[i][0], yi = EGYPT_OUTLINE_COORDS[i][1];
    const xj = EGYPT_OUTLINE_COORDS[j][0], yj = EGYPT_OUTLINE_COORDS[j][1];
    const intersect = yi > lat !== yj > lat && lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-9) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

let scene, camera, renderer, mapGroup;
let govMarkers = [];
let labelElements = [];
let pulseRings = [];
let signalArcs = [];
let nileParticles = [];
let nileMainCurve = null;
let raycaster, mouse;
let isDragging = false;
let dragDistance = 0;
let prevPointer = { x: 0, y: 0 };
let targetTilt = { x: -0.28, y: 0.0 };
let heroVisible = true;
let reducedMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let canvasW = 800;
let canvasH = 560;
let currentGovData = null;
let currentTab = "articles";
let governoratesList = [];

const _worldPos = new THREE.Vector3();
const _projected = new THREE.Vector3();

function initEgyptMap() {
  const canvas = document.getElementById("globe-canvas");
  const wrap = document.getElementById("globe-canvas-wrap");
  canvasW = wrap.clientWidth;
  canvasH = wrap.clientHeight;

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(38, canvasW / canvasH, 1, 1200);
  camera.position.set(0, -18, 255);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.setSize(canvasW, canvasH);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  mapGroup = new THREE.Group();
  // Slight 3D holographic perspective tilt
  mapGroup.rotation.x = targetTilt.x;
  mapGroup.rotation.y = targetTilt.y;
  scene.add(mapGroup);

  // 1. Subtle Holographic Radar Base Rings behind Egypt
  [65, 95, 125].forEach((r, idx) => {
    const ringGeo = new THREE.RingGeometry(r - 0.35, r, 64);
    const ringMat = new THREE.MeshBasicMaterial({
      color: 0x115e59,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.16 - idx * 0.03,
    });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.position.z = -3;
    mapGroup.add(ring);
  });

  // 2. Glowing 3D Egypt Border Outline (Double Layer for 3D Depth)
  const borderPtsTop = [];
  const borderPtsBottom = [];
  EGYPT_OUTLINE_COORDS.forEach(([lon, lat]) => {
    const { x, y } = lonLatToMapXY(lon, lat);
    borderPtsTop.push(new THREE.Vector3(x, y, 1.5));
    borderPtsBottom.push(new THREE.Vector3(x, y, -2.0));
  });

  const topOutlineGeo = new THREE.BufferGeometry().setFromPoints(borderPtsTop);
  mapGroup.add(
    new THREE.Line(
      topOutlineGeo,
      new THREE.LineBasicMaterial({ color: 0x5eead4, transparent: true, opacity: 0.92 })
    )
  );

  const botOutlineGeo = new THREE.BufferGeometry().setFromPoints(borderPtsBottom);
  mapGroup.add(
    new THREE.Line(
      botOutlineGeo,
      new THREE.LineBasicMaterial({ color: 0x0f766e, transparent: true, opacity: 0.45 })
    )
  );

  // 3. Holographic Dotted Interior Matrix strictly inside Egypt's borders
  const interiorDots = [];
  for (let lon = 24.9; lon <= 36.6; lon += 0.23) {
    for (let lat = 22.1; lat <= 31.6; lat += 0.21) {
      if (isInsideEgypt(lon, lat)) {
        const { x, y } = lonLatToMapXY(lon, lat);
        interiorDots.push(x, y, 0);
      }
    }
  }
  const dotsGeo = new THREE.BufferGeometry();
  dotsGeo.setAttribute("position", new THREE.Float32BufferAttribute(interiorDots, 3));
  mapGroup.add(
    new THREE.Points(
      dotsGeo,
      new THREE.PointsMaterial({ color: 0x2dd4bf, size: 1.95, transparent: true, opacity: 0.62 })
    )
  );

  // 4. Glowing Nile River + Delta Branches (Rosetta & Damietta) + Animated Flowing Particles
  const nileCoords = [
    [31.5, 22.2], [32.6, 23.1], [32.9, 24.08], [32.64, 25.68], [32.71, 26.15],
    [31.70, 26.55], [31.18, 27.18], [30.76, 28.08], [31.09, 29.06], [31.20, 29.85],
    [31.24, 30.05],
  ];
  const nileVecs = nileCoords.map(([lon, lat]) => {
    const { x, y } = lonLatToMapXY(lon, lat);
    return new THREE.Vector3(x, y, 1.8);
  });
  nileMainCurve = new THREE.CatmullRomCurve3(nileVecs);
  const nileGeo = new THREE.BufferGeometry().setFromPoints(nileMainCurve.getPoints(70));
  mapGroup.add(
    new THREE.Line(nileGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.95 }))
  );

  // Rosetta Branch (فرع رشيد) & Damietta Branch (فرع دمياط)
  const rosettaPts = [[31.24, 30.05], [30.85, 30.65], [30.45, 31.42]].map(([lon, lat]) => {
    const { x, y } = lonLatToMapXY(lon, lat);
    return new THREE.Vector3(x, y, 1.8);
  });
  const damiettaPts = [[31.24, 30.05], [31.38, 30.70], [31.82, 31.48]].map(([lon, lat]) => {
    const { x, y } = lonLatToMapXY(lon, lat);
    return new THREE.Vector3(x, y, 1.8);
  });
  [rosettaPts, damiettaPts].forEach((branch) => {
    const c = new THREE.CatmullRomCurve3(branch);
    const g = new THREE.BufferGeometry().setFromPoints(c.getPoints(24));
    mapGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 })));
  });

  // Animated flowing light particles along the Nile River
  const nileDotGeo = new THREE.SphereGeometry(0.95, 8, 8);
  const nileDotMat = new THREE.MeshBasicMaterial({ color: 0xbae6fd });
  for (let i = 0; i < 6; i++) {
    const pMesh = new THREE.Mesh(nileDotGeo, nileDotMat);
    mapGroup.add(pMesh);
    nileParticles.push({ mesh: pMesh, t: i / 6 });
  }

  // Pause rendering when scrolled off-screen
  const heroSection = document.getElementById("globe-command");
  const observer = new IntersectionObserver(
    (entries) => {
      heroVisible = entries[0].isIntersecting;
    },
    { threshold: 0.05 }
  );
  observer.observe(heroSection);

  // Interactive Tilt / Pan / Parallax
  wrap.addEventListener(
    "pointerdown",
    (e) => {
      isDragging = true;
      dragDistance = 0;
      prevPointer.x = e.clientX;
      prevPointer.y = e.clientY;
    },
    { passive: true }
  );
  window.addEventListener("pointerup", () => {
    isDragging = false;
  });
  wrap.addEventListener(
    "pointermove",
    (e) => {
      const rect = wrap.getBoundingClientRect();
      const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      const ny = -((e.clientY - rect.top) / rect.height) * 2 + 1;

      if (isDragging) {
        const dx = e.clientX - prevPointer.x;
        const dy = e.clientY - prevPointer.y;
        dragDistance += Math.abs(dx) + Math.abs(dy);
        targetTilt.y = Math.max(-0.55, Math.min(0.55, targetTilt.y + dx * 0.0045));
        targetTilt.x = Math.max(-0.65, Math.min(0.25, targetTilt.x + dy * 0.0045));
        prevPointer.x = e.clientX;
        prevPointer.y = e.clientY;
      } else {
        // Subtle holographic mouse parallax
        targetTilt.y = nx * 0.14;
        targetTilt.x = -0.26 - ny * 0.1;
      }
    },
    { passive: true }
  );
  wrap.addEventListener("click", (e) => {
    if (dragDistance < 8) handleMapClick(e);
  });

  window.addEventListener("resize", () => {
    canvasW = wrap.clientWidth;
    canvasH = wrap.clientHeight;
    camera.aspect = canvasW / canvasH;
    camera.updateProjectionMatrix();
    renderer.setSize(canvasW, canvasH);
  });

  animate();
}

function addGovernorateMarkers(govs) {
  const labelsContainer = document.getElementById("globe-labels");
  labelsContainer.innerHTML = "";
  govMarkers = [];
  labelElements = [];
  pulseRings = [];
  signalArcs = [];

  const cairoCoord = EGYPT_MAP_GOVS["القاهرة"];
  const cairoXY = lonLatToMapXY(cairoCoord.lon, cairoCoord.lat);
  const cairoVec = new THREE.Vector3(cairoXY.x, cairoXY.y, 2.5);

  const sharedPinGeo = new THREE.SphereGeometry(1, 12, 12);
  const sharedDotGeo = new THREE.SphereGeometry(0.95, 8, 8);
  const sharedDotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

  govs.forEach((g, idx) => {
    const mc = EGYPT_MAP_GOVS[g.name] || { lon: g.lon, lat: g.lat };
    const { x, y } = lonLatToMapXY(mc.lon, mc.lat);
    g.mapX = x;
    g.mapY = y;

    const pos = new THREE.Vector3(x, y, 2.5);
    const hasActivity = g.articles_count > 0 || g.events_count > 0 || g.issues_count > 0;
    const size = hasActivity ? Math.min(2.8, 1.35 + Math.log1p(g.articles_count) * 0.28) : 1.15;
    const color = g.issues_count > 0 ? 0xfb7185 : hasActivity ? 0x5eead4 : 0x38bdf8;

    const mesh = new THREE.Mesh(sharedPinGeo, new THREE.MeshBasicMaterial({ color }));
    mesh.scale.setScalar(size);
    mesh.position.copy(pos);
    mesh.userData = g;
    mapGroup.add(mesh);
    govMarkers.push(mesh);

    // Expanding radar ring on the map plane
    if (hasActivity) {
      const ringGeo = new THREE.RingGeometry(size * 1.25, size * 1.7, 24);
      const ringMat = new THREE.MeshBasicMaterial({
        color,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.7,
      });
      const ring = new THREE.Mesh(ringGeo, ringMat);
      ring.position.set(x, y, 2.2);
      mapGroup.add(ring);
      pulseRings.push({ ring, phase: idx * 0.6 });
    }

    // 3D Curved Signal Arcs arching above Egypt from Cairo to active governorates
    if (g.name !== "القاهرة" && hasActivity && signalArcs.length < 14) {
      const dist = pos.distanceTo(cairoVec);
      const mid = new THREE.Vector3((cairoVec.x + x) * 0.5, (cairoVec.y + y) * 0.5, 14 + dist * 0.22);
      const curve = new THREE.QuadraticBezierCurve3(cairoVec, mid, pos);
      const arcGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(28));
      const arcMat = new THREE.LineBasicMaterial({
        color: g.issues_count > 0 ? 0xfb7185 : 0x2dd4bf,
        transparent: true,
        opacity: 0.42,
      });
      mapGroup.add(new THREE.Line(arcGeo, arcMat));

      const dot = new THREE.Mesh(sharedDotGeo, sharedDotMat);
      mapGroup.add(dot);
      signalArcs.push({ curve, dot, t: (idx * 0.14) % 1 });
    }

    // Floating Arabic Governorate Label directly over its pin on the Egypt Map
    const lbl = document.createElement("div");
    lbl.className = "globe-gov-label" + (g.issues_count > 0 ? " has-issue" : "");
    lbl.dataset.gov = g.name;
    lbl.setAttribute("role", "button");
    lbl.setAttribute("tabindex", "0");
    lbl.setAttribute("aria-label", `محافظة ${g.name} (${g.articles_count} مقال، ${g.events_count} حدث، ${g.issues_count} مشكلة)`);
    lbl.innerHTML = `<span class="lbl-dot"></span><span>${g.name}</span>${g.articles_count > 0 ? `<span class="lbl-cnt">${g.articles_count}</span>` : ""}`;
    lbl.addEventListener("click", (ev) => {
      ev.stopPropagation();
      selectGovernorate(g.name);
    });
    lbl.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        selectGovernorate(g.name);
      }
    });
    labelsContainer.appendChild(lbl);
    labelElements.push({ el: lbl, localPos: pos.clone() });
  });
}

function updateFloatingLabels() {
  const len = labelElements.length;
  if (!len) return;
  const matWorld = mapGroup.matrixWorld;

  for (let i = 0; i < len; i++) {
    const item = labelElements[i];
    _worldPos.copy(item.localPos).applyMatrix4(matWorld);
    _projected.copy(_worldPos).project(camera);
    const x = (_projected.x * 0.5 + 0.5) * canvasW;
    const y = (-(_projected.y * 0.5) + 0.5) * canvasH;
    item.el.style.transform = `translate3d(${x.toFixed(0)}px, ${(y - 8).toFixed(0)}px, 0) translate(-50%, -100%)`;
  }
}

function handleMapClick(e) {
  const wrap = document.getElementById("globe-canvas-wrap");
  const rect = wrap.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(govMarkers);
  if (hits.length > 0) {
    selectGovernorate(hits[0].object.userData.name);
  }
}

function animate() {
  requestAnimationFrame(animate);
  if (!heroVisible) return;

  if (!reducedMotion) {
    mapGroup.rotation.x += (targetTilt.x - mapGroup.rotation.x) * 0.08;
    mapGroup.rotation.y += (targetTilt.y - mapGroup.rotation.y) * 0.08;

    const tNow = performance.now() * 0.0035;
    for (let i = 0; i < pulseRings.length; i++) {
      const p = pulseRings[i];
      const s = 1 + 0.45 * Math.sin(tNow + p.phase);
      p.ring.scale.set(s, s, 1);
      p.ring.material.opacity = 0.72 - 0.35 * Math.sin(tNow + p.phase);
    }

    for (let i = 0; i < signalArcs.length; i++) {
      const a = signalArcs[i];
      a.t = (a.t + 0.0065) % 1;
      a.curve.getPointAt(a.t, a.dot.position);
    }

    if (nileMainCurve) {
      for (let i = 0; i < nileParticles.length; i++) {
        const np = nileParticles[i];
        np.t = (np.t + 0.0035) % 1;
        nileMainCurve.getPointAt(np.t, np.mesh.position);
      }
    }
  }

  renderer.render(scene, camera);
  updateFloatingLabels();
}

function renderStateBox(containerEl, type, title, desc, retryFn = null) {
  if (!containerEl) return;
  containerEl.innerHTML = `
    <div class="state-box ${type === "error" ? "error-box" : ""}" role="${type === "error" ? "alert" : "status"}">
      <h4>${title}</h4>
      <p>${desc}</p>
      ${retryFn ? `<button class="btn-primary state-retry-btn">إعادة المحاولة ↻</button>` : ""}
    </div>
  `;
  if (retryFn) {
    const btn = containerEl.querySelector(".state-retry-btn");
    if (btn) btn.addEventListener("click", retryFn);
  }
}

// --- API & UI Rendering ---
async function loadOverview() {
  const res = await fetch("/api/overview");
  const data = await res.json();
  const c = data.counts || {};
  animateNumber("kpi-articles", c.total_articles || 0);
  animateNumber("kpi-alerts", c.total_alerts || 0);
  animateNumber("kpi-events", c.total_events || 0);
  animateNumber("kpi-issues", c.total_issues || 0);
  const clEl = document.getElementById("kpi-clusters");
  if (clEl) clEl.textContent = (c.multi_source_clusters || 0).toLocaleString("ar-EG");
  renderDailyDigest(data.latest_digest);
}

function renderDailyDigest(digestRow) {
  if (!digestRow || !digestRow.digest_payload) return;
  const p = digestRow.digest_payload;
  const meta = document.getElementById("digest-meta");
  if (meta) {
    meta.textContent = `تم تقييم ${digestRow.articles_evaluated || 0} مقال خلال آخر ${digestRow.window_hours || 24} ساعة`;
  }

  const catsEl = document.getElementById("digest-categories");
  if (catsEl && p.category_breakdown) {
    catsEl.innerHTML = Object.entries(p.category_breakdown)
      .sort((a, b) => b[1] - a[1])
      .map(([cat, cnt]) => `<div class="digest-cat-pill"><span>${cat}</span><strong>${cnt}</strong></div>`)
      .join("");
  }

  const renderList = (items, isPos) =>
    (items || [])
      .map(
        (it) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge ${isPos ? "badge-pos" : "badge-neg"}">${isPos ? "إيجابي" : "سلبي"}</span>
          <span class="badge">${it.category || ""}</span>
          <span class="badge">${it.source_name || ""}</span>
        </div>
        <h3>${it.title}</h3>
        <p class="alert-rationale">${it.summary || it.rationale || ""}</p>
        <div class="card-foot">
          <span>${it.rationale || ""}</span>
          <a href="${it.canonical_url}" target="_blank" rel="noopener">المصدر ↗</a>
        </div>
      </article>`
      )
      .join("");

  const posEl = document.getElementById("digest-pos-list");
  const negEl = document.getElementById("digest-neg-list");
  if (posEl) posEl.innerHTML = renderList(p.top_positive_for_egypt, true);
  if (negEl) negEl.innerHTML = renderList(p.top_negative_for_egypt, false);
}

function animateNumber(id, target) {
  const el = document.getElementById(id);
  if (!el) return;
  const obj = { val: 0 };
  gsap.to(obj, {
    val: target,
    duration: 1.1,
    ease: "power2.out",
    onUpdate: () => {
      el.textContent = Math.round(obj.val).toLocaleString("ar-EG");
    },
  });
}

function renderGovPills(filterText = "") {
  const pills = document.getElementById("gov-pills");
  const q = filterText.trim();
  const filtered = q ? governoratesList.filter((g) => g.name.includes(q)) : governoratesList;
  pills.innerHTML = filtered
    .map(
      (g) => `
      <button class="gov-pill" data-gov="${g.name}">
        <span>${g.name}</span>
        <span class="cnt">${g.articles_count}</span>
      </button>`
    )
    .join("");

  pills.querySelectorAll(".gov-pill").forEach((btn) => {
    btn.addEventListener("click", () => selectGovernorate(btn.dataset.gov));
  });
}

async function loadGovernorates() {
  try {
    const res = await fetch("/api/governorates");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    governoratesList = data.governorates || [];
    addGovernorateMarkers(governoratesList);
    renderGovPills();

    const sel = document.getElementById("gov-mobile-select");
    if (sel) {
      sel.innerHTML =
        `<option value="">-- اختر من القائمة (${governoratesList.length}) --</option>` +
        governoratesList
          .map((g) => `<option value="${g.name}">${g.name} (${g.articles_count} مقال)</option>`)
          .join("");
    }
    const ragGovSel = document.getElementById("rag-gov-filter");
    if (ragGovSel) {
      ragGovSel.innerHTML =
        `<option value="">📍 كل محافظات مصر</option>` +
        governoratesList.map((g) => `<option value="${g.name}">📍 ${g.name}</option>`).join("");
    }
  } catch (err) {
    renderStateBox(
      document.getElementById("gov-pills"),
      "error",
      "تعذر تحميل قائمة المحافظات",
      "تأكد من اتصال السيرفر بقاعدة بيانات PostgreSQL.",
      () => loadGovernorates()
    );
  }
}

async function selectGovernorate(govName) {
  const govObj = governoratesList.find((g) => g.name === govName);
  const gx = govObj && govObj.mapX !== undefined ? govObj.mapX : 0;
  const gy = govObj && govObj.mapY !== undefined ? govObj.mapY : 0;

  // Smooth GSAP camera fly-to over the selected governorate on the 3D Egypt Map
  gsap.to(mapGroup.position, {
    x: -gx + (canvasW > 700 ? 45 : 0),
    y: -gy * 0.75,
    duration: 1.05,
    ease: "power3.inOut",
  });
  gsap.to(camera.position, {
    x: 0,
    y: -12,
    z: 165,
    duration: 1.1,
    ease: "power3.inOut",
  });

  document.querySelectorAll(".gov-pill, .globe-gov-label").forEach((b) => {
    b.classList.toggle("active", b.dataset.gov === govName);
  });

  const drawer = document.getElementById("gov-drawer");
  drawer.classList.remove("hidden");
  gsap.fromTo(drawer, { x: -24, opacity: 0 }, { x: 0, opacity: 1, duration: 0.3, ease: "power2.out" });

  document.getElementById("drawer-gov-name").textContent = govName;
  document.getElementById("drawer-body").innerHTML = `<div class="loading-msg">جاري تحميل أخبار وأحداث ${govName}...</div>`;

  const res = await fetch(`/api/governorates/${encodeURIComponent(govName)}`);
  currentGovData = await res.json();

  document.getElementById("d-articles-cnt").textContent = currentGovData.articles.length;
  document.getElementById("d-events-cnt").textContent = currentGovData.events.length;
  document.getElementById("d-issues-cnt").textContent = currentGovData.issues.length;
  renderDrawerTab(currentTab);
}

function resetMapView() {
  closeDrawer();
  gsap.to(mapGroup.position, { x: 0, y: 0, duration: 1.0, ease: "power3.inOut" });
  gsap.to(camera.position, { x: 0, y: -18, z: 255, duration: 1.05, ease: "power3.inOut" });
}

function renderPaginatedList(items, gridEl, moreWrapEl, renderCardFn, pageSize = 36) {
  let shown = 0;
  gridEl.innerHTML = "";
  if (moreWrapEl) moreWrapEl.innerHTML = "";

  function showNext(batchSize) {
    const slice = items.slice(shown, shown + batchSize);
    gridEl.insertAdjacentHTML("beforeend", slice.map(renderCardFn).join(""));
    shown += slice.length;
    if (!moreWrapEl) return;
    const rem = items.length - shown;
    if (rem > 0) {
      moreWrapEl.innerHTML = `
        <button class="btn-primary btn-more">عرض المزيد (${Math.min(pageSize, rem)} من أصل ${rem} متبقي)</button>
        <button class="btn-outline btn-all">عرض الكل (${items.length})</button>
      `;
      moreWrapEl.querySelector(".btn-more").addEventListener("click", () => showNext(pageSize));
      moreWrapEl.querySelector(".btn-all").addEventListener("click", () => showNext(rem));
    } else {
      moreWrapEl.innerHTML = "";
    }
  }

  showNext(pageSize);
}

function renderDrawerTab(tab, showAll = false) {
  currentTab = tab;
  document.querySelectorAll(".dtab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  const body = document.getElementById("drawer-body");
  if (!currentGovData) return;

  const limit = showAll ? 5000 : 40;

  if (tab === "articles") {
    const list = currentGovData.articles || [];
    if (!list.length) {
      body.innerHTML = `<div class="loading-msg">لا توجد مقالات مسجلة لهذه المحافظة بعد.</div>`;
      return;
    }
    const rem = list.length - limit;
    body.innerHTML =
      list
        .slice(0, limit)
        .map(
          (a) => `
        <div class="drawer-item">
          <div class="badges-row" style="margin-bottom:6px">
            <span class="badge">${a.category || ""}</span>
            <span class="badge">${a.source_name || ""}</span>
            ${a.impact_label ? `<span class="badge ${a.impact_label.includes("إيجابي") ? "badge-pos" : a.impact_label.includes("سلبي") ? "badge-neg" : ""}">${a.impact_label}</span>` : ""}
          </div>
          <h4>${a.title}</h4>
          <p>${a.summary_generated || ""}</p>
          <a href="${a.canonical_url}" target="_blank" rel="noopener">اقرأ الخبر الأصلي ↗</a>
        </div>`
        )
        .join("") +
      (rem > 0
        ? `<button id="drawer-show-all" class="btn-primary" style="width:100%;margin-top:8px;">عرض كل الأخبار (${list.length})</button>`
        : "");
  } else if (tab === "events") {
    const list = currentGovData.events || [];
    if (!list.length) {
      body.innerHTML = `<div class="loading-msg">لا توجد أحداث مستخرجة لهذه المحافظة بعد.</div>`;
      return;
    }
    const rem = list.length - limit;
    body.innerHTML =
      list
        .slice(0, limit)
        .map(
          (ev) => `
        <div class="drawer-item">
          <div class="badges-row" style="margin-bottom:6px">
            <span class="badge">${ev.event_type}</span>
            <span class="badge badge-amber">${ev.event_status}</span>
            ${ev.city ? `<span class="badge">${ev.city}</span>` : ""}
          </div>
          <h4>${ev.event_title}</h4>
          <p>${ev.event_description}</p>
          <a href="${ev.canonical_url}" target="_blank" rel="noopener">المصدر: ${ev.source_name} ↗</a>
        </div>`
        )
        .join("") +
      (rem > 0
        ? `<button id="drawer-show-all" class="btn-primary" style="width:100%;margin-top:8px;">عرض كل الأحداث (${list.length})</button>`
        : "");
  } else {
    const list = currentGovData.issues || [];
    if (!list.length) {
      body.innerHTML = `<div class="loading-msg">لا توجد مشكلات مرصودة لهذه المحافظة بعد.</div>`;
      return;
    }
    const rem = list.length - limit;
    body.innerHTML =
      list
        .slice(0, limit)
        .map(
          (iss) => `
        <div class="drawer-item">
          <div class="badges-row" style="margin-bottom:6px">
            <span class="badge badge-neg">${iss.severity}</span>
            <span class="badge">${iss.sector || ""}</span>
            <span class="badge">${iss.issue_type}</span>
          </div>
          <h4>${iss.issue_description}</h4>
          ${iss.evidence ? `<p>"${iss.evidence}"</p>` : ""}
          <a href="${iss.canonical_url}" target="_blank" rel="noopener">المصدر: ${iss.source_name} ↗</a>
        </div>`
        )
        .join("") +
      (rem > 0
        ? `<button id="drawer-show-all" class="btn-primary" style="width:100%;margin-top:8px;">عرض كل المشكلات (${list.length})</button>`
        : "");
  }

  const allBtn = document.getElementById("drawer-show-all");
  if (allBtn) {
    allBtn.addEventListener("click", () => renderDrawerTab(tab, true));
  }
}

function closeDrawer() {
  document.getElementById("gov-drawer").classList.add("hidden");
  document.querySelectorAll(".gov-pill, .globe-gov-label").forEach((b) => b.classList.remove("active"));
  gsap.to(mapGroup.position, { x: 0, y: 0, duration: 0.9, ease: "power2.out" });
  gsap.to(camera.position, { x: 0, y: -18, z: 255, duration: 0.95, ease: "power2.out" });
}

async function loadHighImpact(filter = "new") {
  let url = "/api/high-impact?limit=2000";
  if (filter === "new") url += "&only_new_logic=true";
  if (filter === "pos") url += "&label=" + encodeURIComponent("إيجابي محتمل");
  if (filter === "neg") url += "&label=" + encodeURIComponent("سلبي محتمل");

  const grid = document.getElementById("high-impact-grid");
  const moreWrap = document.getElementById("impact-more-wrap");
  grid.innerHTML = `<div class="loading-msg">جاري تحميل الأخبار عالية الأثر...</div>`;
  if (moreWrap) moreWrap.innerHTML = "";

  const res = await fetch(url);
  const data = await res.json();
  if (!data.alerts.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد تنبيهات مطابقة لهذا الفلتر حاليًا.</div>`;
    return;
  }
  renderPaginatedList(
    data.alerts,
    grid,
    moreWrap,
    (al) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge ${al.impact_label.includes("إيجابي") ? "badge-pos" : "badge-neg"}">${al.impact_label}</span>
          ${al.impact_level ? `<span class="badge badge-amber">${al.impact_level} ${al.impact_scope ? "• " + al.impact_scope : ""}</span>` : ""}
          <span class="badge">${al.category}</span>
          ${al.alert_reason ? `<span class="badge">${al.alert_reason}</span>` : ""}
        </div>
        <h3>${al.title}</h3>
        <p class="alert-rationale">${al.rationale || ""}</p>
        ${al.evidence ? `<div class="alert-evidence">"${al.evidence}"</div>` : ""}
        <div class="card-foot">
          <span>${al.source_name}</span>
          <a href="${al.canonical_url}" target="_blank" rel="noopener">قراءة المقال ↗</a>
        </div>
      </article>`
  );
}

async function loadIssues(severity = "") {
  let url = "/api/issues?limit=2000";
  if (severity) url += "&severity=" + encodeURIComponent(severity);
  const grid = document.getElementById("issues-grid");
  const moreWrap = document.getElementById("issues-more-wrap");
  const res = await fetch(url);
  const data = await res.json();
  if (!data.issues || !data.issues.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد مشكلات مطابقة لهذا الفلتر.</div>`;
    if (moreWrap) moreWrap.innerHTML = "";
    return;
  }
  renderPaginatedList(
    data.issues,
    grid,
    moreWrap,
    (iss) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge ${iss.severity === "حرج" || iss.severity === "مرتفع" ? "badge-neg" : "badge-amber"}">${iss.severity}</span>
          ${iss.sector ? `<span class="badge">${iss.sector}</span>` : ""}
          <span class="badge">${iss.issue_type}</span>
          ${iss.governorate_name ? `<span class="badge">📍 ${iss.governorate_name}</span>` : ""}
        </div>
        <h3>${iss.issue_description}</h3>
        <p class="alert-rationale">${iss.title}</p>
        ${iss.evidence ? `<div class="alert-evidence">"${iss.evidence}"</div>` : ""}
        <div class="card-foot">
          <span>${iss.source_name}</span>
          <a href="${iss.canonical_url}" target="_blank" rel="noopener">قراءة الخبر ↗</a>
        </div>
      </article>`
  );
}

async function loadEvents(status = "") {
  let url = "/api/events?limit=2000";
  if (status) url += "&status=" + encodeURIComponent(status);
  const grid = document.getElementById("events-grid");
  const moreWrap = document.getElementById("events-more-wrap");
  const res = await fetch(url);
  const data = await res.json();
  if (!data.events || !data.events.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد أحداث مطابقة لهذه المرحلة.</div>`;
    if (moreWrap) moreWrap.innerHTML = "";
    return;
  }
  renderPaginatedList(
    data.events,
    grid,
    moreWrap,
    (ev) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge badge-amber">${ev.event_status}</span>
          <span class="badge">${ev.event_type}</span>
          <span class="badge">${ev.action}</span>
          ${ev.locations_label ? `<span class="badge">📍 ${ev.locations_label}</span>` : ""}
        </div>
        <h3>${ev.event_title}</h3>
        <p class="alert-rationale">${ev.event_description}</p>
        ${ev.expected_outcome ? `<div class="alert-evidence">النتيجة المتوقعة: ${ev.expected_outcome}</div>` : ""}
        <div class="card-foot">
          <span>${ev.source_name} ${ev.event_date ? "• " + ev.event_date : ""}</span>
          <a href="${ev.canonical_url}" target="_blank" rel="noopener">المصدر ↗</a>
        </div>
      </article>`
  );
}

async function loadEntities(etype = "") {
  let url = "/api/entities/top?limit=18";
  if (etype) url += "&entity_type=" + encodeURIComponent(etype);
  const cloud = document.getElementById("entities-cloud");
  const res = await fetch(url);
  const data = await res.json();
  if (!data.entities || !data.entities.length) {
    cloud.innerHTML = `<div class="loading-msg">لا توجد كيانات مطابقة.</div>`;
    return;
  }
  cloud.innerHTML = data.entities
    .map(
      (e) => `
      <button class="entity-chip" data-ename="${e.entity_name}">
        <span>${e.entity_name}</span>
        <span class="ent-type">(${e.entity_type})</span>
        <span class="ent-cnt">${e.mentions_count}</span>
      </button>`
    )
    .join("");

  cloud.querySelectorAll(".entity-chip").forEach((btn) => {
    btn.addEventListener("click", () => selectEntity(btn.dataset.ename, btn));
  });
}

async function selectEntity(entityName, btnEl) {
  document.querySelectorAll(".entity-chip").forEach((b) => b.classList.remove("active"));
  if (btnEl) btnEl.classList.add("active");

  const box = document.getElementById("entity-articles-box");
  const list = document.getElementById("entity-articles-list");
  document.getElementById("entity-box-title").textContent = `الأخبار المرتبطة بـ: ${entityName}`;
  box.classList.remove("hidden");
  list.innerHTML = `<div class="loading-msg">جاري تحميل الأخبار...</div>`;

  const res = await fetch(`/api/entities/${encodeURIComponent(entityName)}/articles?limit=500`);
  const data = await res.json();
  renderPaginatedList(
    data.articles || [],
    list,
    null,
    (a) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge">${a.category || ""}</span>
          <span class="badge">${a.source_name || ""}</span>
          ${a.impact_label ? `<span class="badge">${a.impact_label}</span>` : ""}
        </div>
        <h3>${a.title}</h3>
        <p class="alert-rationale">${a.summary_generated || ""}</p>
        <div class="card-foot">
          <span>${a.source_name}</span>
          <a href="${a.canonical_url}" target="_blank" rel="noopener">قراءة الخبر ↗</a>
        </div>
      </article>`,
    500
  );
}

async function loadClusters() {
  const res = await fetch("/api/clusters?limit=500");
  const data = await res.json();
  const grid = document.getElementById("clusters-grid");
  const moreWrap = document.getElementById("clusters-more-wrap");
  if (!data.clusters.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد قصص متعددة المصادر بعد.</div>`;
    if (moreWrap) moreWrap.innerHTML = "";
    return;
  }
  renderPaginatedList(
    data.clusters,
    grid,
    moreWrap,
    (cl) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge badge-amber">${cl.unique_media_count} صحف غطّت الحدث</span>
          <span class="badge">${cl.category}</span>
          <span class="badge ${cl.egypt_impact_label.includes("إيجابي") ? "badge-pos" : cl.egypt_impact_label.includes("سلبي") ? "badge-neg" : ""}">${cl.egypt_impact_label}</span>
        </div>
        <h3>${cl.main_title}</h3>
        <p class="alert-rationale">${cl.summary}</p>
        <div class="cluster-sources" style="display:flex;flex-direction:column;gap:8px;margin-top:10px;">
          ${(cl.sources || [])
            .map(
              (s) => `
            <div style="background:rgba(15,23,42,0.55);border:1px solid rgba(148,163,184,0.16);border-radius:10px;padding:10px 12px;">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap;">
                <strong style="color:#38bdf8;font-size:13px;">${s.source_name}</strong>
                <div style="display:flex;gap:6px;align-items:center;font-size:11px;color:#94a3b8;">
                  ${s.tone ? `<span class="badge" style="font-size:10.5px;padding:2px 7px;">${s.tone}</span>` : ""}
                  ${s.event_date ? `<span>تاريخ الحدث: ${String(s.event_date).slice(0, 10)}</span>` : ""}
                  ${s.published_at ? `<span>النشر: ${String(s.published_at).slice(0, 16).replace("T", " ")}</span>` : ""}
                </div>
              </div>
              <div style="font-weight:700;font-size:13.5px;color:#f1f5f9;margin-bottom:4px;">${s.title}</div>
              ${s.summary_generated ? `<div style="font-size:12.5px;color:#cbd5e1;line-height:1.55;margin-bottom:6px;">${s.summary_generated.slice(0, 190)}${s.summary_generated.length > 190 ? "..." : ""}</div>` : ""}
              <a href="${s.canonical_url}" target="_blank" rel="noopener" style="font-size:12px;color:#2dd4bf;text-decoration:none;font-weight:600;">فتح تغطية ${s.source_name} ↗</a>
            </div>`
            )
            .join("")}
        </div>
      </article>`,
    18
  );
}

async function loadDataQuality() {
  const overallBox = document.getElementById("quality-overall-box");
  const grid = document.getElementById("quality-sources-grid");
  if (!overallBox || !grid) return;

  try {
    const res = await fetch("/api/data-quality");
    const data = await res.json();
    const ov = data.overall || {};
    overallBox.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;">
        <div>
          <h4 class="bento-box-title" style="margin-bottom:4px;">ملخص اكتمال قاعدة البيانات (${(ov.total_articles || 0).toLocaleString("ar-EG")} مقال)</h4>
          <span style="font-size:12.5px;color:#94a3b8;">آخر دورة سحب: ${data.last_scrape_cairo || "متصل"}</span>
        </div>
        <div style="display:flex;gap:14px;flex-wrap:wrap;">
          <span class="badge badge-pos">اكتمال التواريخ: ${ov.date_completeness_pct ?? 100}%</span>
          <span class="badge badge-pos">اكتمال الملخصات: ${ov.summary_completeness_pct ?? 100}%</span>
          <span class="badge badge-amber">نجاح التحليل: ${ov.llm_success_rate_pct ?? 100}% (الأعطال: ${ov.llm_failures_total ?? 0})</span>
          <span class="badge">تغطية الـ Embeddings: ${ov.rag_embedding_coverage_pct ?? 100}%</span>
        </div>
      </div>
    `;

    grid.innerHTML = (data.sources || [])
      .map(
        (s) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge ${s.extraction_status === "healthy" ? "badge-pos" : "badge-amber"}">${s.extraction_status === "healthy" ? "سحب مستقر" : s.extraction_status}</span>
          <span class="badge">${s.analyzed_total.toLocaleString("ar-EG")} مقال محلل</span>
        </div>
        <h3>${s.source_name}</h3>
        <div style="display:flex;flex-direction:column;gap:6px;font-size:13px;color:#cbd5e1;margin:8px 0;">
          <div>• وجود تاريخ نشر صالح: <strong>${s.date_completeness_pct}%</strong></div>
          <div>• وجود ملخص كافٍ (≥ 60 حرف): <strong>${s.summary_completeness_pct}%</strong></div>
          <div>• متوسط طول الخبر: <strong>${s.avg_word_count || "—"} كلمة</strong></div>
          <div>• أخبار مرتبطة بتاريخ حدث صريح: <strong>${s.explicit_event_date_count}</strong></div>
          <div>• فشل التحليل المسجل: <strong>${s.llm_failures_count} (${s.llm_failure_rate_pct}%)</strong></div>
        </div>
      </article>`
      )
      .join("");
  } catch (err) {
    grid.innerHTML = `<div class="loading-msg">تعذر تحميل مؤشرات الجودة.</div>`;
  }
}

let currentArticlesCategory = "";
async function loadAllArticles(q = "", category = currentArticlesCategory) {
  currentArticlesCategory = category;
  let url = `/api/articles?limit=3000`;
  if (q.trim()) url += `&q=${encodeURIComponent(q.trim())}`;
  if (category) url += `&category=${encodeURIComponent(category)}`;

  const grid = document.getElementById("all-articles-grid");
  const moreWrap = document.getElementById("all-articles-more-wrap");
  const totalEl = document.getElementById("all-articles-total");
  grid.innerHTML = `<div class="loading-msg">جاري تحميل كل الأخبار...</div>`;
  if (moreWrap) moreWrap.innerHTML = "";

  const res = await fetch(url);
  const data = await res.json();
  if (totalEl) totalEl.textContent = (data.total || 0).toLocaleString("ar-EG");

  if (!data.articles || !data.articles.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد مقالات مطابقة للبحث.</div>`;
    return;
  }

  renderPaginatedList(
    data.articles,
    grid,
    moreWrap,
    (a) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge">${a.category || ""}</span>
          <span class="badge">${a.source_name || ""}</span>
          ${a.impact_label ? `<span class="badge ${a.impact_label.includes("إيجابي") ? "badge-pos" : a.impact_label.includes("سلبي") ? "badge-neg" : ""}">${a.impact_label}</span>` : ""}
          ${a.impact_level ? `<span class="badge badge-amber">${a.impact_level}</span>` : ""}
        </div>
        <h3>${a.title}</h3>
        <p class="alert-rationale">${a.summary_generated || ""}</p>
        <div class="card-foot">
          <span>${a.source_name}</span>
          <a href="${a.canonical_url}" target="_blank" rel="noopener">قراءة المقال ↗</a>
        </div>
      </article>`,
    36
  );
}

function switchWorkspaceTab(targetId) {
  document.querySelectorAll(".wtab").forEach((t) => {
    const isAct = t.dataset.target === targetId;
    t.classList.toggle("active", isAct);
    t.setAttribute("aria-selected", isAct ? "true" : "false");
  });
  document.querySelectorAll(".wpanel").forEach((p) => {
    p.classList.toggle("active", p.id === targetId);
  });
}

// --- Wire Up Events ---
document.addEventListener("DOMContentLoaded", () => {
  initEgyptMap();
  loadOverview();
  loadGovernorates();
  loadHighImpact("all");
  loadIssues();
  loadEvents();
  loadEntities();
  loadClusters();
  loadAllArticles();
  loadDataQuality();

  const motionBtn = document.getElementById("btn-toggle-motion");
  if (motionBtn) {
    const syncMotionLabel = () => {
      motionBtn.textContent = reducedMotion ? "تشغيل الحركة" : "إيقاف الحركة";
      motionBtn.setAttribute("aria-pressed", reducedMotion ? "true" : "false");
    };
    syncMotionLabel();
    motionBtn.addEventListener("click", () => {
      reducedMotion = !reducedMotion;
      syncMotionLabel();
    });
  }

  const menuBtn = document.getElementById("mobile-menu-btn");
  const topNav = document.getElementById("top-nav");
  if (menuBtn && topNav) {
    menuBtn.addEventListener("click", () => {
      const open = topNav.classList.toggle("open");
      menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  const govSel = document.getElementById("gov-mobile-select");
  if (govSel) {
    govSel.addEventListener("change", (e) => {
      if (e.target.value) selectGovernorate(e.target.value);
    });
  }

function openRagChat() {
  const panel = document.getElementById("rag-chat-panel");
  if (!panel) return;
  panel.classList.remove("hidden");
  gsap.fromTo(panel, { y: 24, opacity: 0 }, { y: 0, opacity: 1, duration: 0.28, ease: "power2.out" });
  const inp = document.getElementById("rag-input");
  if (inp) inp.focus();
}

function closeRagChat() {
  const panel = document.getElementById("rag-chat-panel");
  if (panel) panel.classList.add("hidden");
}

function appendRagMessage(role, htmlText, sources = []) {
  const container = document.getElementById("rag-messages");
  if (!container) return null;

  const msg = document.createElement("div");
  msg.className = `rag-msg ${role === "user" ? "rag-msg-user" : "rag-msg-bot"}`;

  let sourcesHtml = "";
  if (sources && sources.length) {
    sourcesHtml = `
      <div class="rag-sources">
        <span class="rag-sources-title">المصادر والأدلة المقتبسة (${sources.length}):</span>
        ${sources
          .slice(0, 5)
          .map(
            (s, idx) => `
          <a class="rag-source-item" href="${s.canonical_url}" target="_blank" rel="noopener" style="flex-direction:column;align-items:flex-start;gap:4px;">
            <div style="display:flex;justify-content:space-between;width:100%;align-items:center;">
              <span><strong>[${s.ref_index || idx + 1}] ${s.source_name || "مصدر"}${s.governorate ? " • " + s.governorate : ""}:</strong> ${(s.title || "").slice(0, 58)}...</span>
              <span>↗</span>
            </div>
            ${s.supporting_quote ? `<div style="font-size:11.5px;color:#94a3b8;border-right:2px solid #2dd4bf;padding-right:6px;">"${s.supporting_quote}"</div>` : ""}
          </a>`
          )
          .join("")}
      </div>
    `;
  }

  msg.innerHTML = `<div class="rag-bubble">${htmlText}</div>${sourcesHtml}`;
  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
  return msg;
}

async function handleRagQuestion(questionText) {
  const q = questionText.trim();
  if (!q) return;

  const govFilter = (document.getElementById("rag-gov-filter") || {}).value || "";
  const catFilter = (document.getElementById("rag-cat-filter") || {}).value || "";
  const daysRaw = (document.getElementById("rag-days-filter") || {}).value || "";
  const daysFilter = daysRaw ? parseInt(daysRaw, 10) : null;

  const activeScopeTags = [];
  if (govFilter) activeScopeTags.push(`📍 ${govFilter}`);
  if (catFilter) activeScopeTags.push(`🗂️ ${catFilter}`);
  if (daysFilter) activeScopeTags.push(`🕒 آخر ${daysFilter} يوم`);
  const scopePrefix = activeScopeTags.length
    ? `<div style="font-size:11px;opacity:0.8;margin-bottom:4px;">[${activeScopeTags.join(" • ")}]</div>`
    : "";

  appendRagMessage("user", `${scopePrefix}${q}`);
  const thinkingMsg = appendRagMessage(
    "bot",
    `<span class="rag-typing"><span class="pulse-dot"></span> جاري البحث في أخبار مصر وتجهيز الإجابة...</span>`
  );

  try {
    // 1. Try dedicated RAG backend endpoint if available
    const ragRes = await fetch("/api/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: q,
        governorate: govFilter || null,
        category: catFilter || null,
        days: daysFilter,
      }),
    });

    if (ragRes.ok) {
      const data = await ragRes.json();
      if (thinkingMsg) thinkingMsg.remove();
      appendRagMessage("bot", data.answer || "تم استرجاع النتائج.", data.sources || []);
      return;
    }

    // 2. Live Database Preview Fallback (while RAG engine is being wired)
    let sources = [];
    if (govFilter) {
      const gRes = await fetch(`/api/governorates/${encodeURIComponent(govFilter)}?limit=8`);
      const gData = await gRes.json();
      sources = gData.articles || [];
    } else if (q.includes("مشكل") || q.includes("أزم")) {
      const iRes = await fetch("/api/issues?limit=6");
      const iData = await iRes.json();
      sources = (iData.issues || []).map((x) => ({
        title: x.issue_description,
        summary_generated: x.evidence || x.title,
        source_name: x.source_name,
        canonical_url: x.canonical_url,
      }));
    } else if (q.includes("أثر") || q.includes("مهم")) {
      const hRes = await fetch("/api/high-impact?only_new_logic=true&limit=6");
      const hData = await hRes.json();
      sources = (hData.alerts || []).map((x) => ({
        title: x.title,
        summary_generated: x.rationale,
        source_name: x.source_name,
        canonical_url: x.canonical_url,
      }));
    } else {
      const keyword = q
        .replace(/ما|أهم|أبرز|آخر|أخبار|في|عن|اليوم|مصر|\؟/g, " ")
        .trim()
        .split(/\s+/)[0] || "";
      let aUrl = `/api/articles?limit=6`;
      if (keyword.length >= 3) aUrl += `&q=${encodeURIComponent(keyword)}`;
      if (catFilter) aUrl += `&category=${encodeURIComponent(catFilter)}`;
      const aRes = await fetch(aUrl);
      const aData = await aRes.json();
      sources = aData.articles || [];
    }

    if (thinkingMsg) thinkingMsg.remove();
    if (!sources.length) {
      appendRagMessage("bot", "لم أجد أخبارًا مطابقة لهذا السؤال في الفلتر الحالي؛ جرّب تغيير القسم أو المحافظة.");
      return;
    }

    const bullets = sources
      .slice(0, 3)
      .map((s) => `• <strong>${s.title}</strong>: ${s.summary_generated || ""}`)
      .join("<br/>");
    appendRagMessage(
      "bot",
      `إليك أبرز ما ورد في قاعدة البيانات حول سؤالك:<br/><br/>${bullets}`,
      sources
    );
  } catch (err) {
    if (thinkingMsg) thinkingMsg.remove();
    appendRagMessage("bot", "حدث خطأ أثناء الاتصال بالسيرفر؛ تأكد من تشغيل الـ API.");
  }
}

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeDrawer();
      closeRagChat();
      const entBox = document.getElementById("entity-articles-box");
      if (entBox) entBox.classList.add("hidden");
    }
  });

  const openRagBtn = document.getElementById("btn-open-rag");
  if (openRagBtn) openRagBtn.addEventListener("click", openRagChat);
  const ragFab = document.getElementById("rag-fab");
  if (ragFab) ragFab.addEventListener("click", openRagChat);
  const ragClose = document.getElementById("rag-close-btn");
  if (ragClose) ragClose.addEventListener("click", closeRagChat);
  const ragClear = document.getElementById("rag-clear-btn");
  if (ragClear) {
    ragClear.addEventListener("click", () => {
      const box = document.getElementById("rag-messages");
      if (box) {
        box.innerHTML = `<div class="rag-msg rag-msg-bot"><div class="rag-bubble">تم مسح المحادثة. اسألني أي سؤال جديد عن أخبار ومحافظات مصر!</div></div>`;
      }
    });
  }

  document.querySelectorAll(".rag-chip").forEach((chip) => {
    chip.addEventListener("click", () => handleRagQuestion(chip.dataset.q));
  });

  const ragForm = document.getElementById("rag-form");
  if (ragForm) {
    ragForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const inp = document.getElementById("rag-input");
      if (!inp || !inp.value.trim()) return;
      const val = inp.value;
      inp.value = "";
      handleRagQuestion(val);
    });
  }
  document.getElementById("btn-reset-globe").addEventListener("click", resetMapView);
  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("entity-box-close").addEventListener("click", () => {
    document.getElementById("entity-articles-box").classList.add("hidden");
    document.querySelectorAll(".entity-chip").forEach((b) => b.classList.remove("active"));
  });

  document.getElementById("gov-search-input").addEventListener("input", (e) => {
    renderGovPills(e.target.value);
  });

  let searchTimer = null;
  document.getElementById("all-articles-search").addEventListener("input", (e) => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => loadAllArticles(e.target.value, currentArticlesCategory), 280);
  });

  document.querySelectorAll(".abtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".abtn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const q = document.getElementById("all-articles-search").value;
      loadAllArticles(q, btn.dataset.acat);
    });
  });

  document.querySelectorAll(".wtab").forEach((tab) => {
    tab.addEventListener("click", () => switchWorkspaceTab(tab.dataset.target));
  });

  document.querySelectorAll("[data-jump-tab]").forEach((link) => {
    link.addEventListener("click", () => switchWorkspaceTab(link.dataset.jumpTab));
  });

  document.querySelectorAll(".dtab").forEach((btn) => {
    btn.addEventListener("click", () => renderDrawerTab(btn.dataset.tab));
  });

  document.querySelectorAll("#tab-impact .fbtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#tab-impact .fbtn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      loadHighImpact(btn.dataset.filter);
    });
  });

  document.querySelectorAll(".ibtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ibtn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      loadIssues(btn.dataset.sev);
    });
  });

  document.querySelectorAll(".ebtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".ebtn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      loadEvents(btn.dataset.status);
    });
  });

  document.querySelectorAll(".entbtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".entbtn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      loadEntities(btn.dataset.etype);
    });
  });

  setInterval(() => {
    loadOverview();
  }, 60000);
});
