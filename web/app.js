/* Egypt Planet Intelligence — High-FPS Optimized 3D Egypt Globe + GSAP */

const GLOBE_RADIUS = 105;

const EGYPT_PLANET_LAYOUT = {
  "القاهرة":       { sLat:  12, sLon:   0 },
  "الجيزة":        { sLat:   4, sLon: -18 },
  "القليوبية":     { sLat:  24, sLon:   6 },
  "المنوفية":      { sLat:  32, sLon: -14 },
  "الغربية":       { sLat:  44, sLon:  -6 },
  "الدقهلية":      { sLat:  46, sLon:  18 },
  "كفر الشيخ":     { sLat:  58, sLon: -16 },
  "دمياط":         { sLat:  60, sLon:  24 },
  "البحيرة":       { sLat:  42, sLon: -36 },
  "الإسكندرية":    { sLat:  52, sLon: -56 },
  "مطروح":         { sLat:  45, sLon: -95 },
  "الشرقية":       { sLat:  32, sLon:  28 },
  "الإسماعيلية":   { sLat:  26, sLon:  50 },
  "بورسعيد":       { sLat:  46, sLon:  48 },
  "السويس":        { sLat:  10, sLon:  46 },
  "شمال سيناء":    { sLat:  36, sLon:  82 },
  "جنوب سيناء":    { sLat:   4, sLon:  78 },
  "الفيوم":        { sLat: -10, sLon: -26 },
  "بني سويف":      { sLat: -14, sLon:  -4 },
  "المنيا":        { sLat: -28, sLon: -12 },
  "أسيوط":         { sLat: -40, sLon:   2 },
  "سوهاج":         { sLat: -50, sLon:  14 },
  "قنا":           { sLat: -56, sLon:  34 },
  "الأقصر":        { sLat: -66, sLon:  20 },
  "أسوان":         { sLat: -76, sLon:   4 },
  "البحر الأحمر":  { sLat: -32, sLon:  64 },
  "الوادي الجديد": { sLat: -38, sLon: -68 },
};

let scene, camera, renderer, globeGroup;
let govMarkers = [];
let labelElements = [];
let pulseRings = [];
let signalArcs = [];
let raycaster, mouse;
let isDragging = false;
let dragDistance = 0;
let prevPointer = { x: 0, y: 0 };
let autoRotate = true;
let heroVisible = true;
let canvasW = window.innerWidth;
let canvasH = window.innerHeight;
let currentGovData = null;
let currentTab = "articles";
let governoratesList = [];

// Pre-allocated scratch vectors (zero GC allocations per frame)
const _globeCenterWorld = new THREE.Vector3();
const _worldPos = new THREE.Vector3();
const _surfaceNormal = new THREE.Vector3();
const _camDir = new THREE.Vector3();
const _projected = new THREE.Vector3();

function latLonToVec3(lat, lon, radius = GLOBE_RADIUS) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  const x = -(radius * Math.sin(phi) * Math.cos(theta));
  const z = radius * Math.sin(phi) * Math.sin(theta);
  const y = radius * Math.cos(phi);
  return new THREE.Vector3(x, y, z);
}

function getGovSphereCoords(gov) {
  if (EGYPT_PLANET_LAYOUT[gov.name]) {
    return EGYPT_PLANET_LAYOUT[gov.name];
  }
  return {
    sLat: (gov.lat - 27.5) * 14,
    sLon: (gov.lon - 31.0) * 18,
  };
}

function initGlobe() {
  const canvas = document.getElementById("globe-canvas");
  const wrap = document.getElementById("globe-canvas-wrap");
  canvasW = wrap.clientWidth;
  canvasH = wrap.clientHeight;

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(42, canvasW / canvasH, 1, 1200);
  camera.position.set(0, 12, 285);

  renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: false,
    alpha: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(1);
  renderer.setSize(canvasW, canvasH);

  raycaster = new THREE.Raycaster();
  mouse = new THREE.Vector2();

  globeGroup = new THREE.Group();
  globeGroup.position.set(canvasW > 960 ? -48 : 0, 4, 0);
  scene.add(globeGroup);

  // 1. Core Sphere (32x24 segments for fast rendering)
  const coreGeo = new THREE.SphereGeometry(GLOBE_RADIUS - 0.8, 32, 24);
  const coreMat = new THREE.MeshBasicMaterial({ color: 0x04171b });
  globeGroup.add(new THREE.Mesh(coreGeo, coreMat));

  // 2. Wireframe Grid
  const wireGeo = new THREE.SphereGeometry(GLOBE_RADIUS - 0.2, 24, 16);
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x115e59,
    wireframe: true,
    transparent: true,
    opacity: 0.2,
  });
  globeGroup.add(new THREE.Mesh(wireGeo, wireMat));

  // 3. Dotted Egypt Planet Surface (2600 points — crisp & ultra-fast)
  const dotPositions = [];
  const highlightPositions = [];
  const totalDots = 2600;
  const goldenRatio = (1 + Math.sqrt(5)) / 2;
  const govSpheres = Object.values(EGYPT_PLANET_LAYOUT);

  for (let i = 0; i < totalDots; i++) {
    const theta = (2 * Math.PI * i) / goldenRatio;
    const phi = Math.acos(1 - (2 * (i + 0.5)) / totalDots);
    const lat = 90 - (phi * 180) / Math.PI;
    const lon = ((theta * 180) / Math.PI) % 360 - 180;

    let minDist = 999;
    for (let j = 0; j < govSpheres.length; j++) {
      const dLat = lat - govSpheres[j].sLat;
      const dLon = lon - govSpheres[j].sLon;
      const d = dLat * dLat + dLon * dLon;
      if (d < minDist) minDist = d;
    }

    if (minDist < 480) {
      const v = latLonToVec3(lat, lon, GLOBE_RADIUS + 0.45);
      highlightPositions.push(v.x, v.y, v.z);
    } else if (i % 2 === 0) {
      const v = latLonToVec3(lat, lon, GLOBE_RADIUS + 0.15);
      dotPositions.push(v.x, v.y, v.z);
    }
  }

  const bgGeo = new THREE.BufferGeometry();
  bgGeo.setAttribute("position", new THREE.Float32BufferAttribute(dotPositions, 3));
  globeGroup.add(
    new THREE.Points(bgGeo, new THREE.PointsMaterial({ color: 0x0f766e, size: 1.4, transparent: true, opacity: 0.45 }))
  );

  const egGeo = new THREE.BufferGeometry();
  egGeo.setAttribute("position", new THREE.Float32BufferAttribute(highlightPositions, 3));
  globeGroup.add(
    new THREE.Points(egGeo, new THREE.PointsMaterial({ color: 0x5eead4, size: 2.1, transparent: true, opacity: 0.88 }))
  );

  // 4. Glowing Nile River Path
  const nileGovOrder = ["أسوان", "الأقصر", "قنا", "سوهاج", "أسيوط", "المنيا", "بني سويف", "الجيزة", "القاهرة"];
  const nilePts = nileGovOrder.map((name) => {
    const c = EGYPT_PLANET_LAYOUT[name];
    return latLonToVec3(c.sLat, c.sLon, GLOBE_RADIUS + 0.9);
  });
  const nileCurve = new THREE.CatmullRomCurve3(nilePts);
  const nileGeo = new THREE.BufferGeometry().setFromPoints(nileCurve.getPoints(48));
  globeGroup.add(
    new THREE.Line(nileGeo, new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.75 }))
  );

  focusGlobeOnCoords(8, 0, false);

  // Pause WebGL rendering when user scrolls down past hero section
  const heroSection = document.getElementById("globe-hero");
  const observer = new IntersectionObserver(
    (entries) => {
      heroVisible = entries[0].isIntersecting;
    },
    { threshold: 0.05 }
  );
  observer.observe(heroSection);

  // Pointer & Drag Interactions (no raycasting on pointermove!)
  wrap.addEventListener(
    "pointerdown",
    (e) => {
      isDragging = true;
      dragDistance = 0;
      autoRotate = false;
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
      if (!isDragging) return;
      const dx = e.clientX - prevPointer.x;
      const dy = e.clientY - prevPointer.y;
      dragDistance += Math.abs(dx) + Math.abs(dy);
      globeGroup.rotation.y += dx * 0.0055;
      globeGroup.rotation.x = Math.max(-1.1, Math.min(1.1, globeGroup.rotation.x + dy * 0.0055));
      prevPointer.x = e.clientX;
      prevPointer.y = e.clientY;
    },
    { passive: true }
  );
  wrap.addEventListener("click", (e) => {
    if (dragDistance < 8) handleGlobeClick(e);
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

function focusGlobeOnCoords(sLat, sLon, animateGsap = true) {
  const targetRotY = -((sLon + 90) * Math.PI) / 180;
  const targetRotX = (sLat * Math.PI) / 180;
  if (animateGsap) {
    gsap.to(globeGroup.rotation, {
      y: targetRotY,
      x: targetRotX,
      duration: 1.15,
      ease: "power3.inOut",
    });
  } else {
    globeGroup.rotation.y = targetRotY;
    globeGroup.rotation.x = targetRotX;
  }
}

function addGovernorateMarkers(govs) {
  const labelsContainer = document.getElementById("globe-labels");
  labelsContainer.innerHTML = "";
  govMarkers = [];
  labelElements = [];

  const cairoCoords = EGYPT_PLANET_LAYOUT["القاهرة"];
  const cairoVec = latLonToVec3(cairoCoords.sLat, cairoCoords.sLon, GLOBE_RADIUS + 1.3);
  const sharedPinGeo = new THREE.SphereGeometry(1, 10, 10);
  const sharedDotGeo = new THREE.SphereGeometry(0.95, 6, 6);
  const sharedDotMat = new THREE.MeshBasicMaterial({ color: 0xffffff });

  govs.forEach((g, idx) => {
    const sc = getGovSphereCoords(g);
    g.sLat = sc.sLat;
    g.sLon = sc.sLon;

    const pos = latLonToVec3(sc.sLat, sc.sLon, GLOBE_RADIUS + 1.5);
    const hasActivity = g.articles_count > 0 || g.events_count > 0 || g.issues_count > 0;
    const size = hasActivity ? Math.min(3.1, 1.45 + Math.log1p(g.articles_count) * 0.32) : 1.25;
    const color = g.issues_count > 0 ? 0xfb7185 : hasActivity ? 0x5eead4 : 0x38bdf8;

    const mesh = new THREE.Mesh(sharedPinGeo, new THREE.MeshBasicMaterial({ color }));
    mesh.scale.setScalar(size);
    mesh.position.copy(pos);
    mesh.userData = g;
    globeGroup.add(mesh);
    govMarkers.push(mesh);

    if (g.name !== "القاهرة" && hasActivity && signalArcs.length < 10) {
      const mid = pos.clone().add(cairoVec).multiplyScalar(0.5);
      const dist = pos.distanceTo(cairoVec);
      mid.normalize().multiplyScalar(GLOBE_RADIUS + 10 + dist * 0.2);
      const curve = new THREE.QuadraticBezierCurve3(cairoVec, mid, pos);
      const arcGeo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
      const arcMat = new THREE.LineBasicMaterial({
        color: g.issues_count > 0 ? 0xfb7185 : 0x2dd4bf,
        transparent: true,
        opacity: 0.35,
      });
      globeGroup.add(new THREE.Line(arcGeo, arcMat));

      const dot = new THREE.Mesh(sharedDotGeo, sharedDotMat);
      globeGroup.add(dot);
      signalArcs.push({ curve, dot, t: (idx * 0.13) % 1 });
    }

    const lbl = document.createElement("div");
    lbl.className = "globe-gov-label" + (g.issues_count > 0 ? " has-issue" : "");
    lbl.dataset.gov = g.name;
    lbl.innerHTML = `<span class="lbl-dot"></span><span>${g.name}</span>${g.articles_count > 0 ? `<span class="lbl-cnt">${g.articles_count}</span>` : ""}`;
    lbl.addEventListener("click", (ev) => {
      ev.stopPropagation();
      selectGovernorate(g.name);
    });
    labelsContainer.appendChild(lbl);
    labelElements.push({ el: lbl, localPos: pos.clone(), visible: true });
  });
}

function updateFloatingLabels() {
  const len = labelElements.length;
  if (!len) return;

  globeGroup.getWorldPosition(_globeCenterWorld);
  const matWorld = globeGroup.matrixWorld;

  for (let i = 0; i < len; i++) {
    const item = labelElements[i];
    _worldPos.copy(item.localPos).applyMatrix4(matWorld);
    _surfaceNormal.copy(_worldPos).sub(_globeCenterWorld).normalize();
    _camDir.copy(camera.position).sub(_worldPos).normalize();
    const dot = _surfaceNormal.dot(_camDir);

    if (dot < 0.14) {
      if (item.visible) {
        item.el.style.opacity = "0";
        item.el.style.pointerEvents = "none";
        item.visible = false;
      }
      continue;
    }

    if (!item.visible) {
      item.el.style.opacity = "1";
      item.el.style.pointerEvents = "auto";
      item.visible = true;
    }

    _projected.copy(_worldPos).project(camera);
    const x = (_projected.x * 0.5 + 0.5) * canvasW;
    const y = (-(_projected.y * 0.5) + 0.5) * canvasH;
    item.el.style.transform = `translate3d(${x.toFixed(0)}px, ${(y - 8).toFixed(0)}px, 0) translate(-50%, -100%)`;
  }
}

function handleGlobeClick(e) {
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

  if (autoRotate && !isDragging) {
    globeGroup.rotation.y += 0.002;
  }
  for (let i = 0; i < signalArcs.length; i++) {
    const a = signalArcs[i];
    a.t = (a.t + 0.0055) % 1;
    a.curve.getPointAt(a.t, a.dot.position);
  }
  renderer.render(scene, camera);
  updateFloatingLabels();
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
  animateNumber("kpi-clusters", c.multi_source_clusters || 0);
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
  const obj = { val: 0 };
  gsap.to(obj, {
    val: target,
    duration: 1.2,
    ease: "power2.out",
    onUpdate: () => {
      el.textContent = Math.round(obj.val).toLocaleString("ar-EG");
    },
  });
}

async function loadGovernorates() {
  const res = await fetch("/api/governorates");
  const data = await res.json();
  governoratesList = data.governorates || [];
  addGovernorateMarkers(governoratesList);

  const pills = document.getElementById("gov-pills");
  pills.innerHTML = governoratesList
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

async function selectGovernorate(govName) {
  autoRotate = false;

  const govObj = governoratesList.find((g) => g.name === govName);
  const sc = govObj ? getGovSphereCoords(govObj) : { sLat: 12, sLon: 0 };
  focusGlobeOnCoords(sc.sLat, sc.sLon, true);

  gsap.to(globeGroup.position, { x: canvasW > 960 ? 35 : 0, y: 4, duration: 1.1, ease: "power3.inOut" });
  gsap.to(camera.position, { x: 0, y: 8, z: 235, duration: 1.15, ease: "power3.inOut" });
  gsap.to("#hero-overlay", { opacity: 0, x: 40, duration: 0.45, pointerEvents: "none" });

  document.querySelectorAll(".gov-pill, .globe-gov-label").forEach((b) => {
    b.classList.toggle("active", b.dataset.gov === govName);
  });

  const drawer = document.getElementById("gov-drawer");
  drawer.classList.remove("hidden");
  gsap.fromTo(drawer, { x: -30, opacity: 0 }, { x: 0, opacity: 1, duration: 0.35, ease: "power2.out" });

  document.getElementById("drawer-gov-name").textContent = govName;
  document.getElementById("drawer-body").innerHTML = `<div class="loading-msg">جاري تحميل أخبار وأحداث ${govName}...</div>`;

  const res = await fetch(`/api/governorates/${encodeURIComponent(govName)}`);
  currentGovData = await res.json();

  document.getElementById("d-articles-cnt").textContent = currentGovData.articles.length;
  document.getElementById("d-events-cnt").textContent = currentGovData.events.length;
  document.getElementById("d-issues-cnt").textContent = currentGovData.issues.length;
  renderDrawerTab(currentTab);
}

function resetGlobeView() {
  autoRotate = true;
  closeDrawer();

  gsap.to(globeGroup.position, { x: canvasW > 960 ? -48 : 0, y: 4, duration: 1.1, ease: "power3.inOut" });
  gsap.to(camera.position, { x: 0, y: 12, z: 285, duration: 1.15, ease: "power3.inOut" });
  gsap.to("#hero-overlay", { opacity: 1, x: 0, duration: 0.6, pointerEvents: "auto" });
}

function renderDrawerTab(tab) {
  currentTab = tab;
  document.querySelectorAll(".dtab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  const body = document.getElementById("drawer-body");
  if (!currentGovData) return;

  if (tab === "articles") {
    if (!currentGovData.articles.length) {
      body.innerHTML = `<div class="loading-msg">لا توجد مقالات مسجلة لهذه المحافظة بعد.</div>`;
      return;
    }
    body.innerHTML = currentGovData.articles
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
      .join("");
  } else if (tab === "events") {
    if (!currentGovData.events.length) {
      body.innerHTML = `<div class="loading-msg">لا توجد أحداث مستخرجة لهذه المحافظة بعد.</div>`;
      return;
    }
    body.innerHTML = currentGovData.events
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
      .join("");
  } else {
    if (!currentGovData.issues.length) {
      body.innerHTML = `<div class="loading-msg">لا توجد مشكلات مرصودة لهذه المحافظة بعد.</div>`;
      return;
    }
    body.innerHTML = currentGovData.issues
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
      .join("");
  }
}

function closeDrawer() {
  document.getElementById("gov-drawer").classList.add("hidden");
  document.querySelectorAll(".gov-pill, .globe-gov-label").forEach((b) => b.classList.remove("active"));
}

async function loadHighImpact(filter = "new") {
  let url = "/api/high-impact?limit=24";
  if (filter === "new") url += "&only_new_logic=true";
  if (filter === "pos") url += "&label=" + encodeURIComponent("إيجابي محتمل");
  if (filter === "neg") url += "&label=" + encodeURIComponent("سلبي محتمل");

  const grid = document.getElementById("high-impact-grid");
  grid.innerHTML = `<div class="loading-msg">جاري تحميل الأخبار عالية الأثر...</div>`;

  const res = await fetch(url);
  const data = await res.json();
  if (!data.alerts.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد تنبيهات مطابقة لهذا الفلتر حاليًا.</div>`;
    return;
  }
  grid.innerHTML = data.alerts
    .map(
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
    )
    .join("");
}

async function loadClusters() {
  const res = await fetch("/api/clusters?limit=12");
  const data = await res.json();
  const grid = document.getElementById("clusters-grid");
  if (!data.clusters.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد قصص متعددة المصادر بعد.</div>`;
    return;
  }
  grid.innerHTML = data.clusters
    .map(
      (cl) => `
      <article class="alert-card">
        <div class="badges-row">
          <span class="badge badge-amber">${cl.unique_media_count} مواقع غطّت القصة</span>
          <span class="badge">${cl.category}</span>
          <span class="badge ${cl.egypt_impact_label.includes("إيجابي") ? "badge-pos" : cl.egypt_impact_label.includes("سلبي") ? "badge-neg" : ""}">${cl.egypt_impact_label}</span>
        </div>
        <h3>${cl.main_title}</h3>
        <p class="alert-rationale">${cl.summary}</p>
        <div class="cluster-sources">
          ${(cl.sources || [])
            .map(
              (s) => `
            <a class="cluster-src-link" href="${s.canonical_url}" target="_blank" rel="noopener">
              <span>${s.source_name}: ${s.title.slice(0, 58)}...</span>
              <span>↗</span>
            </a>`
            )
            .join("")}
        </div>
      </article>`
    )
    .join("");
}

async function loadIssues(severity = "") {
  let url = "/api/issues?limit=24";
  if (severity) url += "&severity=" + encodeURIComponent(severity);
  const grid = document.getElementById("issues-grid");
  const res = await fetch(url);
  const data = await res.json();
  if (!data.issues || !data.issues.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد مشكلات مطابقة لهذا الفلتر.</div>`;
    return;
  }
  grid.innerHTML = data.issues
    .map(
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
    )
    .join("");
}

async function loadEvents(status = "") {
  let url = "/api/events?limit=24";
  if (status) url += "&status=" + encodeURIComponent(status);
  const grid = document.getElementById("events-grid");
  const res = await fetch(url);
  const data = await res.json();
  if (!data.events || !data.events.length) {
    grid.innerHTML = `<div class="loading-msg">لا توجد أحداث مطابقة لهذه المرحلة.</div>`;
    return;
  }
  grid.innerHTML = data.events
    .map(
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
    )
    .join("");
}

async function loadEntities(etype = "") {
  let url = "/api/entities/top?limit=20";
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

  const res = await fetch(`/api/entities/${encodeURIComponent(entityName)}/articles?limit=12`);
  const data = await res.json();
  list.innerHTML = (data.articles || [])
    .map(
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
      </article>`
    )
    .join("");
}

// --- Wire Up Events ---
document.addEventListener("DOMContentLoaded", () => {
  initGlobe();
  loadOverview();
  loadGovernorates();
  loadHighImpact("new");
  loadIssues();
  loadEvents();
  loadEntities();
  loadClusters();

  document.getElementById("btn-zoom-egypt").addEventListener("click", () => selectGovernorate("القاهرة"));
  document.getElementById("hero-enter-egypt").addEventListener("click", () => {
    autoRotate = false;
    gsap.to("#hero-overlay", { opacity: 0, x: 40, duration: 0.45, pointerEvents: "none" });
    gsap.to(globeGroup.position, { x: 0, y: 4, duration: 1.0, ease: "power3.inOut" });
    gsap.to(camera.position, { x: 0, y: 6, z: 235, duration: 1.05, ease: "power3.inOut" });
  });
  document.getElementById("btn-reset-globe").addEventListener("click", resetGlobeView);
  document.getElementById("drawer-close").addEventListener("click", closeDrawer);
  document.getElementById("entity-box-close").addEventListener("click", () => {
    document.getElementById("entity-articles-box").classList.add("hidden");
    document.querySelectorAll(".entity-chip").forEach((b) => b.classList.remove("active"));
  });

  document.querySelectorAll(".dtab").forEach((btn) => {
    btn.addEventListener("click", () => renderDrawerTab(btn.dataset.tab));
  });

  document.querySelectorAll("#high-impact .fbtn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#high-impact .fbtn").forEach((b) => b.classList.remove("active"));
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

  // Silent Auto-Refresh every 60s
  setInterval(() => {
    loadOverview();
  }, 60000);
});
