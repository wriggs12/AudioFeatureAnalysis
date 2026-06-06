import pandas as pd
import numpy as np
import json

df = pd.read_csv("clusters_3d.csv")

# --- Build k-nearest-neighbor edges from 3D UMAP coords ---
K = 4
coords = df[["x", "y", "z"]].values
nodes = []
cluster_ids = sorted(df["cluster"].unique())

PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
    "#dcbeff",
    "#9a6324",
    "#fffac8",
    "#800000",
    "#aaffc3",
    "#808000",
    "#ffd8b1",
    "#000075",
    "#a9a9a9",
]
cluster_color = {
    int(cid): PALETTE[i % len(PALETTE)]
    for i, cid in enumerate(c for c in cluster_ids if c != -1)
}
cluster_color[-1] = "#555555"

for i, row in df.iterrows():
    nodes.append(
        {
            "id": i,
            "label": row["track"],
            "cluster": int(row["cluster"]),
            "color": cluster_color[int(row["cluster"])],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
        }
    )

print("Computing KNN edges...")
edges = set()
for i in range(len(coords)):
    diffs = coords - coords[i]
    dists = (diffs**2).sum(axis=1)
    dists[i] = np.inf
    nearest = np.argpartition(dists, K)[:K]
    for j in nearest:
        a, b = (i, int(j)) if i < int(j) else (int(j), i)
        edges.add((a, b))

edges = [{"source": a, "target": b} for a, b in edges]
print(f"{len(nodes)} nodes, {len(edges)} edges")

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>Song Graph 3D</title>
  <script type="importmap">
  {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.165.0/examples/jsm/"}}}}
  </script>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f0f1a; overflow: hidden; font-family: sans-serif; }}
    #ui {{
      position: fixed; top: 12px; right: 12px; z-index: 10;
      display: flex; gap: 6px;
    }}
    input {{
      padding: 6px 10px; border-radius: 6px; border: 1px solid #444;
      background: #1a1a2e; color: #eee; font-size: 13px; width: 230px;
    }}
    button {{
      padding: 6px 12px; border-radius: 6px; border: none;
      background: #4363d8; color: white; cursor: pointer; font-size: 13px;
    }}
    #tooltip {{
      position: fixed; pointer-events: none; display: none;
      background: rgba(0,0,0,0.82); color: #eee; padding: 6px 10px;
      border-radius: 6px; font-size: 13px; max-width: 280px;
    }}
    #legend {{
      position: fixed; bottom: 12px; left: 12px; z-index: 10;
      background: rgba(0,0,0,0.55); padding: 8px 12px; border-radius: 8px;
      color: #ccc; font-size: 11px; max-height: 260px; overflow-y: auto; width: 180px;
    }}
    .legend-item {{
      display: flex; align-items: center; gap: 6px; margin: 2px 0;
      cursor: pointer; border-radius: 4px; padding: 2px 4px;
    }}
    .legend-item:hover {{ background: rgba(255,255,255,0.08); }}
    .legend-item.focused {{ background: rgba(255,255,255,0.15); outline: 1px solid rgba(255,255,255,0.3); }}
    .legend-item.dimmed {{ opacity: 0.25; }}
    .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
    #reset-btn {{
      display: none; margin-top: 6px; width: 100%; padding: 4px 0;
      border-radius: 5px; border: 1px solid #555; background: rgba(255,255,255,0.1);
      color: #eee; cursor: pointer; font-size: 11px;
    }}
    #reset-btn:hover {{ background: rgba(255,255,255,0.2); }}
    #hint {{
      position: fixed; bottom: 12px; right: 12px; color: #555; font-size: 11px; z-index: 10;
    }}
    #song-panel {{
      position: fixed; top: 0; right: -320px; width: 300px; height: 100vh;
      background: rgba(15,15,30,0.95); border-left: 1px solid #2a2a4a;
      z-index: 20; display: flex; flex-direction: column;
      transition: right 0.25s ease;
    }}
    #song-panel.open {{ right: 0; }}
    #panel-header {{
      padding: 14px 16px 10px; border-bottom: 1px solid #2a2a4a;
      display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
    }}
    #panel-title {{ color: #eee; font-size: 14px; font-weight: 600; }}
    #panel-close {{
      background: none; border: none; color: #888; font-size: 18px;
      cursor: pointer; padding: 0 4px; line-height: 1;
    }}
    #panel-close:hover {{ color: #eee; }}
    #panel-search {{
      padding: 8px 12px; border-bottom: 1px solid #1e1e3a; flex-shrink: 0;
    }}
    #panel-search input {{
      width: 100%; padding: 5px 8px; border-radius: 5px; border: 1px solid #333;
      background: #1a1a2e; color: #eee; font-size: 12px;
    }}
    #song-list {{
      overflow-y: auto; flex: 1; padding: 6px 0;
    }}
    .song-row {{
      padding: 5px 10px 5px 16px; font-size: 12px; color: #ccc;
      border-bottom: 1px solid #1a1a2e;
      display: flex; align-items: center; gap: 6px;
    }}
    .song-row:hover {{ background: rgba(255,255,255,0.05); }}
    .song-name {{
      flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
      cursor: pointer; color: #ccc;
    }}
    .song-name:hover {{ color: #fff; text-decoration: underline; }}
    .song-name.has-link {{ color: #ff4444; }}
    .song-name.has-link:hover {{ color: #ff6666; }}
    #yt-modal {{
      display: none; position: fixed; inset: 0; z-index: 100;
      background: rgba(0,0,0,0.7); align-items: center; justify-content: center;
    }}
    #yt-modal.open {{ display: flex; }}
    #yt-modal-box {{
      background: #1a1a2e; border: 1px solid #333; border-radius: 10px;
      padding: 20px; width: 420px; max-width: 90vw;
    }}
    #yt-modal-title {{ color: #eee; font-size: 13px; margin-bottom: 12px; font-weight: 600; }}
    #yt-modal-input {{
      width: 100%; padding: 7px 10px; border-radius: 6px; border: 1px solid #444;
      background: #0f0f1a; color: #eee; font-size: 13px; margin-bottom: 12px;
    }}
    #yt-modal-actions {{ display: flex; gap: 8px; justify-content: flex-end; }}
    #yt-modal-actions button {{ font-size: 12px; padding: 5px 14px; }}
    #yt-clear {{ background: #5a1a1a; }}
    #yt-clear:hover {{ background: #7a2a2a; }}
    #yt-save {{ background: #1a3a6a; }}
    #yt-save:hover {{ background: #2a4a8a; }}
    #yt-cancel {{ background: #333; }}
    #yt-cancel:hover {{ background: #444; }}
  </style>
</head>
<body>
<div id="ui">
  <input id="search" type="text" placeholder="Search song..." />
  <button onclick="doSearch()">Find</button>
</div>
<div id="tooltip"></div>
<div id="legend"></div>
<div id="hint">Drag to rotate · Scroll to zoom · Right-drag to pan</div>
<div id="song-panel">
  <div id="panel-header">
    <span id="panel-title">Cluster</span>
    <button id="panel-close" onclick="focusCluster(null)">&#x2715;</button>
  </div>
  <div id="panel-search"><input id="panel-filter" type="text" placeholder="Filter songs..." /></div>
  <div id="song-list"></div>
</div>

<div id="yt-modal">
  <div id="yt-modal-box">
    <div id="yt-modal-title"></div>
    <input id="yt-modal-input" type="text" placeholder="Paste YouTube URL..." />
    <div id="yt-modal-actions">
      <button id="yt-clear">Clear</button>
      <button id="yt-cancel">Cancel</button>
      <button id="yt-save">Save</button>
    </div>
  </div>
</div>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
const RAW_NODES = {json.dumps(nodes)};
const RAW_EDGES = {json.dumps(edges)};
const CLUSTER_IDS = {json.dumps([int(c) for c in cluster_ids])};
const CLUSTER_COLORS = {json.dumps(cluster_color)};

// ── Three.js setup ──────────────────────────────────────────────
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x0f0f1a);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1000);
camera.position.set(0, 0, 30);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

window.addEventListener("resize", () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

// ── Normalize coords to [-10, 10] ──────────────────────────────
const xs = RAW_NODES.map(n => n.x), ys = RAW_NODES.map(n => n.y), zs = RAW_NODES.map(n => n.z);
const xMin = Math.min(...xs), xMax = Math.max(...xs);
const yMin = Math.min(...ys), yMax = Math.max(...ys);
const zMin = Math.min(...zs), zMax = Math.max(...zs);
const SCALE = 20;

function norm(v, lo, hi) {{ return (v - lo) / (hi - lo) * SCALE - SCALE / 2; }}

const nodes3d = RAW_NODES.map(n => ({{
  ...n,
  px: norm(n.x, xMin, xMax),
  py: norm(n.y, yMin, yMax),
  pz: norm(n.z, zMin, zMax),
}}));

// ── Nodes (instanced mesh for perf) ────────────────────────────
const NODE_R = 0.18;
const nodeGeo = new THREE.SphereGeometry(NODE_R, 8, 8);
const nodeMat = new THREE.MeshBasicMaterial({{ vertexColors: true }});

// Build per-instance color array
const dummy = new THREE.Object3D();
const instancedMesh = new THREE.InstancedMesh(nodeGeo, new THREE.MeshBasicMaterial(), nodes3d.length);
instancedMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

const colorArr = new Float32Array(nodes3d.length * 3);
nodes3d.forEach((n, i) => {{
  dummy.position.set(n.px, n.py, n.pz);
  dummy.updateMatrix();
  instancedMesh.setMatrixAt(i, dummy.matrix);
  const c = new THREE.Color(n.color);
  colorArr[i * 3]     = c.r;
  colorArr[i * 3 + 1] = c.g;
  colorArr[i * 3 + 2] = c.b;
  instancedMesh.setColorAt(i, c);
}});
instancedMesh.instanceMatrix.needsUpdate = true;
scene.add(instancedMesh);

// ── Edges ───────────────────────────────────────────────────────
const edgePositions = [];
RAW_EDGES.forEach(e => {{
  const s = nodes3d[e.source], t = nodes3d[e.target];
  edgePositions.push(s.px, s.py, s.pz, t.px, t.py, t.pz);
}});
const edgeGeo = new THREE.BufferGeometry();
edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(edgePositions, 3));
const edgeMat = new THREE.LineBasicMaterial({{ color: 0xffffff, transparent: true, opacity: 0.08 }});
const edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
scene.add(edgeLines);

// ── Focus state ─────────────────────────────────────────────────
let focusedCluster = null;

function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3),16)/255;
  const g = parseInt(hex.slice(3,5),16)/255;
  const b = parseInt(hex.slice(5,7),16)/255;
  return {{ r, g, b }};
}}

function applyFocus() {{
  nodes3d.forEach((n, i) => {{
    const visible = focusedCluster === null || n.cluster === focusedCluster;
    const c = visible ? new THREE.Color(n.color) : new THREE.Color(0x111111);
    instancedMesh.setColorAt(i, c);
  }});
  instancedMesh.instanceColor.needsUpdate = true;
  edgeMat.opacity = focusedCluster === null ? 0.08 : 0.04;

  document.getElementById("reset-btn").style.display = focusedCluster !== null ? "block" : "none";
  document.querySelectorAll(".legend-item").forEach(el => {{
    const id = parseInt(el.dataset.cluster);
    el.classList.toggle("focused", focusedCluster === id);
    el.classList.toggle("dimmed", focusedCluster !== null && focusedCluster !== id);
  }});

  // Song panel
  const panel = document.getElementById("song-panel");
  if (focusedCluster === null) {{
    panel.classList.remove("open");
  }} else {{
    const songs = nodes3d.filter(n => n.cluster === focusedCluster).map(n => n.label).sort();
    document.getElementById("panel-title").textContent =
      `Cluster ${{focusedCluster}} — ${{songs.length}} song${{songs.length !== 1 ? "s" : ""}}`;
    document.getElementById("panel-filter").value = "";
    renderSongList(songs);
    panel.classList.add("open");
  }}
}}

// ── YouTube links (persisted in localStorage) ───────────────────
const YT_KEY = "yt_links";
function loadLinks() {{ return JSON.parse(localStorage.getItem(YT_KEY) || "{{}}"); }}
function saveLinks(links) {{ localStorage.setItem(YT_KEY, JSON.stringify(links)); }}

function ytSearchUrl(song) {{
  return "https://www.youtube.com/results?search_query=" + encodeURIComponent(song);
}}

function openSong(song) {{
  const links = loadLinks();
  window.open(links[song] || ytSearchUrl(song), "_blank");
}}

// ── Song list renderer ───────────────────────────────────────────
function renderSongList(songs) {{
  const links = loadLinks();
  const list = document.getElementById("song-list");
  list.innerHTML = "";
  songs.forEach(s => {{
    const hasLink = !!links[s];
    const row = document.createElement("div");
    row.className = "song-row";

    const name = document.createElement("span");
    name.className = "song-name" + (hasLink ? " has-link" : "");
    name.title = hasLink ? links[s] : "Opens YouTube search";
    name.textContent = s;
    name.addEventListener("click", () => openSong(s));

    row.appendChild(name);
    list.appendChild(row);
  }});
}}

// ── YouTube URL modal ────────────────────────────────────────────
let modalSong = null;

function openYtModal(song) {{
  modalSong = song;
  const links = loadLinks();
  document.getElementById("yt-modal-title").textContent = song;
  document.getElementById("yt-modal-input").value = links[song] || "";
  document.getElementById("yt-modal").classList.add("open");
  setTimeout(() => document.getElementById("yt-modal-input").focus(), 50);
}}

function closeYtModal() {{
  document.getElementById("yt-modal").classList.remove("open");
  modalSong = null;
}}

document.getElementById("yt-save").addEventListener("click", () => {{
  const url = document.getElementById("yt-modal-input").value.trim();
  if (modalSong) {{
    const links = loadLinks();
    if (url) links[modalSong] = url; else delete links[modalSong];
    saveLinks(links);
  }}
  closeYtModal();
  // Re-render the current list
  const q = document.getElementById("panel-filter").value.toLowerCase();
  const songs = nodes3d.filter(n => n.cluster === focusedCluster).map(n => n.label).sort();
  renderSongList(q ? songs.filter(s => s.toLowerCase().includes(q)) : songs);
}});

document.getElementById("yt-clear").addEventListener("click", () => {{
  if (modalSong) {{
    const links = loadLinks();
    delete links[modalSong];
    saveLinks(links);
  }}
  closeYtModal();
  const q = document.getElementById("panel-filter").value.toLowerCase();
  const songs = nodes3d.filter(n => n.cluster === focusedCluster).map(n => n.label).sort();
  renderSongList(q ? songs.filter(s => s.toLowerCase().includes(q)) : songs);
}});

document.getElementById("yt-cancel").addEventListener("click", closeYtModal);
document.getElementById("yt-modal").addEventListener("click", e => {{
  if (e.target === document.getElementById("yt-modal")) closeYtModal();
}});
document.getElementById("yt-modal-input").addEventListener("keydown", e => {{
  if (e.key === "Enter") document.getElementById("yt-save").click();
  if (e.key === "Escape") closeYtModal();
}});

document.getElementById("panel-filter").addEventListener("input", e => {{
  if (focusedCluster === null) return;
  const q = e.target.value.toLowerCase();
  const songs = nodes3d.filter(n => n.cluster === focusedCluster).map(n => n.label).sort();
  renderSongList(q ? songs.filter(s => s.toLowerCase().includes(q)) : songs);
}});

function focusCluster(cid) {{
  focusedCluster = (focusedCluster === cid) ? null : cid;
  applyFocus();
}}

// ── Raycasting for hover/click ──────────────────────────────────
const raycaster = new THREE.Raycaster();
raycaster.params.Points = {{ threshold: 0.3 }};
const mouse = new THREE.Vector2();
const tooltip = document.getElementById("tooltip");
let hoveredIdx = null;
const highlightColor = new THREE.Color(0xffffff);

function onMouseMove(e) {{
  mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObject(instancedMesh);
  const hit = hits.length ? hits[0] : null;
  const idx = hit ? hit.instanceId : null;

  if (idx !== hoveredIdx) {{
    if (hoveredIdx !== null) {{
      const n = nodes3d[hoveredIdx];
      const visible = focusedCluster === null || n.cluster === focusedCluster;
      instancedMesh.setColorAt(hoveredIdx, visible ? new THREE.Color(n.color) : new THREE.Color(0x111111));
    }}
    hoveredIdx = idx;
    if (idx !== null) instancedMesh.setColorAt(idx, highlightColor);
    instancedMesh.instanceColor.needsUpdate = true;
  }}

  if (idx !== null) {{
    const n = nodes3d[idx];
    tooltip.style.display = "block";
    tooltip.style.left = (e.clientX + 14) + "px";
    tooltip.style.top = (e.clientY - 10) + "px";
    tooltip.textContent = n.label + " · Cluster " + n.cluster;
  }} else {{
    tooltip.style.display = "none";
  }}
}}

function onClick(e) {{
  if (hoveredIdx !== null) {{
    focusCluster(nodes3d[hoveredIdx].cluster);
  }} else {{
    focusCluster(null);
  }}
}}

renderer.domElement.addEventListener("mousemove", onMouseMove);
renderer.domElement.addEventListener("click", onClick);

// ── Legend ──────────────────────────────────────────────────────
const counts = {{}};
RAW_NODES.forEach(n => counts[n.cluster] = (counts[n.cluster] || 0) + 1);

const legend = document.getElementById("legend");
legend.innerHTML = "<b style='font-size:12px'>Clusters</b>";
CLUSTER_IDS.forEach(cid => {{
  const div = document.createElement("div");
  div.className = "legend-item";
  div.dataset.cluster = cid;
  const dot = document.createElement("div");
  dot.className = "legend-dot";
  dot.style.background = CLUSTER_COLORS[cid];
  const lbl = document.createElement("span");
  lbl.textContent = cid === -1 ? `Noise (${{counts[-1]||0}})` : `Cluster ${{cid}} (${{counts[cid]||0}})`;
  div.appendChild(dot); div.appendChild(lbl);
  div.addEventListener("click", e => {{ e.stopPropagation(); focusCluster(cid); }});
  legend.appendChild(div);
}});

const resetBtn = document.createElement("button");
resetBtn.id = "reset-btn";
resetBtn.textContent = "Show All";
resetBtn.addEventListener("click", e => {{ e.stopPropagation(); focusCluster(null); }});
legend.appendChild(resetBtn);

// ── Search ──────────────────────────────────────────────────────
function doSearch() {{
  const q = document.getElementById("search").value.toLowerCase().trim();
  if (!q) return;
  const found = nodes3d.find(n => n.label.toLowerCase().includes(q));
  if (!found) return;
  const target = new THREE.Vector3(found.px, found.py, found.pz);
  controls.target.copy(target);
  camera.position.copy(target).addScalar(4);
  controls.update();
  instancedMesh.setColorAt(found.id, highlightColor);
  instancedMesh.instanceColor.needsUpdate = true;
  tooltip.style.display = "block";
  tooltip.style.left = "50%"; tooltip.style.top = "20px";
  tooltip.textContent = found.label + " · Cluster " + found.cluster;
}}

document.getElementById("search").addEventListener("keydown", e => {{
  if (e.key === "Enter") doSearch();
}});

// ── Render loop ─────────────────────────────────────────────────
function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>"""

with open("testing.html", "w") as f:
    f.write(html)

print("Saved testing.html")
