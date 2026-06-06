# Music Feature Analysis & 3D Song Graph

An end-to-end pipeline that extracts audio features from a local MP3 library, clusters songs by sonic similarity, and renders the results as an interactive 3D graph in the browser.

---

## Pipeline Overview

```
mp3s/ ──► features.py ──► features.csv ──► cluster.py ──► clusters_3d.csv ──► visualize.py ──► song_graph.html
```

---

## Step 1 — Feature Extraction (`features.py`)

Each MP3 is loaded with **librosa** and processed in parallel via `multiprocessing.Pool`. The following features are extracted per track:

| Feature | Description |
|---|---|
| `tempo` | BPM estimated via beat tracking |
| `beat_strength` | Mean onset strength at beat positions (danceability proxy) |
| `energy` | RMS energy (overall loudness) |
| `spectral_centroid` | Weighted mean frequency — correlates with perceived brightness |
| `spectral_rolloff` | Frequency below which 85% of energy falls — distinguishes tonal vs. noisy content |
| `zero_crossing_rate` | Rate of sign changes — higher in percussive/noisy signals |
| `key` | Estimated musical key via Krumhansl-Schmuckler key profiles applied to chroma features |
| `mode` | Major or minor |
| `mfcc_0` – `mfcc_12` | 13 Mel-frequency cepstral coefficients — capture overall timbre and texture |

Output: `features.csv` — one row per track, 22 columns.

**1,493 songs** processed.

---

## Step 2 — Clustering (`cluster.py`)

### Preprocessing
All numeric features are standardized with `StandardScaler` so that high-magnitude features (e.g. `spectral_centroid`) don't dominate the distance calculations. `key` and `mode` are encoded numerically.

### Dimensionality Reduction (UMAP)
Two UMAP projections are computed from the standardized 22-dimensional feature space:

- **15-dimensional UMAP** (`n_neighbors=15`, `min_dist=0.0`) — used as input to the clustering algorithm. Preserves local structure tightly.
- **3-dimensional UMAP** (`n_neighbors=15`, `min_dist=0.1`) — used for visualization. The x, y, z coordinates represent the song's position in sonic space.

### Clustering (HDBSCAN)
HDBSCAN is run on the 15-dimensional UMAP embedding with `min_cluster_size=10`, `min_samples=2`. Songs that HDBSCAN marks as noise (outliers) are reassigned to their nearest cluster using `all_points_membership_vectors`, ensuring every song belongs to a cluster.

**49 clusters** identified across 1,493 songs.

Output: `clusters_3d.csv` — track name, cluster ID, and x/y/z coordinates.

---

## Step 3 — Visualization (`visualize.py`)

Generates a self-contained `song_graph.html` using **Three.js** (ES modules via CDN).

### Graph Construction
Each song becomes a node placed at its 3D UMAP coordinates. Edges are drawn between each song and its **4 nearest neighbors** in 3D space, forming a k-nearest-neighbor graph. Songs that sound similar are close together and connected; songs that sound different are far apart.

### Interactivity

| Action | Result |
|---|---|
| Drag | Rotate the graph |
| Scroll | Zoom in/out |
| Right-drag | Pan |
| Hover a node | Shows track name and cluster |
| Click a node or legend item | Isolates that cluster — all others dim |
| Click background or "Show All" | Resets to full view |
| Song panel filter | Filters the song list within a focused cluster |
| Click a song name | Opens YouTube search (or a saved URL) in a new tab |
| YouTube links | Saved to `localStorage` and persist across sessions |

---

## Running the Pipeline

```bash
# 1. Extract features from mp3s/
.venv/bin/python3.12 features.py

# 2. Cluster and reduce to 3D
.venv/bin/python3.12 cluster.py

# 3. Generate the interactive graph
.venv/bin/python3.12 visualize.py
```

Then open `song_graph.html` in any browser.

---