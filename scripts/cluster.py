import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Load & prep ---
df = pd.read_csv("features.csv")
tracks = df["track"]

# Encode key and mode as numeric
key_map = {
    "C": 0,
    "C#": 1,
    "D": 2,
    "D#": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "G": 7,
    "G#": 8,
    "A": 9,
    "A#": 10,
    "B": 11,
}
df["key_num"] = df["key"].map(key_map).fillna(-1)
df["mode_num"] = (df["mode"] == "major").astype(int)

feature_cols = [
    "tempo",
    "beat_strength",
    "energy",
    "spectral_centroid",
    "spectral_rolloff",
    "zero_crossing_rate",
    "key_num",
    "mode_num",
    *[c for c in df.columns if c.startswith("mfcc_")],
]

X = df[feature_cols].values
X = StandardScaler().fit_transform(X)

# --- Dimensionality reduction ---
print("Running UMAP (high-dim for clustering)...")
reducer_hd = umap.UMAP(n_components=15, n_neighbors=15, min_dist=0.0, random_state=42)
X_hd = reducer_hd.fit_transform(X)

print("Running UMAP (3D for visualization)...")
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
X_3d = reducer_3d.fit_transform(X)

# --- Clustering ---
print("Running HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2, prediction_data=True)
labels = clusterer.fit_predict(X_hd)

n_noise = (labels == -1).sum()
print(f"Before reassignment: {n_noise} noise points ({n_noise / len(labels):.1%})")

# Assign noise points to their nearest cluster via membership vectors
membership = hdbscan.all_points_membership_vectors(clusterer)
labels = np.array(
    [
        int(np.argmax(membership[i])) if labels[i] == -1 else labels[i]
        for i in range(len(labels))
    ]
)

n_clusters = len(set(labels))
print(f"After reassignment: {n_clusters} clusters, 0 noise points")

# --- Save results ---
out = pd.DataFrame(
    {
        "track": tracks,
        "cluster": labels,
        "x": X_3d[:, 0],
        "y": X_3d[:, 1],
        "z": X_3d[:, 2],
    }
)
out.to_csv("clusters_3d.csv", index=False)
print("Saved clusters_3d.csv")

# --- Plot (3D) ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection="3d")

colors = cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
for i, cluster_id in enumerate(sorted(set(labels))):
    mask = labels == cluster_id
    ax.scatter(
        X_3d[mask, 0],
        X_3d[mask, 1],
        X_3d[mask, 2],
        color=colors[i % len(colors)],
        s=15,
        alpha=0.7,
        label=f"Cluster {cluster_id} (n={mask.sum()})",
    )

ax.set_title(f"Song Clusters ({n_clusters} clusters, {len(tracks)} songs)", fontsize=14)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_zlabel("UMAP 3")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7, framealpha=0.7)
plt.tight_layout()
plt.savefig("song_clusters.png", dpi=150)
print("Saved song_clusters.png")
plt.show()
