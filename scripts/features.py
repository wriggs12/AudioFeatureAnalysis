import csv
import multiprocessing
import numpy as np
import librosa
from pathlib import Path

# =========================
# CONFIG
# =========================

MP3_DIRECTORY = "./mp3s"
CSV_OUTPUT = "features.csv"

# =========================
# HELPERS
# =========================

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)


def _estimate_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    best_key, best_mode, best_corr = 0, "major", -np.inf
    for i in range(12):
        rotated = np.roll(chroma_mean, -i)
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            corr = float(np.corrcoef(rotated, profile)[0, 1])
            if corr > best_corr:
                best_corr, best_key, best_mode = corr, i, mode
    return KEY_NAMES[best_key], best_mode


def extract_features(mp3_path: str) -> dict | None:
    try:
        y, sr = librosa.load(mp3_path, mono=True)
    except Exception as e:
        print(f"[ERROR] Could not load {mp3_path}: {e}")
        return None

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Beat strength as a danceability proxy
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strength = float(np.mean(onset_env[beats])) if len(beats) else 0.0

    rms = librosa.feature.rms(y=y)
    energy = float(np.mean(rms))

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key, mode = _estimate_key(np.mean(chroma, axis=1))

    return {
        "tempo": tempo,
        "beat_strength": beat_strength,
        "energy": energy,
        "spectral_centroid": spectral_centroid,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "mfccs": mfcc_means,
        "key": key,
        "mode": mode,
    }


# =========================
# MAIN
# =========================


def _process_file(mp3_file: Path) -> tuple[str, dict] | None:
    print(f"\nProcessing: {mp3_file.name}")
    try:
        features = extract_features(str(mp3_file))
        if features is None:
            return None
        mfccs = features.pop("mfccs")
        flat = {f"mfcc_{i}": v for i, v in enumerate(mfccs)}
        row = {**features, **flat}
        track_name = mp3_file.stem
        print(f"{track_name}: {row}")
        return track_name, row
    except Exception as e:
        print(f"[EXCEPTION] {mp3_file.name}: {e}")
        return None


def main():
    mp3_dir = Path(MP3_DIRECTORY)

    if not mp3_dir.exists():
        print(f"Directory does not exist: {mp3_dir}")
        return

    mp3_files = list(mp3_dir.glob("*.mp3"))

    if not mp3_files:
        print("No MP3 files found.")
        return

    with multiprocessing.Pool() as pool:
        results = pool.map(_process_file, mp3_files)

    all_results = {
        name: row for result in results if result is not None for name, row in [result]
    }

    if all_results:
        fieldnames = ["track"] + list(next(iter(all_results.values())).keys())
        with open(CSV_OUTPUT, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for track_name, row in all_results.items():
                writer.writerow({"track": track_name, **row})
        print(f"\nResults written to {CSV_OUTPUT}")

    print("\n=========================")
    print("FINAL RESULTS")
    print("=========================")

    for track_name, row in all_results.items():
        print(f"{track_name}: {row}")


if __name__ == "__main__":
    main()
