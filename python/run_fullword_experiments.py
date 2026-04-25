#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import wave
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "speech_commands_v0.02"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "report" / "figures"

SAMPLE_RATE = 16000
FRAME_MS = 0.025
HOP_MS = 0.01
WINDOW_DURATION_S = 0.10
SCAN_HOP_S = 0.02
NFFT_LOCAL = 512
NFFT_GLOBAL = 2048
N_MELS = 26
N_MFCC = 13
FULL_MELS = 24
FULL_FRAMES = 32
CAP_PER_SPLIT = {"train": 150, "val": 60, "test": 120}
K_GRID = [1, 3, 5, 7, 9]
DTW_TEMPLATE_GRID = [4, 8, 12]
DTW_BAND_GRID = [1, 2, 3]
MAX_KNN_BANK = 80
RNG = np.random.default_rng(0)

TARGET_PHONEMES = ["D", "N", "R", "V"]
TARGET_LABELS = {"D": "d", "N": "n", "R": "r", "V": "v"}

WORD_PHONEMES = {
    "backward": ["B", "AE", "K", "W", "ER", "D"],
    "bed": ["B", "EH", "D"],
    "bird": ["B", "ER", "D"],
    "cat": ["K", "AE", "T"],
    "dog": ["D", "AO", "G"],
    "five": ["F", "AY", "V"],
    "four": ["F", "AO", "R"],
    "go": ["G", "OW"],
    "house": ["HH", "AW", "S"],
    "marvin": ["M", "AA", "R", "V", "IH", "N"],
    "no": ["N", "OW"],
    "on": ["AA", "N"],
    "right": ["R", "AY", "T"],
    "seven": ["S", "EH", "V", "AH", "N"],
    "up": ["AH", "P"],
    "visual": ["V", "IH", "ZH", "UW", "AH", "L"],
    "wow": ["W", "AW"],
    "yes": ["Y", "EH", "S"],
    "zero": ["Z", "IH", "R", "OW"],
}
SELECTED_WORDS = list(WORD_PHONEMES.keys())

FRAME_LENGTH = int(FRAME_MS * SAMPLE_RATE)
HOP_LENGTH = int(HOP_MS * SAMPLE_RATE)
WINDOW_SAMPLES = int(WINDOW_DURATION_S * SAMPLE_RATE)
SCAN_HOP = int(SCAN_HOP_S * SAMPLE_RATE)


def speaker_split(speaker_id: str) -> str:
    bucket = int(speaker_id, 16) % 100
    if bucket < 20:
        return "test"
    if bucket < 30:
        return "val"
    return "train"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav_file:
        signal = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype="<i2")
    return signal.astype(np.float32) / 32768.0


def normalize_signal(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float32)
    signal = signal - np.mean(signal)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal


def frame_signal(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size))
    n_frames = 1 + (signal.size - frame_length) // hop_length
    starts = hop_length * np.arange(n_frames)
    indices = starts[:, None] + np.arange(frame_length)[None, :]
    return signal[indices]


def active_bounds(signal: np.ndarray) -> tuple[int, int]:
    frames = frame_signal(signal, int(0.02 * SAMPLE_RATE), int(0.01 * SAMPLE_RATE))
    energy = np.mean(frames**2, axis=1)
    threshold = max(1e-6, 0.10 * float(np.max(energy)))
    active = np.where(energy >= threshold)[0]
    if active.size == 0:
        return 0, signal.size
    start = max(0, int(active[0] * 0.01 * SAMPLE_RATE - 0.01 * SAMPLE_RATE))
    end = min(signal.size, int(active[-1] * 0.01 * SAMPLE_RATE + 0.02 * SAMPLE_RATE + 0.01 * SAMPLE_RATE))
    if end - start < WINDOW_SAMPLES:
        end = min(signal.size, start + WINDOW_SAMPLES)
    return start, end


def hz_to_mel(frequency_hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + frequency_hz / 700.0)


def mel_to_hz(mel_values: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel_values / 2595.0) - 1.0)


def build_mel_filter_bank(
    sample_rate: int,
    nfft: int,
    n_mels: int,
) -> np.ndarray:
    mel_edges = np.linspace(hz_to_mel(np.array([0.0]))[0], hz_to_mel(np.array([sample_rate / 2.0]))[0], n_mels + 2)
    hz_edges = mel_to_hz(mel_edges)
    bin_edges = np.floor((nfft + 1) * hz_edges / sample_rate).astype(int)
    filter_bank = np.zeros((n_mels, nfft // 2 + 1), dtype=np.float32)
    max_bin = nfft // 2
    for m in range(1, n_mels + 1):
        left = int(np.clip(bin_edges[m - 1], 0, max_bin))
        center = int(np.clip(bin_edges[m], 0, max_bin))
        right = int(np.clip(bin_edges[m + 1], 0, max_bin))
        center = max(center, left + 1)
        right = max(right, center + 1)
        right = min(right, max_bin + 1)
        for k in range(left, center):
            filter_bank[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            filter_bank[m - 1, k] = (right - k) / max(right - center, 1)
    return filter_bank


def build_dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    basis = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    scale = math.sqrt(2.0 / n_mels)
    for k in range(n_mfcc):
        for n in range(n_mels):
            basis[k, n] = math.cos(math.pi * (n + 0.5) * k / n_mels)
    basis *= scale
    basis[0] = math.sqrt(1.0 / n_mels)
    return basis


LOCAL_MEL_FILTER_BANK = build_mel_filter_bank(SAMPLE_RATE, NFFT_LOCAL, N_MELS)
FULL_MEL_FILTER_BANK = build_mel_filter_bank(SAMPLE_RATE, NFFT_LOCAL, FULL_MELS)
DCT_BASIS = build_dct_basis(N_MFCC, N_MELS)


def stft_magnitude(segment: np.ndarray) -> np.ndarray:
    frames = frame_signal(segment, FRAME_LENGTH, HOP_LENGTH) * np.hamming(FRAME_LENGTH)
    return np.abs(np.fft.rfft(frames, n=NFFT_LOCAL, axis=1))


def average_spectrum_feature(magnitude: np.ndarray) -> np.ndarray:
    feature = np.mean(np.log1p(magnitude), axis=0)
    return feature / (np.linalg.norm(feature) + 1e-8)


def mfcc_features(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    power = (magnitude**2) / NFFT_LOCAL
    mel_energy = power @ LOCAL_MEL_FILTER_BANK.T
    log_mel = np.log(mel_energy + 1e-6)
    mfcc = log_mel @ DCT_BASIS.T
    mean_feature = np.mean(mfcc, axis=0)
    mean_feature = mean_feature / (np.linalg.norm(mean_feature) + 1e-8)
    sequence_feature = mfcc - np.mean(mfcc, axis=0, keepdims=True)
    return mean_feature.astype(np.float32), sequence_feature.astype(np.float32)


def resize_time_axis(matrix: np.ndarray, n_frames: int) -> np.ndarray:
    if matrix.shape[1] == 0:
        matrix = np.zeros((matrix.shape[0], 1), dtype=np.float32)
    indices = np.linspace(0, matrix.shape[1] - 1, n_frames).round().astype(int)
    return matrix[:, indices]


def full_logmel_feature(signal: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    start, end = bounds
    active = signal[start:end]
    magnitude = stft_magnitude(active)
    power = (magnitude**2) / NFFT_LOCAL
    mel_energy = power @ FULL_MEL_FILTER_BANK.T
    log_mel = np.log(mel_energy + 1e-6).T
    resized = resize_time_axis(log_mel, FULL_FRAMES)
    resized = (resized - np.mean(resized)) / (np.std(resized) + 1e-6)
    return resized.reshape(-1).astype(np.float32)


def global_fft_feature(signal: np.ndarray, bounds: tuple[int, int]) -> np.ndarray:
    start, end = bounds
    active = signal[start:end]
    windowed = active * np.hamming(active.size)
    magnitude = np.abs(np.fft.rfft(windowed, n=NFFT_GLOBAL))
    feature = np.log1p(magnitude)
    return feature / (np.linalg.norm(feature) + 1e-8)


def scan_window_starts(bounds: tuple[int, int]) -> np.ndarray:
    start, end = bounds
    if end - start <= WINDOW_SAMPLES:
        return np.asarray([start], dtype=np.int64)
    starts = list(range(start, end - WINDOW_SAMPLES + 1, SCAN_HOP))
    last = end - WINDOW_SAMPLES
    if starts[-1] != last:
        starts.append(last)
    return np.asarray(starts, dtype=np.int64)


def extract_scan_features(
    signal: np.ndarray,
    bounds: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = scan_window_starts(bounds)
    spectra = []
    mfcc_means = []
    mfcc_sequences = []
    for start in starts:
        segment = signal[start : start + WINDOW_SAMPLES]
        if segment.size < WINDOW_SAMPLES:
            segment = np.pad(segment, (0, WINDOW_SAMPLES - segment.size))
        magnitude = stft_magnitude(segment)
        spectrum = average_spectrum_feature(magnitude)
        mfcc_mean, mfcc_sequence = mfcc_features(magnitude)
        spectra.append(spectrum)
        mfcc_means.append(mfcc_mean)
        mfcc_sequences.append(mfcc_sequence)
    return (
        starts,
        np.stack(spectra).astype(np.float32),
        np.stack(mfcc_means).astype(np.float32),
        np.stack(mfcc_sequences).astype(np.float32),
    )


def phoneme_occurrences(word: str, target: str) -> list[dict[str, object]]:
    phones = WORD_PHONEMES[word]
    occurrences = []
    for index, phoneme in enumerate(phones):
        if phoneme != target:
            continue
        ratio = (index + 0.5) / len(phones)
        if ratio <= 1.0 / 3.0:
            position = "initial"
        elif ratio >= 2.0 / 3.0:
            position = "final"
        else:
            position = "medial"
        occurrences.append({"index": index, "position": position})
    return occurrences


def pick_occurrence_window(item: dict, target: str, occurrence_index: int) -> int:
    phones = WORD_PHONEMES[item["word"]]
    start, end = item["active_bounds"]
    target_center = start + int((occurrence_index + 0.5) / len(phones) * max(end - start, WINDOW_SAMPLES))
    centers = item["window_starts"] + WINDOW_SAMPLES // 2
    return int(np.argmin(np.abs(centers - target_center)))


def cosine_bank_score(window_features: np.ndarray, template_bank: np.ndarray) -> float:
    similarities = window_features @ template_bank.T
    return float(np.max(similarities))


def knn_bank_score(window_features: np.ndarray, template_bank: np.ndarray, k: int) -> float:
    similarities = window_features @ template_bank.T
    k = min(k, template_bank.shape[0])
    topk = np.partition(similarities, -k, axis=1)[:, -k:]
    return float(np.max(np.mean(topk, axis=1)))


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray, band: int) -> float:
    len_a, len_b = sequence_a.shape[0], sequence_b.shape[0]
    band = max(band, abs(len_a - len_b))
    dp = np.full((len_a + 1, len_b + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, len_a + 1):
        j_start = max(1, i - band)
        j_stop = min(len_b, i + band)
        costs = np.sum((sequence_b[j_start - 1 : j_stop] - sequence_a[i - 1]) ** 2, axis=1)
        for j, cost in zip(range(j_start, j_stop + 1), costs):
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[len_a, len_b] / (len_a + len_b))


def dtw_bank_score(window_sequences: np.ndarray, template_bank: np.ndarray, band: int) -> float:
    best_distance = np.inf
    for query in window_sequences:
        for template in template_bank:
            distance = dtw_distance(query, template, band)
            if distance < best_distance:
                best_distance = distance
    return float(-best_distance)


def choose_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.unique(scores)
    candidates = np.concatenate(
        (
            np.array([candidates[0] - 1e-6]),
            candidates,
            np.array([candidates[-1] + 1e-6]),
        )
    )
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold in candidates:
        metrics = binary_metrics(scores, labels, float(threshold))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = float(threshold)
    return best_threshold, best_f1


def binary_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    tp = float(np.sum((predictions == 1) & (labels == 1)))
    fp = float(np.sum((predictions == 1) & (labels == 0)))
    fn = float(np.sum((predictions == 0) & (labels == 1)))
    tn = float(np.sum((predictions == 0) & (labels == 0)))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def select_representative_bank(features: np.ndarray, max_size: int) -> np.ndarray:
    if features.shape[0] <= max_size:
        return features.astype(np.float32)
    centroid = np.mean(features, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    similarity = features @ centroid
    chosen = np.argsort(similarity)[::-1][:max_size]
    return features[chosen].astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


def train_binary_mlp(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], float]:
    train_mean = np.mean(train_features, axis=0, keepdims=True)
    train_std = np.std(train_features, axis=0, keepdims=True) + 1e-6
    x_train = ((train_features - train_mean) / train_std).astype(np.float32)
    x_val = ((val_features - train_mean) / train_std).astype(np.float32)
    y_train = train_labels.astype(np.float32)[:, None]

    input_dim = x_train.shape[1]
    hidden_dim = 64
    rng = np.random.default_rng(0)

    w1 = rng.normal(scale=np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    w2 = rng.normal(scale=np.sqrt(2.0 / hidden_dim), size=(hidden_dim, 1)).astype(np.float32)
    b2 = np.zeros((1, 1), dtype=np.float32)

    mw1 = np.zeros_like(w1)
    vw1 = np.zeros_like(w1)
    mb1 = np.zeros_like(b1)
    vb1 = np.zeros_like(b1)
    mw2 = np.zeros_like(w2)
    vw2 = np.zeros_like(w2)
    mb2 = np.zeros_like(b2)
    vb2 = np.zeros_like(b2)

    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    learning_rate = 1e-3
    weight_decay = 1e-4
    batch_size = 64
    patience = 12
    max_epochs = 80

    best_state = None
    best_threshold = 0.5
    best_val_f1 = -1.0
    bad_epochs = 0
    step = 0

    for _ in range(max_epochs):
        permutation = rng.permutation(x_train.shape[0])
        for start in range(0, permutation.size, batch_size):
            step += 1
            batch_idx = permutation[start : start + batch_size]
            xb = x_train[batch_idx]
            yb = y_train[batch_idx]

            z1 = xb @ w1 + b1
            h1 = np.maximum(z1, 0.0)
            logits = h1 @ w2 + b2
            probabilities = sigmoid(logits)

            dz2 = (probabilities - yb) / yb.shape[0]
            dw2 = h1.T @ dz2 + weight_decay * w2
            db2 = np.sum(dz2, axis=0, keepdims=True)
            dh1 = dz2 @ w2.T
            dz1 = dh1 * (z1 > 0)
            dw1 = xb.T @ dz1 + weight_decay * w1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            for parameter, gradient, m_buf, v_buf in (
                (w1, dw1, mw1, vw1),
                (b1, db1, mb1, vb1),
                (w2, dw2, mw2, vw2),
                (b2, db2, mb2, vb2),
            ):
                m_buf[:] = beta1 * m_buf + (1.0 - beta1) * gradient
                v_buf[:] = beta2 * v_buf + (1.0 - beta2) * (gradient * gradient)
                m_hat = m_buf / (1.0 - beta1**step)
                v_hat = v_buf / (1.0 - beta2**step)
                parameter[:] -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        val_scores = sigmoid(np.maximum(x_val @ w1 + b1, 0.0) @ w2 + b2).ravel()
        threshold, val_f1 = choose_threshold(val_scores, val_labels)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_threshold = threshold
            best_state = (w1.copy(), b1.copy(), w2.copy(), b2.copy())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("MLP training failed")

    training_state = {"train_mean": train_mean, "train_std": train_std}
    return (*best_state, training_state, best_threshold)


def mlp_predict_scores(
    features: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    training_state: dict[str, np.ndarray],
) -> np.ndarray:
    x = (features - training_state["train_mean"]) / training_state["train_std"]
    logits = np.maximum(x @ w1 + b1, 0.0) @ w2 + b2
    return sigmoid(logits).ravel()


def build_selected_items() -> dict[str, list[dict[str, object]]]:
    items_by_split: dict[str, list[dict[str, object]]] = defaultdict(list)
    count_by_split_word: dict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))
    for word in SELECTED_WORDS:
        for wav_path in sorted((DATA_ROOT / word).glob("*.wav")):
            speaker_id = wav_path.name.split("_nohash_")[0]
            split = speaker_split(speaker_id)
            if count_by_split_word[split][word] >= CAP_PER_SPLIT[split]:
                continue
            items_by_split[split].append({"word": word, "path": wav_path})
            count_by_split_word[split][word] += 1
    return items_by_split


def build_feature_cache(items_by_split: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    cache: dict[str, dict[str, object]] = {}
    flat_items = items_by_split["train"] + items_by_split["val"] + items_by_split["test"]
    for index, item in enumerate(flat_items, start=1):
        path = item["path"]
        signal = normalize_signal(read_wav(path))
        bounds = active_bounds(signal)
        window_starts, scan_spectra, scan_mfcc_mean, scan_mfcc_seq = extract_scan_features(signal, bounds)
        cache[str(path)] = {
            "word": item["word"],
            "path": path,
            "active_bounds": bounds,
            "window_starts": window_starts,
            "scan_spectra": scan_spectra,
            "scan_mfcc_mean": scan_mfcc_mean,
            "scan_mfcc_seq": scan_mfcc_seq,
            "global_fft": global_fft_feature(signal, bounds).astype(np.float32),
            "full_logmel": full_logmel_feature(signal, bounds).astype(np.float32),
        }
        if index % 500 == 0 or index == len(flat_items):
            print(f"cached {index}/{len(flat_items)} utterances")
    return cache


def build_balanced_detector_sets(
    items_by_split: dict[str, list[dict[str, object]]],
    target: str,
) -> dict[str, dict[str, object]]:
    positive_words = {word for word in SELECTED_WORDS if target in WORD_PHONEMES[word]}
    datasets = {}
    for split in ["train", "val", "test"]:
        positives = [item for item in items_by_split[split] if item["word"] in positive_words]
        negatives = [item for item in items_by_split[split] if item["word"] not in positive_words]
        negatives = negatives.copy()
        RNG.shuffle(negatives)
        negatives = negatives[: len(positives)]
        items = positives + negatives
        RNG.shuffle(items)
        labels = np.asarray([1 if item["word"] in positive_words else 0 for item in items], dtype=np.int64)
        datasets[split] = {"items": items, "labels": labels, "positive_words": sorted(positive_words)}
    return datasets


def build_target_banks(
    target: str,
    train_items: list[dict[str, object]],
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    spectra_by_position: dict[str, list[np.ndarray]] = defaultdict(list)
    mfcc_by_position: dict[str, list[np.ndarray]] = defaultdict(list)
    all_spectra = []
    all_mfcc_sequences = []
    all_mfcc_means = []
    for item in train_items:
        if target not in WORD_PHONEMES[item["word"]]:
            continue
        cached = cache[str(item["path"])]
        for occurrence in phoneme_occurrences(item["word"], target):
            window_index = pick_occurrence_window(cached, target, int(occurrence["index"]))
            spectra = cached["scan_spectra"][window_index]
            mfcc_mean = cached["scan_mfcc_mean"][window_index]
            mfcc_sequence = cached["scan_mfcc_seq"][window_index]
            spectra_by_position[str(occurrence["position"])].append(spectra)
            mfcc_by_position[str(occurrence["position"])].append(mfcc_mean)
            all_spectra.append(spectra)
            all_mfcc_means.append(mfcc_mean)
            all_mfcc_sequences.append(mfcc_sequence)

    spectrum_templates = []
    mfcc_templates = []
    for position in ["initial", "medial", "final"]:
        if spectra_by_position[position]:
            spectrum = np.mean(np.stack(spectra_by_position[position]), axis=0)
            spectrum = spectrum / (np.linalg.norm(spectrum) + 1e-8)
            spectrum_templates.append(spectrum)
        if mfcc_by_position[position]:
            mfcc = np.mean(np.stack(mfcc_by_position[position]), axis=0)
            mfcc = mfcc / (np.linalg.norm(mfcc) + 1e-8)
            mfcc_templates.append(mfcc)

    all_spectra_matrix = np.stack(all_spectra).astype(np.float32)
    all_mfcc_sequences_matrix = np.stack(all_mfcc_sequences).astype(np.float32)
    all_mfcc_means_matrix = np.stack(all_mfcc_means).astype(np.float32)

    return {
        "spectrum_mean_bank": np.stack(spectrum_templates).astype(np.float32),
        "mfcc_mean_bank": np.stack(mfcc_templates).astype(np.float32),
        "spectrum_knn_bank": select_representative_bank(all_spectra_matrix, MAX_KNN_BANK),
        "mfcc_selector_bank": all_mfcc_means_matrix,
        "mfcc_sequence_bank": all_mfcc_sequences_matrix,
        "global_fft_template": (
            np.mean(
                np.stack(
                    [
                        cache[str(item["path"])]["global_fft"]
                        for item in train_items
                        if target in WORD_PHONEMES[item["word"]]
                    ]
                ),
                axis=0,
            )
        ).astype(np.float32),
        "average_local_spectrum": np.mean(all_spectra_matrix, axis=0).astype(np.float32),
    }


def evaluate_global_fft(
    datasets: dict[str, dict[str, object]],
    cache: dict[str, dict[str, object]],
    template: np.ndarray,
) -> dict[str, object]:
    template = template / (np.linalg.norm(template) + 1e-8)
    val_scores = np.asarray(
        [float(cache[str(item["path"])]["global_fft"] @ template) for item in datasets["val"]["items"]],
        dtype=np.float32,
    )
    threshold, _ = choose_threshold(val_scores, datasets["val"]["labels"])
    test_scores = np.asarray(
        [float(cache[str(item["path"])]["global_fft"] @ template) for item in datasets["test"]["items"]],
        dtype=np.float32,
    )
    return {
        "val_threshold": threshold,
        "test_scores": test_scores,
        "metrics": binary_metrics(test_scores, datasets["test"]["labels"], threshold),
    }


def evaluate_mean_scan(
    datasets: dict[str, dict[str, object]],
    cache: dict[str, dict[str, object]],
    bank: np.ndarray,
    feature_key: str,
) -> dict[str, object]:
    val_scores = np.asarray(
        [cosine_bank_score(cache[str(item["path"])][feature_key], bank) for item in datasets["val"]["items"]],
        dtype=np.float32,
    )
    threshold, _ = choose_threshold(val_scores, datasets["val"]["labels"])
    test_scores = np.asarray(
        [cosine_bank_score(cache[str(item["path"])][feature_key], bank) for item in datasets["test"]["items"]],
        dtype=np.float32,
    )
    return {
        "val_threshold": threshold,
        "test_scores": test_scores,
        "metrics": binary_metrics(test_scores, datasets["test"]["labels"], threshold),
    }


def evaluate_knn_scan(
    datasets: dict[str, dict[str, object]],
    cache: dict[str, dict[str, object]],
    bank: np.ndarray,
) -> dict[str, object]:
    best_k = K_GRID[0]
    best_threshold = 0.0
    best_f1 = -1.0
    for k in K_GRID:
        val_scores = np.asarray(
            [knn_bank_score(cache[str(item["path"])]["scan_spectra"], bank, k) for item in datasets["val"]["items"]],
            dtype=np.float32,
        )
        threshold, f1 = choose_threshold(val_scores, datasets["val"]["labels"])
        if f1 > best_f1:
            best_f1 = f1
            best_k = k
            best_threshold = threshold
    test_scores = np.asarray(
        [knn_bank_score(cache[str(item["path"])]["scan_spectra"], bank, best_k) for item in datasets["test"]["items"]],
        dtype=np.float32,
    )
    return {
        "best_k": best_k,
        "val_threshold": best_threshold,
        "test_scores": test_scores,
        "metrics": binary_metrics(test_scores, datasets["test"]["labels"], best_threshold),
    }


def evaluate_mfcc_dtw(
    datasets: dict[str, dict[str, object]],
    cache: dict[str, dict[str, object]],
    selector_bank: np.ndarray,
    sequence_bank: np.ndarray,
) -> dict[str, object]:
    centroid = np.mean(selector_bank, axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    selector_similarity = selector_bank @ centroid
    order = np.argsort(selector_similarity)[::-1]
    best_count = DTW_TEMPLATE_GRID[0]
    best_band = DTW_BAND_GRID[0]
    best_threshold = 0.0
    best_f1 = -1.0
    for count in DTW_TEMPLATE_GRID:
        chosen_sequences = sequence_bank[order[:count]]
        for band in DTW_BAND_GRID:
            val_scores = np.asarray(
                [
                    dtw_bank_score(cache[str(item["path"])]["scan_mfcc_seq"], chosen_sequences, band)
                    for item in datasets["val"]["items"]
                ],
                dtype=np.float32,
            )
            threshold, f1 = choose_threshold(val_scores, datasets["val"]["labels"])
            if f1 > best_f1:
                best_f1 = f1
                best_count = count
                best_band = band
                best_threshold = threshold
    chosen_sequences = sequence_bank[order[:best_count]]
    test_scores = np.asarray(
        [
            dtw_bank_score(cache[str(item["path"])]["scan_mfcc_seq"], chosen_sequences, best_band)
            for item in datasets["test"]["items"]
        ],
        dtype=np.float32,
    )
    return {
        "templates": best_count,
        "band": best_band,
        "val_threshold": best_threshold,
        "test_scores": test_scores,
        "metrics": binary_metrics(test_scores, datasets["test"]["labels"], best_threshold),
    }


def evaluate_mlp(
    datasets: dict[str, dict[str, object]],
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    train_features = np.stack([cache[str(item["path"])]["full_logmel"] for item in datasets["train"]["items"]]).astype(np.float32)
    val_features = np.stack([cache[str(item["path"])]["full_logmel"] for item in datasets["val"]["items"]]).astype(np.float32)
    test_features = np.stack([cache[str(item["path"])]["full_logmel"] for item in datasets["test"]["items"]]).astype(np.float32)
    train_labels = datasets["train"]["labels"]
    val_labels = datasets["val"]["labels"]
    test_labels = datasets["test"]["labels"]
    w1, b1, w2, b2, training_state, threshold = train_binary_mlp(train_features, train_labels, val_features, val_labels)
    test_scores = mlp_predict_scores(test_features, w1, b1, w2, b2, training_state)
    return {
        "hidden_dim": 64,
        "val_threshold": threshold,
        "test_scores": test_scores,
        "metrics": binary_metrics(test_scores, test_labels, threshold),
    }


def aggregate_method_results(results: dict[str, dict[str, dict[str, object]]]) -> dict[str, dict[str, float]]:
    method_names = ["global_fft", "mfcc_mean", "mfcc_dtw", "stft_mean", "stft_knn", "mlp"]
    summary = {}
    for method in method_names:
        accuracies = [results[target][method]["metrics"]["accuracy"] for target in TARGET_PHONEMES]
        f1_scores = [results[target][method]["metrics"]["f1"] for target in TARGET_PHONEMES]
        precisions = [results[target][method]["metrics"]["precision"] for target in TARGET_PHONEMES]
        recalls = [results[target][method]["metrics"]["recall"] for target in TARGET_PHONEMES]
        summary[method] = {
            "macro_accuracy": float(np.mean(accuracies)),
            "macro_f1": float(np.mean(f1_scores)),
            "macro_precision": float(np.mean(precisions)),
            "macro_recall": float(np.mean(recalls)),
        }
    return summary


def plot_average_spectra(average_spectra: dict[str, np.ndarray]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    frequency_axis = np.fft.rfftfreq(NFFT_LOCAL, d=1.0 / SAMPLE_RATE)
    plt.figure(figsize=(8.2, 4.8))
    for target in TARGET_PHONEMES:
        average = average_spectra[target]
        average_db = 20.0 * np.log10(average / (np.max(average) + 1e-8) + 1e-8)
        plt.plot(frequency_axis, average_db, linewidth=2, label=TARGET_LABELS[target])
    plt.xlim(0, 4000)
    plt.ylim(-40, 1)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized magnitude (dB)")
    plt.title("Average Local Spectra of Four Voiced Consonants")
    plt.grid(True, alpha=0.25)
    plt.legend(title="Target")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fullword_average_local_spectra.png", dpi=200)
    plt.close()


def plot_method_comparison(summary: dict[str, dict[str, float]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = [
        ("global_fft", "Global FFT"),
        ("mfcc_mean", "MFCC Mean"),
        ("mfcc_dtw", "MFCC + DTW"),
        ("stft_mean", "STFT Mean"),
        ("stft_knn", "STFT k-NN"),
        ("mlp", "AI MLP"),
    ]
    names = [label for _, label in methods]
    f1_values = [100.0 * summary[key]["macro_f1"] for key, _ in methods]
    acc_values = [100.0 * summary[key]["macro_accuracy"] for key, _ in methods]
    x = np.arange(len(methods))
    width = 0.36
    plt.figure(figsize=(8.8, 4.8))
    f1_bars = plt.bar(x - width / 2, f1_values, width=width, label="Macro F1")
    acc_bars = plt.bar(x + width / 2, acc_values, width=width, label="Macro Accuracy")
    plt.xticks(x, names, rotation=18, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.title("Position-Independent Detection Performance")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    for bars in (f1_bars, acc_bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.0,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fullword_method_comparison.png", dpi=200)
    plt.close()


def plot_per_letter_f1(results: dict[str, dict[str, dict[str, object]]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    methods = [("mfcc_dtw", "MFCC + DTW"), ("stft_knn", "STFT k-NN"), ("mlp", "AI MLP")]
    targets = [TARGET_LABELS[target] for target in TARGET_PHONEMES]
    x = np.arange(len(targets))
    width = 0.25
    plt.figure(figsize=(7.8, 4.5))
    for offset, (method_key, label) in zip([-width, 0.0, width], methods):
        values = [results[target][method_key]["metrics"]["f1"] for target in TARGET_PHONEMES]
        bars = plt.bar(x + offset, values, width=width, label=label)
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{100.0 * height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
            )
    plt.xticks(x, targets)
    plt.ylim(0, 1.08)
    plt.ylabel("F1 score")
    plt.xlabel("Target consonant")
    plt.title("Per-Letter Detection F1 on the Test Set")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fullword_per_letter_f1.png", dpi=200)
    plt.close()


def example_target_window(cache_item: dict[str, object], word: str, target: str) -> tuple[int, int]:
    occurrence = phoneme_occurrences(word, target)[0]
    window_index = pick_occurrence_window(cache_item, target, int(occurrence["index"]))
    start = int(cache_item["window_starts"][window_index])
    return start, start + WINDOW_SAMPLES


def plot_position_examples(cache: dict[str, dict[str, object]], items_by_split: dict[str, list[dict[str, object]]]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    choices = [("bed", "D"), ("seven", "N"), ("zero", "R"), ("visual", "V")]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    for axis, (word, target) in zip(axes.ravel(), choices):
        item = next(item for item in items_by_split["train"] if item["word"] == word)
        signal = normalize_signal(read_wav(item["path"]))
        cached = cache[str(item["path"])]
        start, end = cached["active_bounds"]
        active = signal[start:end]
        magnitude = np.log1p(stft_magnitude(active)).T
        time_axis = np.arange(magnitude.shape[1]) * HOP_MS * 1000.0
        freq_axis = np.fft.rfftfreq(NFFT_LOCAL, d=1.0 / SAMPLE_RATE)
        axis.imshow(
            magnitude,
            origin="lower",
            aspect="auto",
            extent=[0.0, time_axis[-1] + FRAME_MS * 1000.0, freq_axis[0], freq_axis[-1]],
            cmap="magma",
        )
        axis.set_ylim(0, 4000)
        axis.set_xlabel("Time (ms)")
        axis.set_ylabel("Frequency (Hz)")
        axis.set_title(f"{TARGET_LABELS[target]} in '{word}'")
        win_start, win_end = example_target_window(cached, word, target)
        rect_start_ms = (win_start - start) / SAMPLE_RATE * 1000.0
        rect_width_ms = (win_end - win_start) / SAMPLE_RATE * 1000.0
        axis.add_patch(
            patches.Rectangle(
                (rect_start_ms, 0.0),
                rect_width_ms,
                4000.0,
                linewidth=1.8,
                edgecolor="white",
                facecolor="none",
            )
        )
    figure.suptitle("Representative Full-Word Spectrograms with Detected Target Regions", fontsize=13)
    figure.savefig(FIG_DIR / "fullword_position_examples.png", dpi=200)
    plt.close(figure)


def write_json(path: Path, data: dict) -> None:
    def json_default(obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=json_default)


def write_summary(path: Path, summary: dict[str, dict[str, float]], results: dict[str, dict[str, dict[str, object]]]) -> None:
    lines = ["Position-independent voiced-consonant detection summary", ""]
    for method_key, label in (
        ("global_fft", "Global FFT"),
        ("mfcc_mean", "MFCC mean-template"),
        ("mfcc_dtw", "MFCC + DTW"),
        ("stft_mean", "STFT mean-template"),
        ("stft_knn", "STFT k-NN"),
        ("mlp", "AI MLP"),
    ):
        lines.append(
            f"{label}: macro F1={summary[method_key]['macro_f1']:.4f}, "
            f"macro accuracy={summary[method_key]['macro_accuracy']:.4f}"
        )
    lines.append("")
    lines.append("Per-letter F1")
    for target in TARGET_PHONEMES:
        pieces = [f"{TARGET_LABELS[target]}"]
        for method_key in ("global_fft", "mfcc_mean", "mfcc_dtw", "stft_mean", "stft_knn", "mlp"):
            pieces.append(f"{method_key}={results[target][method_key]['metrics']['f1']:.4f}")
        lines.append(", ".join(pieces))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    items_by_split = build_selected_items()
    cache = build_feature_cache(items_by_split)

    results: dict[str, dict[str, dict[str, object]]] = {}
    average_spectra = {}

    for target in TARGET_PHONEMES:
        print(f"evaluating target {TARGET_LABELS[target]}")
        datasets = build_balanced_detector_sets(items_by_split, target)
        banks = build_target_banks(target, datasets["train"]["items"], cache)
        average_spectra[target] = banks["average_local_spectrum"]

        results[target] = {
            "metadata": {
                "positive_words": datasets["train"]["positive_words"],
                "train_size": int(len(datasets["train"]["items"])),
                "val_size": int(len(datasets["val"]["items"])),
                "test_size": int(len(datasets["test"]["items"])),
            },
            "global_fft": evaluate_global_fft(datasets, cache, banks["global_fft_template"]),
            "mfcc_mean": evaluate_mean_scan(datasets, cache, banks["mfcc_mean_bank"], "scan_mfcc_mean"),
            "mfcc_dtw": evaluate_mfcc_dtw(datasets, cache, banks["mfcc_selector_bank"], banks["mfcc_sequence_bank"]),
            "stft_mean": evaluate_mean_scan(datasets, cache, banks["spectrum_mean_bank"], "scan_spectra"),
            "stft_knn": evaluate_knn_scan(datasets, cache, banks["spectrum_knn_bank"]),
            "mlp": evaluate_mlp(datasets, cache),
        }

    summary = aggregate_method_results(results)

    plot_average_spectra(average_spectra)
    plot_method_comparison(summary)
    plot_per_letter_f1(results)
    plot_position_examples(cache, items_by_split)

    output = {
        "dataset": {
            "selected_words": SELECTED_WORDS,
            "phoneme_map": WORD_PHONEMES,
            "caps_per_split": CAP_PER_SPLIT,
            "targets": TARGET_LABELS,
        },
        "results": results,
        "summary": summary,
    }
    write_json(RESULTS_DIR / "fullword_results.json", output)
    write_summary(RESULTS_DIR / "summary.txt", summary, results)


if __name__ == "__main__":
    main()
