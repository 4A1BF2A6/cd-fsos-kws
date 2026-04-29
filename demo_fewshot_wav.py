import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

import models  # noqa: F401 - registers model builders
from models.utils import get_model


SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".opus", ".mp3"}


def build_model_opts():
    return {
        "model_name": "repr_conv",
        "x_dim": [1, 49, 10],
        "hid_dim": 64,
        "z_dim": 64,
        "encoding": "DSCNNL_LAYERNORM",
        "z_norm": True,
        "preprocessing": "mfcc",
        "mfcc": {
            "window_size_ms": 40,
            "window_stride_ms": 20,
            "sample_rate": 16000,
            "n_mfcc": 40,
            "feature_bin_count": 10,
        },
        "loss": {
            "type": "triplet",
            "margin": 0.5,
        },
    }


def load_backbone(checkpoint_path, device):
    model = get_model(build_model_opts())
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.encoder.load_state_dict(checkpoint.encoder.state_dict(), strict=True)
    model.to(device)
    if hasattr(model.preprocessing, "mfcc"):
        model.preprocessing.mfcc.to(device)
    model.eval()
    return model


def normalize_audio(wav, sample_rate, target_rate=16000, seconds=1.0):
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sample_rate != target_rate:
        wav = torchaudio.functional.resample(wav, sample_rate, target_rate)

    target_samples = int(target_rate * seconds)
    if wav.size(1) < target_samples:
        wav = F.pad(wav, (0, target_samples - wav.size(1)))
    elif wav.size(1) > target_samples:
        # Keep the center crop so a manually recorded keyword with small leading
        # or trailing silence still has a good chance of staying in the window.
        start = max((wav.size(1) - target_samples) // 2, 0)
        wav = wav[:, start : start + target_samples]
    return wav


def read_audio(path, seconds):
    wav, sample_rate = torchaudio.load(str(path), normalize=True)
    return normalize_audio(wav, sample_rate, seconds=seconds)


def iter_audio_files(path):
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
        return

    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield item


def load_support_set(support_dir, seconds):
    support_dir = Path(support_dir)
    class_dirs = [p for p in sorted(support_dir.iterdir()) if p.is_dir()]
    if not class_dirs:
        raise ValueError(f"No class folders found under {support_dir}")

    support = {}
    for class_dir in class_dirs:
        files = list(iter_audio_files(class_dir))
        if not files:
            continue
        support[class_dir.name] = torch.stack([read_audio(path, seconds) for path in files])

    if not support:
        raise ValueError(f"No audio files found under {support_dir}")
    return support


@torch.no_grad()
def embed_batch(model, wav_batch, device):
    wav_batch = wav_batch.to(device)
    embeddings = model.get_embeddings(wav_batch)
    return F.normalize(embeddings, p=2, dim=-1).cpu()


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def format_flops(flops):
    if flops >= 1e9:
        return f"{flops / 1e9:.3f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.3f} MFLOPs"
    if flops >= 1e3:
        return f"{flops / 1e3:.3f} KFLOPs"
    return f"{flops:.0f} FLOPs"


@torch.no_grad()
def estimate_forward_flops(model, device, seconds):
    dummy = torch.zeros(1, 1, int(16000 * seconds), device=device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    try:
        _ = model.get_embeddings(dummy)
        sync_device(device)
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            _ = model.get_embeddings(dummy)
            sync_device(device)
        return sum(event.flops or 0 for event in prof.key_averages())
    except Exception as exc:
        print(f"FLOPs estimate unavailable: {exc}")
        return None


@torch.no_grad()
def timed_embed_one(model, wav, device, seconds):
    sync_device(device)
    start = time.perf_counter()
    embedding = embed_batch(model, wav, device).squeeze(0)
    sync_device(device)
    elapsed = time.perf_counter() - start
    return embedding, elapsed, elapsed / seconds


def build_prototypes(model, support, device):
    labels = []
    prototypes = []
    counts = []
    for label, wav_batch in support.items():
        embeddings = embed_batch(model, wav_batch, device)
        labels.append(label)
        prototypes.append(embeddings.mean(dim=0))
        counts.append(wav_batch.size(0))

    prototypes = F.normalize(torch.stack(prototypes), p=2, dim=-1)
    return labels, prototypes, counts


def classify_embedding(embedding, labels, prototypes):
    distances = torch.cdist(embedding.unsqueeze(0), prototypes).squeeze(0)
    scores = -distances.pow(2)
    probabilities = torch.softmax(scores, dim=0)
    ranked = sorted(
        [
            {
                "label": labels[i],
                "probability": float(probabilities[i].item()),
                "distance": float(distances[i].item()),
            }
            for i in range(len(labels))
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )
    return ranked[0], ranked


def main():
    parser = argparse.ArgumentParser(
        description="Few-shot KWS demo with folder-based support wav files."
    )
    parser.add_argument("--support_dir", required=True, help="Folder with one subfolder per class.")
    parser.add_argument("--query", required=True, help="Query wav file or folder.")
    parser.add_argument(
        "--model_path",
        default="results/Pretrain_DSCNN_MSWC/best_model.pt",
        help="Pretrained Adapt-KWS model path.",
    )
    parser.add_argument("--seconds", type=float, default=1.0, help="Audio window length.")
    parser.add_argument("--threshold", type=float, default=None, help="Optional probability threshold.")
    parser.add_argument("--cuda", action="store_true", help="Run on CUDA.")
    parser.add_argument("--topk", type=int, default=3, help="Number of classes to print per query.")
    parser.add_argument("--skip_flops", action="store_true", help="Skip PyTorch profiler FLOPs estimate.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    model = load_backbone(args.model_path, device)
    if not args.skip_flops:
        flops = estimate_forward_flops(model, device, args.seconds)
        if flops is not None and flops > 0:
            print(
                f"Approx forward FLOPs per {args.seconds:.2f}s query: {format_flops(flops)} "
                "(PyTorch profiler estimate)"
            )
        elif flops == 0:
            print("Approx forward FLOPs per query: unavailable or not reported by PyTorch profiler")
    support = load_support_set(args.support_dir, args.seconds)
    labels, prototypes, counts = build_prototypes(model, support, device)

    print("Registered classes:")
    for label, count in zip(labels, counts):
        print(f"  {label}: {count} support wav(s)")
    print()

    query_files = list(iter_audio_files(args.query))
    if not query_files:
        raise ValueError(f"No query audio files found at {args.query}")

    for query_path in query_files:
        wav = read_audio(query_path, args.seconds).unsqueeze(0)
        embedding, elapsed, rtf = timed_embed_one(model, wav, device, args.seconds)
        best, ranked = classify_embedding(embedding, labels, prototypes)
        decision = best["label"]
        if args.threshold is not None and best["probability"] < args.threshold:
            decision = "_reject_"

        print(f"{query_path}")
        print(
            f"  decision={decision} best={best['label']} "
            f"prob={best['probability']:.4f} dist={best['distance']:.4f}"
        )
        print(f"  inference_time={elapsed * 1000:.2f} ms rtf={rtf:.4f}")
        for item in ranked[: args.topk]:
            print(f"    {item['label']}: prob={item['probability']:.4f} dist={item['distance']:.4f}")
        print()


if __name__ == "__main__":
    main()
