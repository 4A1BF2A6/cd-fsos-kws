"""
Train a fixed N+1 KWS classifier on top of the pretrained DSCNN encoder.

Pipeline:
    raw wav (B,1,T) → MFCC → DSCNN encoder (frozen) → classifier head → softmax

Replaces the few-shot CKA + prototype approach for fixed wake-word deployment.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchaudio
from tqdm import tqdm

# noqa: F401 — registers model builders
import models
from data.CompanyKWS import CompanyKWSDataset, variable_length_collate


DSCNNL_EMB_DIM = 276    # DSCNNL_LAYERNORM output dim


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class KWSClassifier(nn.Module):
    """DSCNN encoder (frozen by default) + Linear classification head.

    Supports both fixed-length (legacy, lengths=None) and variable-length
    (lengths=raw-sample-count) inputs. Variable length routes through
    mask-aware GAP inside the encoder so zero-padded positions don't
    dilute the embedding.
    """

    def __init__(self, encoder_ckpt, n_classes, freeze_encoder=True,
                 head='linear', head_hidden=128, head_dropout=0.2):
        super().__init__()
        repr_model = torch.load(encoder_ckpt, map_location='cpu')
        self.preprocessing = repr_model.preprocessing
        # legacy checkpoints may lack newer attrs added to MFCC
        if hasattr(self.preprocessing, 'mfcc') and not hasattr(self.preprocessing, 'force_cpu'):
            self.preprocessing.force_cpu = False

        # Rebuild encoder against the current (variable-length) DSCNN code,
        # then port the pretrained weights over. The old/new architectures
        # differ only in AvgPool↔AdaptiveAvgPool and LayerNorm↔GroupNorm —
        # neither has learnable params, so state_dict keys & shapes match.
        from models.encoder.DSCNN import DSCNNL_LAYERNORM
        old_encoder = repr_model.encoder
        new_encoder = DSCNNL_LAYERNORM([1, 49, 10])
        new_encoder.load_state_dict(old_encoder.state_dict(), strict=True)
        new_encoder.return_feat_maps = False
        self.encoder = new_encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        if head == 'linear':
            self.head = nn.Linear(DSCNNL_EMB_DIM, n_classes)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(DSCNNL_EMB_DIM, head_hidden),
                nn.LayerNorm(head_hidden),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, n_classes),
            )
        else:
            raise ValueError("head must be 'linear' or 'mlp', got {}".format(head))
        self.head_type = head
        self.head_hidden = head_hidden
        self.head_dropout = head_dropout

        # MFCC win/hop in raw samples — used to convert sample-length → frame-length.
        # Pull from the preprocessing wrapper so we honour whatever the ckpt was
        # configured with (currently 40/20 ms @ 16 kHz).
        prep = self.preprocessing
        self._mfcc_win = int(round(prep.window_size_ms / 1000 * prep.sample_rate))
        self._mfcc_hop = int(round(prep.window_stride_ms / 1000 * prep.sample_rate))

    def _samples_to_frames(self, sample_lengths):
        """raw waveform samples → MFCC frame count (matches center=False)."""
        T = sample_lengths.to(torch.long)
        frames = ((T - self._mfcc_win).clamp(min=0)) // self._mfcc_hop + 1
        return frames.clamp(min=1)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, x, lengths=None):
        """
        x       : (B, 1, T_samples) raw waveform (zero-padded if variable).
        lengths : (B,) int tensor of valid sample counts. None → fixed-length
                  legacy path (encoder does global GAP, identical to old AvgPool).
        """
        feat = self.preprocessing.extract_features(x)   # (B, 1, T_frames, F)
        if lengths is None:
            emb = self.encoder(feat)
        else:
            mfcc_lengths = self._samples_to_frames(lengths)
            emb = self.encoder(feat, lengths=mfcc_lengths)
        return self.head(emb)                            # (B, n_classes)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_speech_args(args):
    """Mimic parser_kws.py defaults so CompanyKWSDataset can ingest."""
    return {
        'sample_rate':      16000,
        'clip_duration':    1000,
        'window_size':      40,
        'window_stride':    20,
        'n_mfcc':           40,
        'num_features':     10,
        'foreground_volume': 1.0,
        'augment_gain_db': args.augment_gain_db,
        'augment_silence_pad_ms': args.augment_silence_pad_ms,
        'time_shift':       100,
        'include_noise':    True,
        'bg_volume':        0.1,
        'bg_frequency':     1.0,
        'include_silence':  False,
        'num_silence':      0,
        'silence_as_unknown': not args.no_silence_unknown,
        'include_unknown':  True,
        'channel':          args.channel,
        'crop_strategy':    args.crop_strategy,
        'max_duration_ms':  args.max_duration_ms,
        'merge_val':        'none',     # keep val standalone for early stopping
        'gsc_unknown_dir':  args.gsc_dir,
        'gsc_unknown_words': args.gsc_unknown_words,
        'gsc_unknown_splits': 'all',    # 80% GSC → training, 20% → testing
        'librispeech_dir':  args.librispeech_dir,
        'librispeech_samples_per_file': args.librispeech_samples_per_file,
        'librispeech_max_files': args.librispeech_max_files,
        'gsc_noise_dir':    args.gsc_noise_dir,
        'gsc_noise_samples_per_file': args.gsc_noise_samples_per_file,
        'bg_duration_min_ms': args.bg_duration_min_ms,
        'bg_duration_max_ms': args.bg_duration_max_ms,
    }


def unknown_source(record):
    """Coarse source bucket for _unknown_ records.

    CompanyKWSDataset stores all rejection data under the same label, but the
    source mix is very uneven. GSC words can dominate the physical dataset, so
    the sampler balances these buckets inside the _unknown_ class.
    """
    if record.get('is_silence', False):
        return 'silence'
    speaker = str(record.get('speaker', ''))
    if speaker == 'gsc_unknown':
        return 'gsc_words'
    if speaker.startswith('librispeech_'):
        return 'librispeech'
    if speaker.startswith('gsc_noise_'):
        return 'gsc_noise'
    if speaker == 'background':
        return 'company_background'
    return 'unknown_other'


def parse_unknown_source_weights(text):
    defaults = {
        'company_background': 0.30,
        'gsc_noise': 0.25,
        'gsc_words': 0.25,
        'silence': 0.10,
        'librispeech': 0.10,
        'unknown_other': 0.10,
    }
    if not text:
        return defaults
    out = {}
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError(
                '--unknown_source_weights entries must be name=value, got {}'.format(item))
        name, value = item.split('=', 1)
        out[name.strip()] = float(value)
    return out


def build_sample_weights(records, word_to_index, unknown_source_weights=None,
                         samples_per_class_per_epoch=0):
    """Balance wake classes and stratify _unknown_ by source bucket."""
    n_classes = len(word_to_index)
    counts = [0] * n_classes
    unknown_label = '_unknown_'
    source_counts = {}
    unknown_source_weights = unknown_source_weights or parse_unknown_source_weights(None)
    for r in records:
        counts[word_to_index[r['label']]] += 1
        if r['label'] == unknown_label:
            src = unknown_source(r)
            source_counts[src] = source_counts.get(src, 0) + 1

    active_source_weights = {
        src: unknown_source_weights.get(src, unknown_source_weights.get('unknown_other', 0.0))
        for src in source_counts
    }
    total_source_weight = sum(w for w in active_source_weights.values() if w > 0)
    if source_counts and total_source_weight <= 0:
        raise ValueError('All active _unknown_ source weights are <= 0: {}'.format(
            active_source_weights))

    weights = []
    for r in records:
        if r['label'] == unknown_label and source_counts:
            src = unknown_source(r)
            src_weight = max(0.0, active_source_weights.get(src, 0.0)) / total_source_weight
            weights.append(src_weight / source_counts[src])
        else:
            weights.append(1.0 / counts[word_to_index[r['label']]])
    if samples_per_class_per_epoch and samples_per_class_per_epoch > 0:
        num_samples = int(samples_per_class_per_epoch) * n_classes
    else:
        num_samples = len(records)

    print('  class counts:', dict(zip(word_to_index.keys(),
                                       [counts[i] for i in word_to_index.values()])))
    if source_counts:
        print('  _unknown_ source counts:', dict(sorted(source_counts.items())))
        print('  _unknown_ source sampling weights:', {
            k: round(max(0.0, v) / total_source_weight, 4)
            for k, v in sorted(active_source_weights.items())
        })
    print('  samples per epoch:', num_samples)
    return weights, num_samples


def make_weighted_sampler(records, word_to_index, unknown_source_weights=None,
                          samples_per_class_per_epoch=0):
    weights, num_samples = build_sample_weights(
        records, word_to_index,
        unknown_source_weights=unknown_source_weights,
        samples_per_class_per_epoch=samples_per_class_per_epoch)
    return WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)


def record_length_samples(record, sample_rate, max_duration_samples):
    if record.get('is_silence', False):
        dur = record.get('silence_duration_seconds')
        if dur is not None:
            return max(1, int(float(dur) * sample_rate))
    dur = record.get('bg_duration_seconds')
    if dur is not None:
        return max(1, int(float(dur) * sample_rate))
    try:
        info = torchaudio.info(record['file'])
        n = int(info.num_frames * sample_rate / info.sample_rate)
    except Exception:
        n = max_duration_samples
    return min(max(1, n), max_duration_samples)


class WeightedLengthBucketBatchSampler:
    """Weighted replacement sampler that groups sampled indices by duration."""

    def __init__(self, weights, num_samples, batch_size, lengths, bucket_samples):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.lengths = list(lengths)
        self.bucket_samples = int(bucket_samples)

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        sampled = torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()
        sampled.sort(key=lambda idx: (
            self.lengths[idx] // self.bucket_samples if self.bucket_samples > 0 else 0,
            self.lengths[idx],
        ))
        batches = []
        for i in range(0, len(sampled), self.batch_size):
            batches.append(sampled[i:i + self.batch_size])

        for i in torch.randperm(len(batches)).tolist():
            yield batches[i]


def get_iid_dataloader_balanced(ds, split, batch_size, num_workers=4, pin_memory=True,
                                unknown_source_weights=None,
                                samples_per_class_per_epoch=0,
                                length_bucket_ms=500):
    """Like ds.get_iid_dataloader but with WeightedRandomSampler for balance.

    Uses variable_length_collate so pad-mode datasets pad to batch-max and
    surface a `lengths` tensor. For fixed-length modes it's a no-op pad
    (all samples already same length) plus a `lengths` tensor.
    """
    records = ds.data_set[split]
    transforms_ds = ds.get_transform_dataset(records, list(ds.words_list), augment=True)
    weights, num_samples = build_sample_weights(
        records, ds.word_to_index,
        unknown_source_weights=unknown_source_weights,
        samples_per_class_per_epoch=samples_per_class_per_epoch)
    if length_bucket_ms and length_bucket_ms > 0:
        bucket_samples = max(1, int(ds.sample_rate * length_bucket_ms / 1000))
        lengths = [record_length_samples(r, ds.sample_rate, ds.max_duration_samples)
                   for r in records]
        batch_sampler = WeightedLengthBucketBatchSampler(
            weights, num_samples, batch_size, lengths, bucket_samples)
        print('  length bucketing: {} ms buckets'.format(length_bucket_ms))
        return DataLoader(transforms_ds, batch_sampler=batch_sampler,
                          num_workers=num_workers, pin_memory=pin_memory,
                          persistent_workers=num_workers > 0,
                          collate_fn=variable_length_collate)
    sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
    return DataLoader(transforms_ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0,
                      collate_fn=variable_length_collate)


# ---------------------------------------------------------------------------
# Forward strategies
# ---------------------------------------------------------------------------

def _window_starts(valid_len, window_samples, hop_samples):
    """Return starts for fixed-size sliding windows over one valid waveform."""
    if valid_len <= window_samples:
        return [0]
    starts = list(range(0, valid_len - window_samples + 1, hop_samples))
    tail_start = valid_len - window_samples
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return starts


def sliding_window_logits(model, x, lengths, window_ms=1000, hop_ms=250,
                          agg='max_logit', window_batch=1024):
    """Run fixed-size windows through the classifier and aggregate per sample.

    x       : (B, 1, T) right-padded waveform batch.
    lengths : (B,) valid raw-sample lengths; only valid audio is windowed.
    """
    if lengths is None:
        lengths = torch.full((x.size(0),), x.size(-1), dtype=torch.long, device=x.device)
    sample_rate = int(model.preprocessing.sample_rate)
    window_samples = max(1, int(round(sample_rate * window_ms / 1000.0)))
    hop_samples = max(1, int(round(sample_rate * hop_ms / 1000.0)))
    window_batch = max(1, int(window_batch))

    windows = []
    ranges = []
    for i in range(x.size(0)):
        valid_len = int(lengths[i].item())
        valid_len = max(1, min(valid_len, x.size(-1)))
        starts = _window_starts(valid_len, window_samples, hop_samples)
        range_start = len(windows)
        for start in starts:
            end = min(start + window_samples, valid_len)
            win = x[i:i + 1, :, start:end]
            if win.size(-1) < window_samples:
                win = F.pad(win, (0, window_samples - win.size(-1)))
            windows.append(win.squeeze(0))
        ranges.append((range_start, len(windows)))

    window_tensor = torch.stack(windows, dim=0)
    logits_chunks = []
    for start in range(0, window_tensor.size(0), window_batch):
        chunk = window_tensor[start:start + window_batch]
        # Fixed 1s windows intentionally use the legacy fixed-length path.
        logits_chunks.append(model(chunk, lengths=None))
    window_logits = torch.cat(logits_chunks, dim=0)

    sample_logits = []
    for start, end in ranges:
        cur = window_logits[start:end]
        if agg == 'max_logit':
            sample_logits.append(cur.max(dim=0).values)
        elif agg == 'logsumexp':
            sample_logits.append(torch.logsumexp(cur, dim=0))
        elif agg == 'mean_logit':
            sample_logits.append(cur.mean(dim=0))
        else:
            raise ValueError(
                "sliding_agg must be 'max_logit', 'logsumexp' or 'mean_logit', got {}".format(agg))
    return torch.stack(sample_logits, dim=0)


def forward_with_strategy(model, x, lengths=None, input_strategy='pad',
                          sliding_window_ms=1000, sliding_hop_ms=250,
                          sliding_agg='max_logit', sliding_window_batch=1024):
    if input_strategy == 'pad':
        return model(x, lengths=lengths)
    if input_strategy == 'sliding':
        return sliding_window_logits(
            model, x, lengths,
            window_ms=sliding_window_ms,
            hop_ms=sliding_hop_ms,
            agg=sliding_agg,
            window_batch=sliding_window_batch)
    raise ValueError("input_strategy must be 'pad' or 'sliding', got {}".format(
        input_strategy))


def input_strategy_from_crop(crop_strategy):
    """Map user-facing crop_strategy to the model forward strategy."""
    return 'sliding' if crop_strategy == 'sliding' else 'pad'


# ---------------------------------------------------------------------------
# Evaluation (mirrors test_model in target_adapting_querying.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, n_classes, unk_id, device,
                        force_unk_labels=False, desc='eval',
                        input_strategy='pad',
                        sliding_window_ms=1000,
                        sliding_hop_ms=250,
                        sliding_agg='max_logit',
                        sliding_window_batch=1024):
    model.eval()
    y_score, y_pred, y_true = [], [], []
    y_pred_close, y_pred_ood = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch['data'].to(device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(device)
        logits = forward_with_strategy(
            model, x, lengths=lengths,
            input_strategy=input_strategy,
            sliding_window_ms=sliding_window_ms,
            sliding_hop_ms=sliding_hop_ms,
            sliding_agg=sliding_agg,
            sliding_window_batch=sliding_window_batch)
        p_y = F.softmax(logits, dim=1).cpu()
        _, pred = p_y.max(1)
        conf = p_y.gather(1, pred.unsqueeze(1)).squeeze(1)

        if unk_id is not None:
            unk_lab = torch.full((p_y.size(0),), unk_id, dtype=torch.long)
            mask = (1 - F.one_hot(unk_lab, n_classes)).bool()
            close = p_y[mask].reshape(p_y.size(0), -1)
            ood = p_y[:, unk_id]
        else:
            close = p_y
            ood = None

        y_score.extend(conf.tolist())
        y_pred.extend(pred.tolist())
        y_pred_close.extend(close.tolist())
        if ood is not None:
            y_pred_ood.extend(ood.tolist())
        if force_unk_labels:
            y_true.extend([unk_id] * x.size(0))
        else:
            y_true.extend(batch['label_idx'].tolist())

    return y_score, y_pred, y_true, y_pred_close, (y_pred_ood if unk_id is not None else None)


def evaluate(model, ds, split, batch_size, device, num_workers=4, pin_memory=True,
             input_strategy='pad',
             sliding_window_ms=1000,
             sliding_hop_ms=250,
             sliding_agg='max_logit',
             sliding_window_batch=1024):
    """Compute compute_metrics() on a given split. Splits into pos (wake) and neg (_unknown_).

    pos/neg loaders parallel-load audio in num_workers processes so validation
    doesn't stall the GPU between epochs. Default 4 is a safe middle ground
    (training uses 8; val/test data is smaller so 4 is usually enough).
    """
    from metrics import compute_metrics

    word_to_index = ds.word_to_index
    n_classes = len(word_to_index)
    unk_id = word_to_index.get('_unknown_')
    split_list = list(split) if isinstance(split, (list, tuple)) else [split]
    split_name = '+'.join(split_list)

    def _records_for(classes):
        out = []
        for split_item in split_list:
            out.extend(ds.dataset_filter_class(ds.data_set[split_item], classes))
        return out

    pos_classes = [w for w in ds.words_list if w != '_unknown_']
    neg_classes = ['_unknown_'] if unk_id is not None else []

    loader_kwargs = dict(
        batch_size=batch_size, shuffle=False,
        collate_fn=variable_length_collate,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == 'cuda',
        persistent_workers=num_workers > 0,
    )

    pos_records = _records_for(pos_classes)
    pos_loader = DataLoader(ds.get_transform_dataset(pos_records, pos_classes),
                            **loader_kwargs)
    y_score_p, y_pred_p, y_true_p, y_close_p, y_ood_p = collect_predictions(
        model, pos_loader, n_classes, unk_id, device, desc=f'{split_name} pos',
        input_strategy=input_strategy,
        sliding_window_ms=sliding_window_ms,
        sliding_hop_ms=sliding_hop_ms,
        sliding_agg=sliding_agg,
        sliding_window_batch=sliding_window_batch)

    if neg_classes:
        neg_records = _records_for(neg_classes)
        if neg_records:
            neg_loader = DataLoader(ds.get_transform_dataset(neg_records, neg_classes),
                                    **loader_kwargs)
            y_score_n, y_pred_n, y_true_n, y_close_n, y_ood_n = collect_predictions(
                model, neg_loader, n_classes, unk_id, device,
                force_unk_labels=True, desc=f'{split_name} neg',
                input_strategy=input_strategy,
                sliding_window_ms=sliding_window_ms,
                sliding_hop_ms=sliding_hop_ms,
                sliding_agg=sliding_agg,
                sliding_window_batch=sliding_window_batch)
        else:
            y_score_n = y_pred_n = y_true_n = y_close_n = y_ood_n = None
    else:
        y_score_n = y_pred_n = y_true_n = y_close_n = y_ood_n = None

    return compute_metrics(
        y_score_p, y_pred_p, y_true_p, y_close_p, y_ood_p,
        y_score_n, y_pred_n, y_true_n, y_close_n, y_ood_n,
        word_to_index, target_far=0.05, verbose=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_ckpt', required=True,
                        help='Pretrained ReprModel checkpoint (best_model.pt)')
    parser.add_argument('--datadir', required=True, help='CompanyKWS root dir')
    parser.add_argument('--gsc_dir', required=True, help='GSC speech_commands_v0.02 root')
    parser.add_argument('--channel', default='ch07')
    parser.add_argument('--crop_strategy', default='pad',
                        choices=['center', 'energy', 'pad', 'sliding'],
                        help="'pad' keeps native variable length (recommended). "
                             "'sliding' keeps native length but trains/evaluates via "
                             "fixed 1s sliding windows. 'center' and 'energy' keep "
                             "fixed-window 1s crop modes.")
    parser.add_argument('--max_duration_ms', type=int, default=3100,
                        help="Upper bound for variable-length crop strategies "
                             'pad/sliding (energy-crop longer samples). '
                             'Default 3100 (CompanyKWS P99).')
    parser.add_argument('--no_silence_unknown', action='store_true',
                        help='Disable adding pure-silence samples to the _unknown_ class.')
    parser.add_argument('--gsc_unknown_words', type=str,
                        default='backward,forward,visual,follow,learn,bed,bird,cat,dog')
    parser.add_argument('--librispeech_dir', default=None,
                        help='LibriSpeech root (e.g. .../train-clean-100/). Adds random '
                             '1s slices of continuous English speech to _unknown_ class')
    parser.add_argument('--librispeech_samples_per_file', type=int, default=2,
                        help='Random 1s slices per utterance (default: 2)')
    parser.add_argument('--librispeech_max_files', type=int, default=0,
                        help='Cap on number of .flac files to use; 0 = no cap (default: 0)')
    parser.add_argument('--gsc_noise_dir', default=None,
                        help='GSC _background_noise_/ dir; if set, sliced into '
                             'variable-length _unknown_ negatives. E.g. '
                             '<gsc_dir>/_background_noise_/')
    parser.add_argument('--gsc_noise_samples_per_file', type=int, default=50,
                        help='Random slices per GSC noise wav (default: 50)')
    parser.add_argument('--bg_duration_min_ms', type=int, default=500,
                        help='Lower bound on random duration for _unknown_ slices '
                             '(default 500ms, matches wake-word P1)')
    parser.add_argument('--bg_duration_max_ms', type=int, default=3100,
                        help='Upper bound on random duration for _unknown_ slices '
                             '(default 3100ms, matches wake-word P99)')
    parser.add_argument('--augment_gain_db', type=float, default=3.0,
                        help='Training-only random waveform gain range in dB. '
                             'Uses uniform(-x, +x). Set 0 to disable. Default: 3.')
    parser.add_argument('--augment_silence_pad_ms', type=int, default=100,
                        help='Training-only random leading/trailing zero padding for '
                             'wake-word positives. Set 0 to disable. Default: 100ms.')
    parser.add_argument('--head', choices=['linear', 'mlp'], default='linear',
                        help='Classifier head capacity. Default: linear.')
    parser.add_argument('--head_hidden', type=int, default=64,
                        help='Hidden units for --head mlp. Default: 128.')
    parser.add_argument('--head_dropout', type=float, default=0.4,
                        help='Dropout for --head mlp. Default: 0.3.')
    parser.add_argument('--unknown_source_weights', type=str,
                        default='company_background=0.15,gsc_noise=0.20,gsc_words=0.30,silence=0.10,librispeech=0.25',
                        help='Comma-separated sampling weights inside _unknown_. '
                             'Active sources are renormalized. Default: '
                             'company_background=0.15,gsc_noise=0.20,'
                             'gsc_words=0.30,silence=0.10,librispeech=0.25')
    parser.add_argument('--samples_per_class_per_epoch', type=int, default=0,
                        help='If >0, each epoch samples this many examples per '
                             'top-level class according to the weighted sampler. '
                             'For 3 classes and 800, epoch length is 2400. '
                             'Default 0 keeps legacy len(dataset) epoch length.')
    parser.add_argument('--length_bucket_ms', type=int, default=500,
                        help='Group weighted training samples into duration buckets '
                             'before batching to reduce right-padding. Set 0 to '
                             'disable. Default: 500ms.')
    parser.add_argument('--sliding_window_ms', type=int, default=1000,
                        help="Window size when --crop_strategy sliding. Default: 1000ms.")
    parser.add_argument('--sliding_hop_ms', type=int, default=250,
                        help="Hop size when --crop_strategy sliding. Default: 250ms.")
    parser.add_argument('--sliding_agg', choices=['max_logit', 'logsumexp', 'mean_logit'],
                        default='max_logit',
                        help='How to aggregate window logits into one sample logit vector. '
                             'Default: max_logit.')
    parser.add_argument('--sliding_window_batch', type=int, default=1024,
                        help='Max fixed windows per classifier forward under sliding strategy. '
                             'Lower this if GPU memory spikes. Default: 1024.')
    parser.add_argument('--task', default='CompanyKWS_ALL')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Stop if validation selection metric does not improve '
                             'for this many epochs. Set 0 to disable. Default: 10.')
    parser.add_argument('--early_stop_min_epochs', type=int, default=0,
                        help='Do not early-stop before this epoch count. Default: 0.')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--val_include_test', action='store_true',
                        help='During training only, evaluate validation metrics on '
                             'validation+testing combined for model selection. '
                             'Training samples are unchanged. Default: off.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_encoder', type=float, default=1e-4,
                        help='LR for encoder when unfrozen')
    parser.add_argument('--unfreeze_encoder', action='store_true')
    parser.add_argument('--out', required=True, help='Output checkpoint path')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader workers; bump up if GPU is starving (default: 8)')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('Device:', device)
    if device.type == 'cuda':
        print('  CUDA device:', torch.cuda.get_device_name(0))

    speech_args = build_speech_args(args)
    speech_args['gsc_unknown_words'] = args.gsc_unknown_words
    input_strategy = input_strategy_from_crop(args.crop_strategy)

    print('Loading CompanyKWS dataset …')
    ds = CompanyKWSDataset(args.datadir, args.task, device.type == 'cuda', speech_args)
    word_to_index = ds.word_to_index
    n_classes = len(word_to_index)
    val_split = ('validation', 'testing') if args.val_include_test else 'validation'
    val_split_name = '+'.join(val_split) if isinstance(val_split, tuple) else val_split
    val_selection_count = sum(
        len(ds.data_set[s]) for s in (val_split if isinstance(val_split, tuple) else (val_split,)))
    print('Classes:', dict(word_to_index))
    print('Counts train/val/test:',
          len(ds.data_set['training']),
          len(ds.data_set['validation']),
          len(ds.data_set['testing']))
    print('Validation for model selection: {} ({} samples)'.format(
        val_split_name, val_selection_count))

    print('Building model …')
    model = KWSClassifier(args.encoder_ckpt, n_classes,
                          freeze_encoder=not args.unfreeze_encoder,
                          head=args.head,
                          head_hidden=args.head_hidden,
                          head_dropout=args.head_dropout).to(device)
    if hasattr(model.preprocessing, 'mfcc'):
        model.preprocessing.mfcc.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_trainable:,}  (unfreeze_encoder={args.unfreeze_encoder})')
    if args.head == 'mlp':
        print(f'Head: {args.head}  hidden={args.head_hidden}  dropout={args.head_dropout}')
    else:
        print(f'Head: {args.head}')
    print('Crop/Input: {} -> {}  window={}ms  hop={}ms  agg={}  window_batch={}'.format(
        args.crop_strategy, input_strategy, args.sliding_window_ms, args.sliding_hop_ms,
        args.sliding_agg, args.sliding_window_batch))
    print('Augment: gain_db=±{:.3g}  silence_pad={}ms'.format(
        args.augment_gain_db, args.augment_silence_pad_ms))
    print(f'Model on: head={next(model.head.parameters()).device}  encoder={next(model.encoder.parameters()).device}')

    train_loader = get_iid_dataloader_balanced(ds, 'training', args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=device.type == 'cuda',
                                                unknown_source_weights=parse_unknown_source_weights(
                                                    args.unknown_source_weights),
                                                samples_per_class_per_epoch=(
                                                    args.samples_per_class_per_epoch),
                                                length_bucket_ms=args.length_bucket_ms)

    if args.unfreeze_encoder:
        params = [
            {'params': model.head.parameters(), 'lr': args.lr},
            {'params': model.encoder.parameters(), 'lr': args.lr_encoder},
        ]
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    ce_loss = nn.CrossEntropyLoss()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)

    def _selection_key(m):
        """Lexicographic preference for best-epoch selection:
            1) higher acc_far05
            2) higher aucROC
            3) higher accuracy_neg
        accuracy_pos isn't needed as a tiebreaker — it's already baked into
        acc_far05 (which is (TPR_at_FAR05 + TNR_at_FAR05) / 2).
        """
        return (
            m.get('acc_far05', m.get('accuracy_pos', 0.0)),
            m.get('aucROC', 0.0),
            m.get('accuracy_neg', 0.0),
        )

    best_key = (-1.0, -1.0, -1.0)
    best_metrics = None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_seen = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}'):
            x = batch['data'].to(device, non_blocking=True)
            y = batch['label_idx'].to(device, non_blocking=True)
            lengths = batch.get('lengths')
            if lengths is not None:
                lengths = lengths.to(device, non_blocking=True)
            logits = forward_with_strategy(
                model, x, lengths=lengths,
                input_strategy=input_strategy,
                sliding_window_ms=args.sliding_window_ms,
                sliding_hop_ms=args.sliding_hop_ms,
                sliding_agg=args.sliding_agg,
                sliding_window_batch=args.sliding_window_batch)
            loss = ce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        val_metrics = evaluate(model, ds, val_split, args.val_batch_size, device,
                                num_workers=args.num_workers,
                                pin_memory=device.type == 'cuda',
                                input_strategy=input_strategy,
                                sliding_window_ms=args.sliding_window_ms,
                                sliding_hop_ms=args.sliding_hop_ms,
                                sliding_agg=args.sliding_agg,
                                sliding_window_batch=args.sliding_window_batch)
        key = _selection_key(val_metrics)
        print(f'[Epoch {epoch+1}] loss={running_loss/max(n_seen,1):.4f}  '
              f'val accuracy_pos={val_metrics["accuracy_pos"]:.4f}  '
              f'accuracy_neg={val_metrics["accuracy_neg"]:.4f}  '
              f'acc_far05={val_metrics["acc_far05"]:.4f}  '
              f'thr_far05={val_metrics["thr_far05"]:.4f}  '
              f'aucROC={val_metrics["aucROC"]:.4f}')

        if key > best_key:
            best_key = key
            best_metrics = val_metrics
            epochs_without_improvement = 0
            torch.save({
                'state_dict':       model.state_dict(),
                'word_to_index':    word_to_index,
                'class_list':       list(ds.words_list),
                'thr_far05':        val_metrics['thr_far05'],
                'acc_far05':        val_metrics['acc_far05'],
                'aucROC':           val_metrics['aucROC'],
                'speech_args':      speech_args,
                'data_dir':         args.datadir,
                'task':             args.task,
                'encoder_ckpt':     args.encoder_ckpt,
                'n_classes':        n_classes,
                'freeze_encoder':   not args.unfreeze_encoder,
                'emb_dim':          DSCNNL_EMB_DIM,
                'head':             args.head,
                'head_hidden':      args.head_hidden,
                'head_dropout':     args.head_dropout,
                'unknown_source_weights': args.unknown_source_weights,
                'samples_per_class_per_epoch': args.samples_per_class_per_epoch,
                'length_bucket_ms': args.length_bucket_ms,
                'sliding_window_ms': args.sliding_window_ms,
                'sliding_hop_ms': args.sliding_hop_ms,
                'sliding_agg': args.sliding_agg,
                'sliding_window_batch': args.sliding_window_batch,
                'val_include_test': args.val_include_test,
                'validation_split': val_split_name,
                'epoch':            epoch,
            }, args.out)
            print(f'  ✓ saved best to {args.out}')
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience > 0:
                print('  no improvement for {}/{} epoch(s)'.format(
                    epochs_without_improvement, args.early_stop_patience))
        scheduler.step()

        reached_min_epochs = (epoch + 1) >= args.early_stop_min_epochs
        if (args.early_stop_patience > 0 and reached_min_epochs and
                epochs_without_improvement >= args.early_stop_patience):
            print('Early stopping at epoch {}: best validation key={} with metrics:'.format(
                epoch + 1, best_key))
            break

    print('\n=== Best validation metrics ===')
    print(json.dumps({k: float(v) if isinstance(v, (int, float)) else v
                      for k, v in best_metrics.items()}, indent=2))


if __name__ == '__main__':
    main()
