"""
Train a fixed N+1 KWS classifier on top of the pretrained DSCNN encoder.

Pipeline:
    raw wav (B,1,T) → MFCC → DSCNN encoder (frozen) → Linear(276, n_classes) → softmax

Replaces the few-shot CKA + prototype approach for fixed wake-word deployment.
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# noqa: F401 — registers model builders
import models
from data.CompanyKWS import CompanyKWSDataset, variable_length_collate
from metrics import compute_metrics


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

    def __init__(self, encoder_ckpt, n_classes, freeze_encoder=True):
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

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.head = nn.Linear(DSCNNL_EMB_DIM, n_classes)

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
        'time_shift':       100,
        'include_noise':    True,
        'bg_volume':        0.1,
        'bg_frequency':     1.0,
        'include_silence':  False,
        'num_silence':      0,
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


def make_weighted_sampler(records, word_to_index):
    """Per-class inverse-frequency weighting to balance Camy/Reco/_unknown_."""
    n_classes = len(word_to_index)
    counts = [0] * n_classes
    for r in records:
        counts[word_to_index[r['label']]] += 1
    weights = [1.0 / counts[word_to_index[r['label']]] for r in records]
    print('  class counts:', dict(zip(word_to_index.keys(),
                                       [counts[i] for i in word_to_index.values()])))
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


def get_iid_dataloader_balanced(ds, split, batch_size, num_workers=4, pin_memory=True):
    """Like ds.get_iid_dataloader but with WeightedRandomSampler for balance.

    Uses variable_length_collate so pad-mode datasets pad to batch-max and
    surface a `lengths` tensor. For fixed-length modes it's a no-op pad
    (all samples already same length) plus a `lengths` tensor.
    """
    records = ds.data_set[split]
    transforms_ds = ds.get_transform_dataset(records, list(ds.words_list))
    sampler = make_weighted_sampler(records, ds.word_to_index)
    return DataLoader(transforms_ds, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0,
                      collate_fn=variable_length_collate)


# ---------------------------------------------------------------------------
# Evaluation (mirrors test_model in target_adapting_querying.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(model, loader, n_classes, unk_id, device,
                        force_unk_labels=False, desc='eval'):
    model.eval()
    y_score, y_pred, y_true = [], [], []
    y_pred_close, y_pred_ood = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch['data'].to(device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(device)
        logits = model(x, lengths=lengths)
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


def evaluate(model, ds, split, batch_size, device):
    """Compute compute_metrics() on a given split. Splits into pos (wake) and neg (_unknown_)."""
    word_to_index = ds.word_to_index
    n_classes = len(word_to_index)
    unk_id = word_to_index.get('_unknown_')

    pos_classes = [w for w in ds.words_list if w != '_unknown_']
    neg_classes = ['_unknown_'] if unk_id is not None else []

    pos_records = ds.dataset_filter_class(ds.data_set[split], pos_classes)
    pos_loader = DataLoader(ds.get_transform_dataset(pos_records, pos_classes),
                            batch_size=batch_size, shuffle=False,
                            collate_fn=variable_length_collate)
    y_score_p, y_pred_p, y_true_p, y_close_p, y_ood_p = collect_predictions(
        model, pos_loader, n_classes, unk_id, device, desc=f'{split} pos')

    if neg_classes:
        neg_records = ds.dataset_filter_class(ds.data_set[split], neg_classes)
        if neg_records:
            neg_loader = DataLoader(ds.get_transform_dataset(neg_records, neg_classes),
                                    batch_size=batch_size, shuffle=False,
                                    collate_fn=variable_length_collate)
            y_score_n, y_pred_n, y_true_n, y_close_n, y_ood_n = collect_predictions(
                model, neg_loader, n_classes, unk_id, device,
                force_unk_labels=True, desc=f'{split} neg')
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
                        choices=['center', 'energy', 'stretch', 'pad'],
                        help="'pad' keeps native variable length (recommended). "
                             "Use 'stretch' to reproduce the old fixed-1s training.")
    parser.add_argument('--max_duration_ms', type=int, default=3100,
                        help="Upper bound for crop_strategy='pad' (energy-crop "
                             'longer samples). Default 3100 (CompanyKWS P99).')
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
    parser.add_argument('--task', default='CompanyKWS_ALL')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=256)
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

    print('Loading CompanyKWS dataset …')
    ds = CompanyKWSDataset(args.datadir, args.task, device.type == 'cuda', speech_args)
    word_to_index = ds.word_to_index
    n_classes = len(word_to_index)
    print('Classes:', dict(word_to_index))
    print('Counts train/val/test:',
          len(ds.data_set['training']),
          len(ds.data_set['validation']),
          len(ds.data_set['testing']))

    print('Building model …')
    model = KWSClassifier(args.encoder_ckpt, n_classes,
                          freeze_encoder=not args.unfreeze_encoder).to(device)
    if hasattr(model.preprocessing, 'mfcc'):
        model.preprocessing.mfcc.to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_trainable:,}  (unfreeze_encoder={args.unfreeze_encoder})')
    print(f'Model on: head={model.head.weight.device}  encoder={next(model.encoder.parameters()).device}')

    train_loader = get_iid_dataloader_balanced(ds, 'training', args.batch_size,
                                                num_workers=args.num_workers,
                                                pin_memory=device.type == 'cuda')

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

    best_score = -1.0
    best_metrics = None
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
            logits = model(x, lengths=lengths)
            loss = ce_loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n_seen += x.size(0)

        val_metrics = evaluate(model, ds, 'validation', args.val_batch_size, device)
        score = val_metrics['acc_far05'] if 'acc_far05' in val_metrics else val_metrics['accuracy_pos']
        print(f'[Epoch {epoch+1}] loss={running_loss/max(n_seen,1):.4f}  '
              f'val accuracy_pos={val_metrics["accuracy_pos"]:.4f}  '
              f'accuracy_neg={val_metrics["accuracy_neg"]:.4f}  '
              f'acc_far05={val_metrics["acc_far05"]:.4f}  '
              f'thr_far05={val_metrics["thr_far05"]:.4f}  '
              f'aucROC={val_metrics["aucROC"]:.4f}')

        if score > best_score:
            best_score = score
            best_metrics = val_metrics
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
                'epoch':            epoch,
            }, args.out)
            print(f'  ✓ saved best to {args.out}')
        scheduler.step()

    print('\n=== Best validation metrics ===')
    print(json.dumps({k: float(v) if isinstance(v, (int, float)) else v
                      for k, v in best_metrics.items()}, indent=2))


if __name__ == '__main__':
    main()
