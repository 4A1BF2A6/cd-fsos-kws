"""
Sliding-window real-time KWS demo.

Three loading modes:
  3. --classifier <ckpt>              (RECOMMENDED for fixed wake words)
     Load a fixed N+1 way classifier trained by train_kws_classifier.py.
     Threshold defaults to thr_far05 saved in the checkpoint.

  1. --adapted_model + --prototypes   (legacy: CKA few-shot)
     Load the best-episode adapted model and prototypes saved by
     target_adapting_querying.py.

  2. --model + --support              (quick test, no adaptation)
     Load the pretrained backbone and build prototypes on the fly from
     a support directory (sub-dirs = class labels, each holding wavs).

Usage (mode 3 鈥?recommended):
    python demo_sliding_kws.py \
        --classifier results/kws_classifier/best.pt \
        --wav <audio.wav> \
        --ema_alpha 0.3 --cooldown_ms 1000
"""

import argparse
import collections
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio

import numpy as np

import models  # noqa: F401 鈥?registers model builders
from models.utils import get_model
from models.CKAs_module import ReprModel_cka
from demo_fewshot_wav import (
    build_model_opts,
    build_prototypes,
    load_backbone,
    load_support_set,
)


def reproj_scores(query, muK, beta, device):
    """Device-agnostic prototype reprojection (mirrors get_reproj_dist in paper).

    query : (1, D)
    muK   : (N, D)  L2-normalised prototypes
    beta  : scalar Parameter
    Returns (N,) scores 鈥?higher = more similar (matches evaluation space).
    """
    support = muK.unsqueeze(1).to(device)          # (N, 1, D)
    lam = support.size(1) / support.size(2)        # 1/D
    rho = beta.to(device).exp()
    st  = support.permute(0, 2, 1)                 # (N, D, 1)
    sst = support.matmul(st)                       # (N, 1, 1)
    sst_plus_ri = sst + torch.eye(sst.size(-1), device=device).unsqueeze(0) * lam
    sst_inv = torch.tensor(
        np.linalg.inv(sst_plus_ri.detach().cpu().numpy()),
        dtype=query.dtype, device=device)
    w     = query.matmul(st.matmul(sst_inv))       # (N, 1, 1)
    Q_bar = w.matmul(support).mul(rho)             # (N, 1, D)
    dist  = (Q_bar - query.unsqueeze(0)).pow(2).sum(2).permute(1, 0).neg()  # (1, N)
    return dist.squeeze(0)                         # (N,)

SR = 16000
WINDOW_S = 1.0


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def load_adapted(adapted_model_path, prototypes_path, device):
    """Load CKA-adapted model + saved prototypes (mode 1)."""
    ckpt = torch.load(prototypes_path, map_location='cpu')
    model_opt = ckpt['model_opt']
    adapting_opt = ckpt['adapting_opt']
    criterion = ckpt['criterion']
    x_dim = ckpt['x_dim']

    # build full opt dict expected by ReprModel_cka
    full_opt = dict(adapting_opt)
    full_opt['data.cuda'] = device.type == 'cuda'

    # reconstruct base model
    base = get_model(model_opt)
    base.eval()
    base.to(device)
    if hasattr(base.preprocessing, 'mfcc'):
        base.preprocessing.mfcc.to(device)

    # wrap with CKA adapters (same architecture as during training)
    model = ReprModel_cka(base, full_opt, criterion, x_dim)
    state = torch.load(adapted_model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    if hasattr(model.preprocessing, 'mfcc'):
        model.preprocessing.mfcc.to(device)
    model.eval()

    muK = ckpt['muK'].to(device)           # (N, D) already L2-normalised
    beta = ckpt.get('beta')                # reprojection parameter
    labels = ckpt['class_list']
    word_to_index = ckpt['word_to_index']

    thr = ckpt.get('thr_far05')
    print('Loaded adapted model  ep={}  acc_far05={:.4f}  thr_far05={}'.format(
        ckpt.get('ep', '?'),
        ckpt.get('acc_far05', float('nan')),
        '{:.4f}'.format(thr) if thr is not None else 'n/a (rerun evaluation to get)'))
    print('Classes:', labels)
    if thr is not None:
        print('Suggested threshold: {:.4f}  (use --threshold {:.4f})'.format(thr, thr))
    return model, muK, beta, labels, word_to_index


def load_pretrained(model_path, support_dir, seconds, device):
    """Load pretrained backbone + build prototypes from support dir (mode 2)."""
    model = load_backbone(model_path, device)
    support = load_support_set(support_dir, seconds)
    labels, prototypes, counts = build_prototypes(model, support, device)
    word_to_index = {l: i for i, l in enumerate(labels)}
    print('Classes:', ', '.join('{} ({})'.format(l, c) for l, c in zip(labels, counts)))
    return model, prototypes, None, labels, word_to_index


def load_classifier(ckpt_path, device):
    """Load a fixed N+1 way KWS classifier (mode 3)."""
    from train_kws_classifier import KWSClassifier
    # weights_only=False：本地受信 ckpt 含 numpy scalar / dict 等非张量字段；
    # torch 2.6+ 默认 weights_only=True 会拦下来。torch 1.13 也支持此 kwarg。
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = KWSClassifier(ckpt['encoder_ckpt'], ckpt['n_classes'],
                          freeze_encoder=ckpt['freeze_encoder'],
                          head=ckpt.get('head', 'linear'),
                          head_hidden=ckpt.get('head_hidden', 128),
                          head_dropout=ckpt.get('head_dropout', 0.2)).to(device)
    if hasattr(model.preprocessing, 'mfcc'):
        model.preprocessing.mfcc.to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    labels = ckpt['class_list']
    word_to_index = ckpt['word_to_index']
    thr = ckpt.get('thr_far05')
    print('Loaded classifier  epoch={}  acc_far05={:.4f}  thr_far05={}'.format(
        ckpt.get('epoch', '?'),
        ckpt.get('acc_far05', float('nan')),
        '{:.4f}'.format(thr) if thr is not None else 'n/a'))
    print('Classes:', labels)
    if thr is not None:
        print('Suggested threshold: {:.4f}  (use --threshold {:.4f})'.format(thr, thr))
    return model, labels, word_to_index, thr


# ---------------------------------------------------------------------------
# Sliding window detector
# ---------------------------------------------------------------------------

class SegmentKWS:
    """Segment-based KWS detector aligned with variable-length training.

    Silero VAD finds speech segments, then each complete segment is
    classified as a single variable-length input.

    Use this for classifier_mode (mode 3). For prototype-based modes
    (1/2), SlidingKWS remains the right choice.
    """

    def __init__(self, model, labels, *, threshold=0.6, hop_ms=20,
                 vad_threshold=0.5, silence_hangover_ms=300,
                 min_speech_ms=200, max_segment_ms=3100,
                 pre_onset_ms=100, cooldown_ms=1000, device,
                 external_vad=False, speech_onset_ms=60, debug=False):
        self.model = model
        self.labels = list(labels)
        self.threshold = threshold
        self.device = device
        self.debug = debug
        self.external_vad = bool(external_vad)
        self.hop_samples = int(SR * hop_ms / 1000)
        self.vad_threshold = vad_threshold
        self.silence_hangover_ms = silence_hangover_ms
        self.silence_hangover_frames = max(1, int(silence_hangover_ms / hop_ms))
        self.speech_onset_frames = max(1, int(speech_onset_ms / hop_ms))
        self.cooldown_frames = max(0, int(cooldown_ms / hop_ms))
        self.min_speech_samples = int(SR * min_speech_ms / 1000)
        self.max_segment_samples = int(SR * max_segment_ms / 1000)
        self.pre_onset_samples = int(SR * pre_onset_ms / 1000)

        if not self.external_vad:
            try:
                from silero_vad import VADIterator, load_silero_vad
            except ImportError as exc:
                raise ImportError(
                    "SegmentKWS requires the 'silero-vad' package. "
                    "Install it with: pip install silero-vad"
                ) from exc

            self._vad_model = load_silero_vad()
            self._vad_iter = VADIterator(
                self._vad_model,
                threshold=self.vad_threshold,
                sampling_rate=SR,
                min_silence_duration_ms=silence_hangover_ms,
                speech_pad_ms=pre_onset_ms,
            )
            self._vad_window_samples = 512
        else:
            # external_vad=True：由调用方传入 is_speech flag，不加载 Silero VAD
            self._vad_model = None
            self._vad_iter = None
            self._vad_window_samples = None

        # rolling pre-onset buffer (always-on, captures leading consonants)
        self._pre_buf = collections.deque(maxlen=self.pre_onset_samples)
        self._vad_buf = []
        self._buf = []           # active segment audio
        self._state = 'IDLE'
        self._cooldown = 0
        self._frame = 0
        self._audio_offset = 0
        # counter-based state machine used when external_vad=True
        self._consec_speech  = 0
        self._consec_silence = 0
        # Both _unknown_ and _silence_ are rejection classes from the deployment's
        # point of view — mask both out before picking the best wake.
        self._reject_idxs = [self.labels.index(l) for l in ('_unknown_', '_silence_')
                             if l in self.labels]
        # exposed for UI compat with SlidingKWS
        self._smooth_probs = None
        # tracks current speech segment start time for debug output
        self._seg_start_frame = None

    @torch.no_grad()
    def push(self, chunk, voice_active=None):
        """chunk: (T,) float32 PCM, typically one hop worth.
        voice_active: 0/1 flag from an upstream VAD; required when
                      external_vad=True, ignored otherwise.
        Returns (label, score) on trigger else (None, None)."""
        chunk_list = chunk.tolist() if torch.is_tensor(chunk) else (
            chunk.tolist() if hasattr(chunk, 'tolist') else list(chunk))
        self._frame += 1
        self._pre_buf.extend(chunk_list)

        if self.external_vad:
            # ── External VAD path: counter-based state machine ──────
            flag = int(bool(voice_active))
            if self._cooldown > 0:
                self._cooldown -= 1
                return None, None

            if self._state == 'IDLE':
                if flag:
                    self._consec_speech += 1
                    if self._consec_speech >= self.speech_onset_frames:
                        self._state = 'IN_SPEECH'
                        self._buf = list(self._pre_buf)
                        self._seg_start_frame = self._frame
                        self._consec_speech = 0
                        self._consec_silence = 0
                        if self.debug:
                            t = self._frame * self.hop_samples / SR
                            print(f'  [speech onset t={t:.2f}s  vad=1]')
                else:
                    self._consec_speech = 0
                return None, None

            # IN_SPEECH
            self._buf.extend(chunk_list)
            if len(self._buf) >= self.max_segment_samples:
                return self._classify_current_segment(ended_by_maxlen=True)

            if not flag:
                self._consec_silence += 1
                if self._consec_silence >= self.silence_hangover_frames:
                    return self._classify_current_segment(ended_by_maxlen=False)
            else:
                self._consec_silence = 0
            return None, None

        # ── Silero VAD path (original) ───────────────────────────────
        self._vad_buf.extend(chunk_list)
        if self._state == 'IN_SPEECH':
            self._buf.extend(chunk_list)
            if len(self._buf) >= self.max_segment_samples:
                return self._classify_current_segment(ended_by_maxlen=True)

        event = None
        while len(self._vad_buf) >= self._vad_window_samples:
            vad_chunk = torch.tensor(
                self._vad_buf[:self._vad_window_samples],
                dtype=torch.float32,
            )
            del self._vad_buf[:self._vad_window_samples]
            event = self._vad_iter(vad_chunk)
            if event is None:
                continue
            if 'start' in event and self._state == 'IDLE':
                self._state = 'IN_SPEECH'
                self._buf = list(self._pre_buf)
                self._seg_start_frame = self._frame
                if self.debug:
                    print(f'  [speech onset sample={event["start"]}]')
            elif 'end' in event and self._state == 'IN_SPEECH':
                break

        if event is not None and 'end' in event and self._state == 'IN_SPEECH':
            if self._cooldown > 0:
                self._cooldown -= 1
                self._reset()
                return None, None
            return self._classify_current_segment(ended_by_maxlen=False)

        if self._cooldown > 0:
            self._cooldown -= 1
        return None, None

    def _classify_current_segment(self, ended_by_maxlen=False):
        seg_len = min(len(self._buf), self.max_segment_samples)
        if ended_by_maxlen:
            speech_len = seg_len
        else:
            tail_silence_samples = int(SR * self.silence_hangover_ms / 1000)
            speech_len = max(self.min_speech_samples, seg_len - tail_silence_samples)

        t_end = self._audio_offset + self._frame * self.hop_samples / SR
        t_start = t_end - seg_len / SR

        if speech_len < self.min_speech_samples:
            if self.debug:
                print(f'  [segment too short t={t_start:.2f}-{t_end:.2f}s '
                      f'len={speech_len/SR:.2f}s discarded]')
            self._reset()
            return None, None

        wav = torch.tensor(self._buf[:seg_len], dtype=torch.float32,
                           device=self.device).unsqueeze(0).unsqueeze(0)
        L = torch.tensor([speech_len], device=self.device)
        logits = self.model(wav, lengths=L).squeeze(0)
        probs = torch.softmax(logits, dim=0)
        self._smooth_probs = probs.detach()

        reject_idxs = self._reject_idxs
        if reject_idxs:
            mask = torch.ones(len(self.labels), dtype=torch.bool, device=self.device)
            for j in reject_idxs:
                mask[j] = False
            wake_probs = probs[mask]
            wake_labels = [l for j, l in enumerate(self.labels) if j not in reject_idxs]
        else:
            wake_probs = probs
            wake_labels = list(self.labels)
        best_i = int(wake_probs.argmax())
        best_label = wake_labels[best_i]
        score = float(wake_probs[best_i])

        if self.debug:
            prob_str = '  '.join(f'{l}:{float(probs[j]):.3f}'
                                  for j, l in enumerate(self.labels))
            print(f'  [segment t={t_start:.2f}-{t_end:.2f}s '
                  f'speech_len={speech_len/SR:.2f}s] [{prob_str}]  '
                  f'wake_best={best_label}:{score:.3f}')

        triggered = score >= self.threshold
        if triggered:
            self._cooldown = self.cooldown_frames
        self._reset()
        return (best_label, score) if triggered else (None, None)

    def _reset(self):
        self._buf = []
        self._state = 'IDLE'
        self._seg_start_frame = None
        self._consec_speech  = 0
        self._consec_silence = 0


class SlidingKWS:
    """Stateful sliding-window keyword detector.

    Call push(chunk) with each new audio chunk (any size, float32 PCM at SR).
    Returns (label, score) when a keyword is confirmed, otherwise (None, None).
    """

    def __init__(self, model, prototypes, labels, *, beta=None, threshold=0.85,
                 hop_ms=20, trigger_frames=3, cooldown_ms=1000, window_s=1.0,
                 ema_alpha=1.0, classifier_mode=False, device, debug=False):
        self.model = model
        self.prototypes = prototypes    # (N, D) L2-normalised, on device (None in mode 3)
        self.beta = beta                # reprojection parameter; None 鈫?raw cosine
        self.labels = labels
        self.threshold = threshold
        self.device = device
        self.hop = int(SR * hop_ms / 1000)
        self.window = int(SR * window_s)
        self._buf = np.zeros(self.window, dtype=np.float32)
        self._write_pos = 0
        self._buf_filled = 0
        self.trigger_frames = trigger_frames
        self.cooldown_frames = int(cooldown_ms / hop_ms)  # frames to suppress after trigger
        self.ema_alpha = ema_alpha       # 1.0 = no smoothing; 0.3 = strong smoothing
        self.classifier_mode = classifier_mode  # True 鈫?model returns logits directly
        self._consec = 0
        self._streak_label = None   # which class is currently building a streak
        self._cooldown = 0   # remaining suppression frames
        self._pending = 0
        self._smooth_probs = None   # EMA-smoothed softmax probabilities
        self.debug = debug
        self._frame = 0
        self._audio_offset = 0   # set by main loop to align debug time with audio
        # Both _unknown_ and _silence_ are rejection classes from the deployment's
        # point of view — mask both out before picking the best wake.
        _labels_list = list(labels)
        self._reject_idxs = [_labels_list.index(l) for l in ('_unknown_', '_silence_')
                             if l in _labels_list]

    @torch.no_grad()
    def push(self, chunk: torch.Tensor):
        """chunk: (T,) float32 PCM.  Returns (label, score) or (None, None)."""
        chunk_np = np.asarray(chunk.tolist(), dtype=np.float32)
        n = len(chunk_np)
        end = self._write_pos + n
        if end <= self.window:
            self._buf[self._write_pos:end] = chunk_np
        else:
            split = self.window - self._write_pos
            self._buf[self._write_pos:] = chunk_np[:split]
            self._buf[:n - split] = chunk_np[split:]
        self._write_pos = (self._write_pos + n) % self.window
        self._buf_filled = min(self._buf_filled + n, self.window)
        self._pending += n

        if self._buf_filled < self.window or self._pending < self.hop:
            return None, None
        self._pending = 0

        ordered = np.roll(self._buf, -self._write_pos)
        wav = torch.tensor(ordered.tolist(), dtype=torch.float32,
                           device=self.device).unsqueeze(0).unsqueeze(0)

        if self.classifier_mode:
            # Mode 3: model is a KWSClassifier 鈫?forward returns logits (1, N).
            # Pass the actual sample length so encoders trained in pad mode get
            # the right mask-aware GAP; on a fixed 1s window this collapses to
            # the legacy global GAP anyway.
            L = torch.tensor([wav.size(-1)], device=self.device)
            logits = self.model(wav, lengths=L).squeeze(0)
            raw = logits
        else:
            # Mode 1/2: prototype-based 鈥?embed + similarity
            emb = F.normalize(self.model.get_embeddings(wav), dim=-1)  # (1, D)
            if self.beta is not None:
                raw = reproj_scores(emb, self.prototypes, self.beta, self.device)
            else:
                raw = (emb @ self.prototypes.T).squeeze(0)

        probs = torch.softmax(raw, dim=0)              # (N,)
        # EMA smoothing
        if self._smooth_probs is None:
            self._smooth_probs = probs.detach().clone()
        else:
            self._smooth_probs = self.ema_alpha * probs + (1 - self.ema_alpha) * self._smooth_probs
        sprobs = self._smooth_probs

        reject_idxs = self._reject_idxs
        if reject_idxs:
            mask = torch.ones(len(self.labels), dtype=torch.bool, device=self.device)
            for j in reject_idxs:
                mask[j] = False
            wake_probs  = sprobs[mask]
            wake_labels = [l for j, l in enumerate(self.labels) if j not in reject_idxs]
        else:
            wake_probs  = sprobs
            wake_labels = list(self.labels)
        best_local = int(wake_probs.argmax())
        best_label = wake_labels[best_local]
        score = float(wake_probs[best_local])

        if self.debug:
            t = self._audio_offset + self._frame * self.hop / SR
            score_str = '  '.join('{}:{:.3f}'.format(l, float(sprobs[i]))
                                  for i, l in enumerate(self.labels))
            print(f'  t={t:.2f}s  [{score_str}]  wake_best={best_label}:{score:.3f}')
        self._frame += 1

        if self._cooldown > 0:
            self._cooldown -= 1
            self._consec = 0
            self._streak_label = None
            return None, None

        if score >= self.threshold:
            if best_label == self._streak_label:
                self._consec += 1
            else:
                # class switched mid-streak 鈫?restart count for the new class
                self._streak_label = best_label
                self._consec = 1
            if self._consec >= self.trigger_frames:
                self._consec = 0
                self._streak_label = None
                self._cooldown = self.cooldown_frames
                return best_label, score
        else:
            self._consec = 0
            self._streak_label = None
        return None, None


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def stream_wav(wav_path, hop_samples):
    wav, file_sr = torchaudio.load(str(wav_path), normalize=True)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if file_sr != SR:
        wav = torchaudio.functional.resample(wav, file_sr, SR)
    wav = wav.squeeze(0)
    for start in range(0, wav.size(0), hop_samples):
        yield wav[start:start + hop_samples]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sliding-window KWS demo")

    # mode 1: adapted model + saved prototypes
    parser.add_argument('--adapted_model', default=None,
                        help='best_adapted_model.pt saved by target_adapting_querying.py')
    parser.add_argument('--prototypes', default=None,
                        help='best_prototypes.pt saved by target_adapting_querying.py')

    # mode 2: pretrained backbone + on-the-fly support set
    parser.add_argument('--model', default=None,
                        help='Pretrained encoder checkpoint (.pt)')
    parser.add_argument('--support', default=None,
                        help='Support-set dir; sub-dirs = class labels')
    parser.add_argument('--seconds', type=float, default=WINDOW_S,
                        help='Support-set clip duration in seconds (mode 2 only)')

    # mode 3: fixed N+1 way classifier (recommended)
    parser.add_argument('--classifier', default=None,
                        help='KWSClassifier checkpoint from train_kws_classifier.py')

    # shared inference args
    parser.add_argument('--wav', required=True,
                        help='WAV file to stream through the detector')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Trigger threshold. In mode 3 defaults to thr_far05 from ckpt.')
    parser.add_argument('--ema_alpha', type=float, default=1.0,
                        help='EMA smoothing on softmax probs (1.0=off, 0.3=strong). Default 1.0')
    parser.add_argument('--hop_ms', type=int, default=20,
                        help='Inference hop in milliseconds (default: 20)')
    parser.add_argument('--trigger_frames', type=int, default=10,
                        help='Consecutive above-threshold frames to trigger (default: 3)')
    parser.add_argument('--cooldown_ms', type=int, default=1000,
                        help='Suppression window after trigger in ms (default: 1000)')
    parser.add_argument('--window_s', type=float, default=1.0,
                        help='Sliding window duration in seconds (default: 1.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Print per-frame scores (useful for threshold tuning)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU even if CUDA is available')

    # detector choice 鈥?classifier mode defaults to segment (matches training)
    parser.add_argument('--detector', choices=['auto', 'segment', 'sliding'],
                        default='auto',
                        help="'auto' (default) 鈫?segment in classifier_mode, "
                             "sliding for prototype modes. Use 'sliding' to "
                             "force the legacy fixed-window detector.")
    # SegmentKWS Silero VAD knobs
    parser.add_argument('--vad_threshold', type=float, default=0.5,
                        help='Silero VAD speech probability threshold '
                             '(default: 0.5; higher is stricter)')
    parser.add_argument('--silence_hangover_ms', type=int, default=200,
                        help='Trailing non-speech duration required before '
                             'closing a segment, in ms (default: 200)')
    parser.add_argument('--min_speech_ms', type=int, default=200,
                        help='Discard detected speech segments shorter than '
                             'this many ms (default: 200)')
    parser.add_argument('--max_segment_ms', type=int, default=3100,
                        help='Force-close and classify a segment after this '
                             'many ms (default: 3100)')
    parser.add_argument('--pre_onset_ms', type=int, default=100,
                        help='Audio padding/backfill kept before Silero speech '
                             'onset, in ms (default: 100)')

    args = parser.parse_args()

    # validate mode (mode 3 takes precedence)
    mode3 = args.classifier
    mode1 = args.adapted_model and args.prototypes
    mode2 = args.model and args.support
    if not mode1 and not mode2 and not mode3:
        parser.error('Provide one of: --classifier <ckpt>  |  '
                     '--adapted_model + --prototypes  |  --model + --support')

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Device: {device}')

    classifier_mode = False
    prototypes = None
    beta = None
    suggested_thr = None
    if mode3:
        print('Mode 3: loading fixed N+1 KWS classifier')
        model, labels, _, suggested_thr = load_classifier(args.classifier, device)
        classifier_mode = True
    elif mode1:
        print('Mode 1: loading CKA-adapted model + saved prototypes')
        model, prototypes, beta, labels, _ = load_adapted(
            args.adapted_model, args.prototypes, device)
    else:
        print('Mode 2: loading pretrained backbone + building prototypes from support set')
        model, prototypes, beta, labels, _ = load_pretrained(
            args.model, args.support, args.seconds, device)

    # Threshold default: mode 3 鈫?thr_far05 from ckpt; others 鈫?0.85 (legacy)
    if args.threshold is None:
        threshold = suggested_thr if suggested_thr is not None else 0.85
        print(f'Using threshold = {threshold:.4f}')
    else:
        threshold = args.threshold

    # Resolve detector type
    use_segment = (args.detector == 'segment' or
                   (args.detector == 'auto' and classifier_mode))
    if use_segment:
        if not classifier_mode:
            parser.error('--detector segment only supports classifier mode '
                         '(--classifier).')
        detector = SegmentKWS(
            model, labels,
            threshold=threshold,
            hop_ms=args.hop_ms,
            vad_threshold=args.vad_threshold,
            silence_hangover_ms=args.silence_hangover_ms,
            min_speech_ms=args.min_speech_ms,
            max_segment_ms=args.max_segment_ms,
            pre_onset_ms=args.pre_onset_ms,
            cooldown_ms=args.cooldown_ms,
            device=device,
            debug=args.debug,
        )
        detector_kind = 'segment'
    else:
        detector = SlidingKWS(
            model, prototypes, labels,
            beta=beta,
            threshold=threshold,
            hop_ms=args.hop_ms,
            trigger_frames=args.trigger_frames,
            cooldown_ms=args.cooldown_ms,
            window_s=args.window_s,
            ema_alpha=args.ema_alpha,
            classifier_mode=classifier_mode,
            device=device,
            debug=args.debug,
        )
        detector_kind = 'sliding'

    hop_samples = int(SR * args.hop_ms / 1000)
    wav_path = Path(args.wav)
    if not wav_path.is_file():
        raise FileNotFoundError(args.wav)

    wav_info = torchaudio.info(str(wav_path))
    total_s = wav_info.num_frames / wav_info.sample_rate
    if detector_kind == 'segment':
        print(f"\nStreaming '{wav_path.name}' ({total_s:.2f}s) | "
              f"detector=segment | hop={args.hop_ms}ms | threshold={threshold:.4f} | "
              f"vad_threshold={args.vad_threshold} | "
              f"silence_hangover={args.silence_hangover_ms}ms | "
              f"cooldown_ms={args.cooldown_ms}\n")
    else:
        print(f"\nStreaming '{wav_path.name}' ({total_s:.2f}s) | "
              f"detector=sliding | hop={args.hop_ms}ms | threshold={threshold:.4f} | "
              f"trigger_frames={args.trigger_frames} | ema_alpha={args.ema_alpha} | "
              f"cooldown_ms={args.cooldown_ms}\n")

    t0 = time.perf_counter()
    n_chunks = 0
    n_triggers = 0
    # tell detector its time offset so debug prints show real audio timestamps;
    # segment detector measures from t=0, sliding from window_s
    detector._audio_offset = 0.0 if detector_kind == 'segment' else args.window_s

    for chunk in stream_wav(wav_path, hop_samples):
        audio_t = n_chunks * args.hop_ms / 1000.0
        label, score = detector.push(chunk)
        if label is not None:
            print(f'  [TRIGGER] t={audio_t:.3f}s  label={label}  score={score:.4f}')
            n_triggers += 1
        n_chunks += 1

    wall = time.perf_counter() - t0
    print(f'\nDone. {n_triggers} trigger(s) in {total_s:.2f}s audio.')
    print(f'Wall time: {wall:.3f}s  RTF = {wall / total_s:.4f}')


if __name__ == '__main__':
    main()
