"""
Real-time microphone KWS demo with matplotlib (TkAgg) visualization.

Streams audio from a microphone into the same SlidingKWS detector used by
demo_sliding_kws.py, and shows a live UI:
  - rolling waveform (last 5 s)
  - per-class softmax probability history + threshold line + trigger marks
  - current frame probability bars
  - recent trigger log
  - sliders for threshold / EMA alpha / trigger_frames (mutable at runtime)

Every session is dumped to <output_dir>/live_kws_<timestamp>/:
  - recording.wav   16 kHz mono int16, the raw microphone stream
  - triggers.csv    one row per fired trigger: time_s, label, score

Usage:
    # list input devices, then pick one
    python demo_live_kws.py --list_devices

    python demo_live_kws.py \
        --classifier results/kws_classifier/best_v3_frozen.pt \
        --device 1

Dependencies (Windows):  pip install sounddevice
"""

import argparse
import csv
import queue
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    import sounddevice as sd
except ImportError as e:
    print("sounddevice is required: pip install sounddevice", file=sys.stderr)
    raise

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider

from demo_sliding_kws import SR, SlidingKWS, load_classifier


WINDOW_VIEW_S = 5.0
WAVE_DOWNSAMPLE = 16  # display waveform at 1 kHz (16 kHz / 16)
UI_INTERVAL_MS = 100  # matplotlib redraw period


def list_input_devices():
    print(sd.query_devices())
    print('\nUse --device <index> to select one of the input-capable rows above.')


def make_run_dir(output_dir):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f'live_kws_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class LiveKWS:
    def __init__(self, detector, labels, device_idx, hop_ms, run_dir):
        self.detector = detector
        self.labels = list(labels)
        self.n_classes = len(self.labels)
        self.device_idx = device_idx
        self.hop_ms = hop_ms
        self.hop_samples = int(SR * hop_ms / 1000)
        self.run_dir = run_dir

        self.audio_q = queue.Queue()
        self.state_q = queue.Queue()
        self.stop_flag = threading.Event()

        # display buffers
        n_wave = int(WINDOW_VIEW_S * SR / WAVE_DOWNSAMPLE)
        self.wave_buf = deque([0.0] * n_wave, maxlen=n_wave)

        n_prob = int(WINDOW_VIEW_S * 1000 / hop_ms)
        self.prob_hist = [deque([1.0 / self.n_classes] * n_prob, maxlen=n_prob)
                          for _ in self.labels]
        self.time_hist = deque([0.0] * n_prob, maxlen=n_prob)

        # full trigger history (for plotting old marks); also written to csv
        self.trigger_log = []

        # output writers — opened in run()
        self.wav_writer = None
        self.csv_file = None
        self.csv_writer = None

        self._t_start = None

    # ------------------------------------------------------------------ audio
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print('Audio status:', status, file=sys.stderr)
        self.audio_q.put(indata[:, 0].copy())

    def inference_loop(self):
        while not self.stop_flag.is_set():
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            # persist raw audio
            if self.wav_writer is not None:
                ints = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16)
                self.wav_writer.writeframes(ints.tobytes())

            label, score = self.detector.push(torch.from_numpy(chunk))

            sprobs = self.detector._smooth_probs
            if sprobs is None:
                probs = np.full(self.n_classes, 1.0 / self.n_classes, dtype=np.float32)
            else:
                probs = sprobs.detach().cpu().numpy().astype(np.float32)

            now = time.perf_counter() - self._t_start
            self.state_q.put({
                't': now,
                'chunk': chunk,
                'probs': probs,
                'trigger': (label, float(score)) if label is not None else None,
            })

    # ------------------------------------------------------------------- ui
    def drain_state(self):
        while True:
            try:
                ev = self.state_q.get_nowait()
            except queue.Empty:
                break

            self.wave_buf.extend(ev['chunk'][::WAVE_DOWNSAMPLE].tolist())
            for i in range(self.n_classes):
                self.prob_hist[i].append(float(ev['probs'][i]))
            self.time_hist.append(ev['t'])

            if ev['trigger'] is not None:
                label, score = ev['trigger']
                self.trigger_log.append((ev['t'], label, score))
                if self.csv_writer is not None:
                    self.csv_writer.writerow([f'{ev["t"]:.3f}', label, f'{score:.4f}'])
                    self.csv_file.flush()
                print(f'  [TRIGGER] t={ev["t"]:.3f}s  label={label}  score={score:.4f}')

    def _build_ui(self):
        fig = plt.figure(figsize=(13, 9))
        gs = GridSpec(6, 2, figure=fig,
                      height_ratios=[1.0, 1.4, 1.2, 0.25, 0.25, 0.25],
                      hspace=0.55, wspace=0.25)

        # ---- waveform
        ax_wave = fig.add_subplot(gs[0, :])
        ax_wave.set_title('Microphone waveform (last 5 s)')
        ax_wave.set_ylim(-1, 1)
        ax_wave.set_xlim(-WINDOW_VIEW_S, 0)
        ax_wave.set_xticks([])
        wave_x = np.linspace(-WINDOW_VIEW_S, 0, len(self.wave_buf))
        wave_line, = ax_wave.plot(wave_x, list(self.wave_buf), lw=0.5, color='#444')

        # ---- probability history
        ax_prob = fig.add_subplot(gs[1, :])
        ax_prob.set_title('Smoothed softmax probability')
        ax_prob.set_ylim(0, 1.05)
        ax_prob.set_xlim(-WINDOW_VIEW_S, 0)
        ax_prob.set_xlabel('time (s, relative to now)')

        palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        colors = []
        pi = 0
        for l in self.labels:
            if l == '_unknown_':
                colors.append('#888888')
            else:
                colors.append(palette[pi % len(palette)])
                pi += 1

        prob_lines = []
        for i, l in enumerate(self.labels):
            line, = ax_prob.plot([], [], label=l, lw=1.6, color=colors[i])
            prob_lines.append(line)

        thr_line = ax_prob.axhline(self.detector.threshold, ls='--', color='red',
                                   lw=1.0, alpha=0.7,
                                   label=f'thr={self.detector.threshold:.3f}')
        legend = ax_prob.legend(loc='upper left', fontsize=8, ncol=self.n_classes + 1)
        trig_scatter = ax_prob.scatter([], [], marker='v', color='red', s=90,
                                       zorder=5, edgecolors='black', linewidths=0.5)

        # ---- current bars
        ax_bar = fig.add_subplot(gs[2, 0])
        ax_bar.set_title('Current frame probabilities')
        ax_bar.set_ylim(0, 1)
        bars = ax_bar.bar(self.labels, [0] * self.n_classes, color=colors)
        bar_thr_line = ax_bar.axhline(self.detector.threshold, ls='--',
                                       color='red', lw=1.0, alpha=0.7)
        for tick in ax_bar.get_xticklabels():
            tick.set_rotation(15)

        # ---- trigger log text
        ax_log = fig.add_subplot(gs[2, 1])
        ax_log.set_title('Recent triggers')
        ax_log.axis('off')
        log_txt = ax_log.text(0.0, 1.0, '(none yet)', va='top', ha='left',
                              family='monospace', fontsize=10)

        # ---- sliders (3 rows, each spans both columns)
        ax_thr = fig.add_subplot(gs[3, :])
        ax_ema = fig.add_subplot(gs[4, :])
        ax_trg = fig.add_subplot(gs[5, :])
        slider_thr = Slider(ax_thr, 'threshold', 0.0, 1.0,
                            valinit=self.detector.threshold, valstep=0.005)
        slider_ema = Slider(ax_ema, 'ema_alpha', 0.05, 1.0,
                            valinit=self.detector.ema_alpha, valstep=0.05)
        slider_trg = Slider(ax_trg, 'trigger_frames', 1, 10,
                            valinit=self.detector.trigger_frames, valstep=1)

        def on_thr(v):
            self.detector.threshold = float(v)
            thr_line.set_ydata([v, v])
            bar_thr_line.set_ydata([v, v])
            # rebuild legend label so the threshold value stays visible
            thr_line.set_label(f'thr={v:.3f}')
            ax_prob.legend(loc='upper left', fontsize=8, ncol=self.n_classes + 1)

        def on_ema(v):
            self.detector.ema_alpha = float(v)

        def on_trg(v):
            self.detector.trigger_frames = int(v)

        slider_thr.on_changed(on_thr)
        slider_ema.on_changed(on_ema)
        slider_trg.on_changed(on_trg)

        # ---- animation
        def update(_):
            self.drain_state()

            wave_line.set_ydata(list(self.wave_buf))

            t_now = self.time_hist[-1] if self.time_hist else 0.0
            xs = np.array(self.time_hist) - t_now
            for i, line in enumerate(prob_lines):
                line.set_data(xs, list(self.prob_hist[i]))

            for i, b in enumerate(bars):
                b.set_height(self.prob_hist[i][-1])

            if self.trigger_log:
                tail = self.trigger_log[-10:]
                log_txt.set_text('\n'.join(
                    f'{t:7.2f}s  {label:<12}  {score:.3f}'
                    for t, label, score in tail))
            # trigger marks on prob plot, only those inside the visible window
            trig_xy = []
            for t, _, score in self.trigger_log:
                dx = t - t_now
                if -WINDOW_VIEW_S <= dx <= 0:
                    trig_xy.append((dx, min(score + 0.04, 1.02)))
            if trig_xy:
                trig_scatter.set_offsets(np.asarray(trig_xy))
            else:
                trig_scatter.set_offsets(np.empty((0, 2)))

            return (wave_line, *prob_lines, *bars, log_txt, trig_scatter)

        self._anim = FuncAnimation(fig, update, interval=UI_INTERVAL_MS,
                                   blit=False, cache_frame_data=False)
        # keep slider refs alive
        self._sliders = (slider_thr, slider_ema, slider_trg)
        self.fig = fig

    # ----------------------------------------------------------------- main
    def run(self):
        wav_path = self.run_dir / 'recording.wav'
        csv_path = self.run_dir / 'triggers.csv'
        self.wav_writer = wave.open(str(wav_path), 'wb')
        self.wav_writer.setnchannels(1)
        self.wav_writer.setsampwidth(2)
        self.wav_writer.setframerate(SR)
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['time_s', 'label', 'score'])

        print(f'Recording → {wav_path}')
        print(f'Triggers  → {csv_path}')

        infer_thread = threading.Thread(target=self.inference_loop, daemon=True)
        infer_thread.start()

        self._t_start = time.perf_counter()
        stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype='float32',
            blocksize=self.hop_samples,
            device=self.device_idx,
            callback=self.audio_callback,
        )
        stream.start()

        self._build_ui()
        try:
            plt.show()
        finally:
            self.stop_flag.set()
            stream.stop()
            stream.close()
            infer_thread.join(timeout=1.0)
            self.wav_writer.close()
            self.csv_file.close()
            print(f'\nSession saved to {self.run_dir}')


def main():
    parser = argparse.ArgumentParser(description='Live microphone KWS demo')
    parser.add_argument('--list_devices', action='store_true',
                        help='Print available audio devices and exit')
    parser.add_argument('--classifier',
                        help='KWSClassifier checkpoint from train_kws_classifier.py')
    parser.add_argument('--device', type=int, default=None,
                        help='sounddevice input device index (default: system default)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Initial trigger threshold (default: thr_far05 from ckpt)')
    parser.add_argument('--ema_alpha', type=float, default=0.3)
    parser.add_argument('--hop_ms', type=int, default=20)
    parser.add_argument('--trigger_frames', type=int, default=3)
    parser.add_argument('--cooldown_ms', type=int, default=1000)
    parser.add_argument('--window_s', type=float, default=1.0)
    parser.add_argument('--output_dir', default='live_kws_logs')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return
    if not args.classifier:
        parser.error('--classifier is required (or use --list_devices)')

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Torch device: {device}')

    model, labels, _, suggested_thr = load_classifier(args.classifier, device)
    threshold = args.threshold if args.threshold is not None else (
        suggested_thr if suggested_thr is not None else 0.5)

    run_dir = make_run_dir(args.output_dir)

    detector = SlidingKWS(
        model, None, labels,
        beta=None,
        threshold=threshold,
        hop_ms=args.hop_ms,
        trigger_frames=args.trigger_frames,
        cooldown_ms=args.cooldown_ms,
        window_s=args.window_s,
        ema_alpha=args.ema_alpha,
        classifier_mode=True,
        device=device,
        debug=False,
    )

    print(f'Classes        : {labels}')
    print(f'Threshold      : {threshold:.4f}')
    print(f'EMA alpha      : {args.ema_alpha}')
    print(f'Trigger frames : {args.trigger_frames}')
    print(f'Cooldown ms    : {args.cooldown_ms}')
    print(f'Mic device idx : {args.device if args.device is not None else "(default)"}\n')

    app = LiveKWS(detector, labels, args.device, args.hop_ms, run_dir)
    app.run()


if __name__ == '__main__':
    main()
