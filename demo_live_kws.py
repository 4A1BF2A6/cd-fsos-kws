"""
Real-time microphone KWS demo with matplotlib (TkAgg) visualization.

Uses SlidingKWS by default: all microphone audio is continuously fed into a
fixed rolling window and classified every hop. SegmentKWS is still available
via --detector segment if you want energy-VAD segmenting later.

UI:
  - per-class softmax probability history
  - current probability bars
  - recent trigger log
  - 3 sliders mutable at runtime:
      * threshold
      * ema_alpha (sliding) or stop_thr (SegmentKWS)
      * trigger_frames (sliding) or silence_hangover_ms (SegmentKWS)

Every session is dumped to <output_dir>/live_kws_<timestamp>/:
  - recording.wav   16 kHz mono int16, the raw microphone stream
  - triggers.csv    one row per fired trigger: time_s, label, score

Usage:
    # list input devices, then pick one
    python demo_live_kws.py --list_devices

    python demo_live_kws.py \
        --classifier results/kws_classifier/best_v4_pad.pt \
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

from demo_sliding_kws import SR, SegmentKWS, SlidingKWS, load_classifier


WINDOW_VIEW_S = 5.0
UI_INTERVAL_MS = 100  # matplotlib redraw period


def list_input_devices():
    print(sd.query_devices())
    print('\nUse --device <index> to select one of the input-capable rows above.')


def describe_input_device(device_idx):
    info = sd.query_devices(device_idx, 'input')
    hostapi = sd.query_hostapis(info['hostapi'])['name']
    return '{} {}, {} in'.format(info['name'], hostapi, info['max_input_channels'])


def validate_input_channel(device_idx, stream_channels, input_channel, parser):
    if input_channel >= stream_channels:
        parser.error('--channel {} is outside the opened stream width {}. '
                     'Use --stream_channels at least {}.'
                     .format(input_channel, stream_channels, input_channel + 1))

    info = sd.query_devices(device_idx, 'input')
    max_channels = int(info['max_input_channels'])
    if stream_channels > max_channels:
        hostapi = sd.query_hostapis(info['hostapi'])['name']
        parser.error(
            '--stream_channels {} requested, but device {} '
            '({} / {}) exposes only {}. Run --list_devices and choose a device '
            'that shows 7 in, e.g. the EMEET WASAPI or WDM-KS entry.'
            .format(stream_channels, device_idx, info['name'],
                    hostapi, max_channels)
        )
    return info


def make_run_dir(output_dir):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(output_dir) / f'live_kws_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


class LiveKWS:
    def __init__(self, detector, labels, device_idx, stream_channels,
                 input_channel, hop_ms, run_dir, detector_kind='segment'):
        self.detector = detector
        self.detector_kind = detector_kind   # 'segment' or 'sliding'
        self.labels = list(labels)
        self.n_classes = len(self.labels)
        self.device_idx = device_idx
        self.stream_channels = stream_channels
        self.input_channel = input_channel
        self.hop_ms = hop_ms
        self.hop_samples = int(SR * hop_ms / 1000)
        self.run_dir = run_dir

        self.audio_q = queue.Queue()
        self.state_q = queue.Queue()
        self.stop_flag = threading.Event()

        # display buffers
        n_prob = int(WINDOW_VIEW_S * 1000 / hop_ms)
        self.prob_hist = [deque([1.0 / self.n_classes] * n_prob, maxlen=n_prob)
                          for _ in self.labels]
        self.time_hist = deque([0.0] * n_prob, maxlen=n_prob)

        # trigger history for plotting and log text; bounded to prevent unbounded growth
        self.trigger_log = deque(maxlen=1000)

        # output writers — opened in run()
        self.wav_writer = None
        self.csv_file = None
        self.csv_writer = None

        self._t_start = None
        self._in_speech = False   # live IN_SPEECH flag for UI indicator

    # ------------------------------------------------------------------ audio
    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print('Audio status:', status, file=sys.stderr)
        self.audio_q.put(indata[:, self.input_channel].copy())

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

            # torch.from_numpy()/Tensor.numpy() fail in some old torch + newer
            # Python environments when NumPy's C API cannot be initialized.
            chunk_tensor = torch.tensor(chunk.tolist(), dtype=torch.float32)
            label, score = self.detector.push(chunk_tensor)

            sprobs = self.detector._smooth_probs
            if sprobs is None:
                probs = np.full(self.n_classes, 1.0 / self.n_classes, dtype=np.float32)
            else:
                probs = np.asarray(sprobs.detach().cpu().tolist(), dtype=np.float32)

            in_speech = getattr(self.detector, '_state', None) == 'IN_SPEECH'

            now = time.perf_counter() - self._t_start
            self.state_q.put({
                't': now,
                'probs': probs,
                'trigger': (label, float(score)) if label is not None else None,
                'in_speech': in_speech,
            })

    # ------------------------------------------------------------------- ui
    def drain_state(self):
        while True:
            try:
                ev = self.state_q.get_nowait()
            except queue.Empty:
                break

            for i in range(self.n_classes):
                self.prob_hist[i].append(float(ev['probs'][i]))
            self.time_hist.append(ev['t'])
            self._in_speech = bool(ev.get('in_speech', False))

            if ev['trigger'] is not None:
                label, score = ev['trigger']
                self.trigger_log.append((ev['t'], label, score))
                if self.csv_writer is not None:
                    self.csv_writer.writerow([f'{ev["t"]:.3f}', label, f'{score:.4f}'])
                    self.csv_file.flush()
                print(f'  [TRIGGER] t={ev["t"]:.3f}s  label={label}  score={score:.4f}')

    def _build_ui(self):
        fig = plt.figure(figsize=(13, 7.5))
        gs = GridSpec(5, 2, figure=fig,
                      height_ratios=[1.6, 1.2, 0.25, 0.25, 0.25],
                      hspace=0.55, wspace=0.25)

        # ---- probability history
        ax_prob = fig.add_subplot(gs[0, :])
        prob_title = ax_prob.set_title('Smoothed softmax probability')
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
        ax_bar = fig.add_subplot(gs[1, 0])
        ax_bar.set_title('Current frame probabilities')
        ax_bar.set_ylim(0, 1)
        bars = ax_bar.bar(self.labels, [0] * self.n_classes, color=colors)
        bar_thr_line = ax_bar.axhline(self.detector.threshold, ls='--',
                                       color='red', lw=1.0, alpha=0.7)
        for tick in ax_bar.get_xticklabels():
            tick.set_rotation(15)

        # ---- trigger log text
        ax_log = fig.add_subplot(gs[1, 1])
        ax_log.set_title('Recent triggers')
        ax_log.axis('off')
        log_txt = ax_log.text(0.0, 1.0, '(none yet)', va='top', ha='left',
                              family='monospace', fontsize=10)

        # ---- sliders (3 rows, each spans both columns)
        ax_thr = fig.add_subplot(gs[2, :])
        ax_b = fig.add_subplot(gs[3, :])
        ax_c = fig.add_subplot(gs[4, :])
        slider_thr = Slider(ax_thr, 'threshold', 0.0, 1.0,
                            valinit=self.detector.threshold, valstep=0.005)

        def on_thr(v):
            self.detector.threshold = float(v)
            thr_line.set_ydata([v, v])
            bar_thr_line.set_ydata([v, v])
            thr_line.set_label(f'thr={v:.3f}')
            ax_prob.legend(loc='upper left', fontsize=8, ncol=self.n_classes + 1)
        slider_thr.on_changed(on_thr)

        if self.detector_kind == 'segment':
            # SegmentKWS knobs: stop_thr (energy gate floor) + silence_hangover_ms
            slider_b = Slider(ax_b, 'stop_thr', 0.001, 0.05,
                              valinit=self.detector.stop_thr, valstep=0.001)
            cur_hang_ms = self.detector.silence_hangover_frames * self.hop_ms
            slider_c = Slider(ax_c, 'silence_hangover_ms', 100, 1000,
                              valinit=cur_hang_ms, valstep=20)
            def on_b(v):
                self.detector.stop_thr = float(v)
            def on_c(v):
                self.detector.silence_hangover_frames = max(1, int(v / self.hop_ms))
            slider_b.on_changed(on_b)
            slider_c.on_changed(on_c)
        else:
            # SlidingKWS knobs (legacy): EMA alpha + trigger frames
            slider_b = Slider(ax_b, 'ema_alpha', 0.05, 1.0,
                              valinit=self.detector.ema_alpha, valstep=0.05)
            slider_c = Slider(ax_c, 'trigger_frames', 1, 10,
                              valinit=self.detector.trigger_frames, valstep=1)
            def on_b(v):
                self.detector.ema_alpha = float(v)
            def on_c(v):
                self.detector.trigger_frames = int(v)
            slider_b.on_changed(on_b)
            slider_c.on_changed(on_c)

        # ---- animation
        def update(_):
            self.drain_state()

            if self.detector_kind == 'segment':
                prob_title.set_text(
                    'Smoothed softmax probability   '
                    + ('● IN_SPEECH' if self._in_speech else '○ idle'))

            t_now = self.time_hist[-1] if self.time_hist else 0.0
            xs = np.array(self.time_hist) - t_now
            for i, line in enumerate(prob_lines):
                line.set_data(xs, list(self.prob_hist[i]))

            for i, b in enumerate(bars):
                b.set_height(self.prob_hist[i][-1])

            if self.trigger_log:
                tail = list(self.trigger_log)[-10:]
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

            return (*prob_lines, *bars, log_txt, trig_scatter)

        self._anim = FuncAnimation(fig, update, interval=UI_INTERVAL_MS,
                                   blit=False, cache_frame_data=False)
        # keep slider refs alive
        self._sliders = (slider_thr, slider_b, slider_c)
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
            channels=self.stream_channels,
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
    parser.add_argument('--input_channel', '--channel', type=int, default=6,
                        help='0-based input channel to use from the selected device '
                             '(default: 6, ch07)')
    parser.add_argument('--stream_channels', type=int, default=7,
                        help='number of input channels to open from the device '
                             '(default: 7)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Initial trigger threshold (default: thr_far05 from ckpt)')
    parser.add_argument('--hop_ms', type=int, default=20)
    parser.add_argument('--cooldown_ms', type=int, default=1000)
    parser.add_argument('--output_dir', default='live_kws_logs')
    parser.add_argument('--cpu', action='store_true')

    # detector choice
    parser.add_argument('--detector', choices=['sliding', 'segment'],
                        default='sliding',
                        help='sliding: no VAD, continuously classify a rolling '
                             'fixed window. segment: energy-VAD then classify '
                             'each detected speech segment.')

    # SegmentKWS knobs
    parser.add_argument('--start_thr', type=float, default=0.01)
    parser.add_argument('--stop_thr', type=float, default=0.005)
    parser.add_argument('--speech_onset_ms', type=int, default=60)
    parser.add_argument('--silence_hangover_ms', type=int, default=300)
    parser.add_argument('--min_speech_ms', type=int, default=200)
    parser.add_argument('--max_segment_ms', type=int, default=3100)
    parser.add_argument('--pre_onset_ms', type=int, default=100)

    # SlidingKWS knobs (only used with --detector sliding)
    parser.add_argument('--ema_alpha', type=float, default=0.3)
    parser.add_argument('--trigger_frames', type=int, default=3)
    parser.add_argument('--window_s', type=float, default=1.0)

    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return
    if not args.classifier:
        parser.error('--classifier is required (or use --list_devices)')
    if args.input_channel < 0:
        parser.error('--input_channel must be >= 0')
    if args.stream_channels <= 0:
        parser.error('--stream_channels must be > 0')
    validate_input_channel(args.device, args.stream_channels, args.input_channel, parser)

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f'Torch device: {device}')

    model, labels, _, suggested_thr = load_classifier(args.classifier, device)
    threshold = args.threshold if args.threshold is not None else (
        suggested_thr if suggested_thr is not None else 0.5)

    run_dir = make_run_dir(args.output_dir)

    if args.detector == 'segment':
        detector = SegmentKWS(
            model, labels,
            threshold=threshold,
            hop_ms=args.hop_ms,
            start_thr=args.start_thr,
            stop_thr=args.stop_thr,
            speech_onset_ms=args.speech_onset_ms,
            silence_hangover_ms=args.silence_hangover_ms,
            min_speech_ms=args.min_speech_ms,
            max_segment_ms=args.max_segment_ms,
            pre_onset_ms=args.pre_onset_ms,
            cooldown_ms=args.cooldown_ms,
            device=device,
            debug=False,
        )
    else:
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

    print(f'Detector       : {args.detector}')
    print(f'Classes        : {labels}')
    print(f'Threshold      : {threshold:.4f}')
    if args.detector == 'segment':
        print(f'start_thr      : {args.start_thr}')
        print(f'stop_thr       : {args.stop_thr}')
        print(f'silence_hangover_ms : {args.silence_hangover_ms}')
        print(f'max_segment_ms : {args.max_segment_ms}')
    else:
        print(f'EMA alpha      : {args.ema_alpha}')
        print(f'Trigger frames : {args.trigger_frames}')
        print(f'window_s       : {args.window_s}')
    print(f'Cooldown ms    : {args.cooldown_ms}')
    print(f'Mic device idx : {args.device if args.device is not None else "(default)"}')
    print(f'Mic device     : {describe_input_device(args.device)}')
    print(f'Stream channels: {args.stream_channels}')
    print(f'Input channel  : {args.input_channel} (0-based, ch{args.input_channel + 1:02d})\n')

    app = LiveKWS(detector, labels, args.device, args.stream_channels,
                  args.input_channel, args.hop_ms, run_dir,
                  detector_kind=args.detector)
    app.run()


if __name__ == '__main__':
    main()
