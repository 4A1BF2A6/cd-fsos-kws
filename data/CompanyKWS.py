# Wrapper for the in-house wake-word dataset.
# Layout:
#   <data_dir>/<wake>/<emp_id>/<env>_<dist>_<speed>_<take>/ch01.wav .. ch07.wav
#   <data_dir>/<wake>/<env>_background/*.wav
# Channel selection is via args['channel'] (default 'ch07', i.e. DSP-enhanced).
# Train/test split is speaker-disjoint by emp_id.

import os
from functools import partial
import glob
import math
import os.path
import random

import numpy as np
import torch
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset
import torchaudio
import torchaudio.functional as AF
import torch.nn.functional as F

from .data_utils import SetDataset


SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 1
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 0
RANDOM_SEED = 59185

BACKGROUND_DIR_SUFFIX = '_background'


def variable_length_collate(batch):
    """Collator for variable-length raw-waveform batches.

    Each item is the dict produced by CompanyKWSDataset's transforms:
        {'data': (C, T_i) float tensor, 'label_idx': scalar, 'label': str, ...}
    Output: same keys plus 'lengths' (B,) holding the original T_i.
    'data' is zero-padded on the right to batch-max T.

    Works unchanged for fixed-length modes (T_i is constant → no padding).
    Picklable (top-level def), so it survives multi-worker DataLoader.
    """
    from torch.utils.data._utils.collate import default_collate
    lengths = torch.tensor([b['data'].size(-1) for b in batch], dtype=torch.long)
    T_max = int(lengths.max())
    sample0 = batch[0]['data']
    data = torch.zeros(len(batch), sample0.size(0), T_max, dtype=sample0.dtype)
    for i, b in enumerate(batch):
        T_i = b['data'].size(-1)
        data[i, :, :T_i] = b['data']
    out = default_collate([{k: v for k, v in b.items() if k != 'data'} for b in batch])
    out['data'] = data
    out['lengths'] = lengths
    return out


class EpisodicFixedBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes, fixed_silence_unknown=False, include_unknown=True):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes
        if fixed_silence_unknown:
            skip = 2
            fixed_class = torch.tensor([SILENCE_INDEX, UNKNOWN_WORD_INDEX])
            n_way = n_way - skip
            self.sampling = []
            for _ in range(self.n_episodes):
                selected = torch.randperm(self.n_classes - skip)[:n_way]
                selected = torch.cat((fixed_class, selected.add(skip)))
                self.sampling.append(selected)
        else:
            self.sampling = [torch.randperm(self.n_classes)[:self.n_way] for _ in range(self.n_episodes)]

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield self.sampling[i]


def prepare_words_list(wanted_words, silence, unknown):
    extra_words = []
    if silence:
        extra_words.append(SILENCE_LABEL)
    if unknown:
        extra_words.append(UNKNOWN_WORD_LABEL)
    return extra_words + wanted_words


class CompanyKWSDataset:
    def __init__(self, data_dir, TaskType, cuda, args):
        self.sample_rate = args['sample_rate']
        self.clip_duration_ms = args['clip_duration']
        self.window_size_ms = args['window_size']
        self.window_stride_ms = args['window_stride']
        self.n_mfcc = args['n_mfcc']
        self.feature_bin_count = args['num_features']
        self.foreground_volume = args['foreground_volume']
        self.time_shift_ms = args['time_shift']
        self.desired_samples = int(self.sample_rate * self.clip_duration_ms / 1000)

        self.use_background = args['include_noise']
        self.background_volume = args['bg_volume']
        self.background_frequency = args['bg_frequency']

        self.silence = args['include_silence']
        self.silence_num_samples = args['num_silence']
        self.unknown = args['include_unknown']

        self.channel = args.get('channel', 'ch07')
        self.crop_strategy = args.get('crop_strategy', 'center')
        if self.crop_strategy not in ('center', 'energy', 'stretch', 'pad'):
            raise ValueError("crop_strategy must be 'center', 'energy', 'stretch' or 'pad', got {}".format(self.crop_strategy))
        # 'pad' mode upper bound; samples longer than this get energy-cropped.
        self.max_duration_ms = int(args.get('max_duration_ms', 3100))
        self.max_duration_samples = int(self.sample_rate * self.max_duration_ms / 1000)
        self.merge_val = args.get('merge_val', 'none')
        if self.merge_val not in ('none', 'train', 'test'):
            raise ValueError("merge_val must be 'none', 'train' or 'test', got {}".format(self.merge_val))
        self.gsc_unknown_dir = args.get('gsc_unknown_dir', None)
        self.gsc_unknown_words = [w.strip() for w in
            args.get('gsc_unknown_words', 'backward,forward,visual,follow,learn').split(',') if w.strip()]
        self.gsc_unknown_splits = args.get('gsc_unknown_splits', 'test')
        # LibriSpeech continuous-speech unknown augmentation
        self.librispeech_dir = args.get('librispeech_dir', None)
        self.librispeech_samples_per_file = int(args.get('librispeech_samples_per_file', 2))
        self.librispeech_max_files = int(args.get('librispeech_max_files', 0))  # 0 = no cap
        self.data_cache = {}
        self.data_dir = data_dir

        # split percentages
        params = {
            'silence_percentage': 10.0,
            'unknown_percentage': 0.0,    # unknown disabled by default for this dataset
            'validation_percentage': 10.0,
            'testing_percentage': 10.0,
        }

        wake_dirs = self._discover_wakes()
        if TaskType in (None, '', 'CompanyKWS_ALL'):
            wanted_words = sorted(wake_dirs)
        else:
            # explicit comma-separated list e.g. "wake1,wake2,wake3"
            wanted_words = [w for w in TaskType.split(',') if w]
            missing = [w for w in wanted_words if w not in wake_dirs]
            if missing:
                raise ValueError('Wake words not found under {}: {}'.format(data_dir, missing))
        params['wanted_words'] = wanted_words
        params['unknown_words'] = []  # background handled separately

        self.generate_data_dictionary(params)
        self.background_data = self.load_background_data()

        self.cuda = cuda
        self.max_class = len(wanted_words)

    # ------------------------------------------------------------------ utils

    def _discover_wakes(self):
        wakes = []
        for entry in os.listdir(self.data_dir):
            full = os.path.join(self.data_dir, entry)
            if not os.path.isdir(full):
                continue
            if entry.endswith(BACKGROUND_DIR_SUFFIX):
                continue
            wakes.append(entry)
        return wakes

    def _list_take_wavs(self, wake_dir):
        # Yield (emp_id, take_dir_basename, abs_wav_path) for the chosen channel.
        out = []
        chan_file = self.channel + '.wav'
        for emp_id in os.listdir(wake_dir):
            emp_path = os.path.join(wake_dir, emp_id)
            if not os.path.isdir(emp_path):
                continue
            if emp_id.endswith(BACKGROUND_DIR_SUFFIX):
                continue
            for take_name in os.listdir(emp_path):
                take_path = os.path.join(emp_path, take_name)
                if not os.path.isdir(take_path):
                    continue
                wav_path = os.path.join(take_path, chan_file)
                if os.path.isfile(wav_path):
                    out.append((emp_id, take_name, wav_path))
        return out

    def _split_speakers(self, all_speakers, val_pct, test_pct):
        speakers = sorted(all_speakers)
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(speakers)
        n = len(speakers)
        n_test = max(1, int(round(n * test_pct / 100.0))) if n >= 3 else 0
        n_val = max(1, int(round(n * val_pct / 100.0))) if n >= 3 else 0
        # keep at least one training speaker
        if n - n_test - n_val < 1:
            n_val = 0
            n_test = max(0, n - 1)
        test_set = set(speakers[:n_test])
        val_set = set(speakers[n_test:n_test + n_val])
        train_set = set(speakers[n_test + n_val:])
        return train_set, val_set, test_set

    # ------------------------------------------------------------ dataloaders

    def get_episodic_fixed_sampler(self, num_classes, n_way, n_episodes,
                                   fixed_silence_unknown=False, include_unknown=True):
        return EpisodicFixedBatchSampler(num_classes, n_way, n_episodes,
                                         fixed_silence_unknown=fixed_silence_unknown,
                                         include_unknown=include_unknown)

    def get_episodic_dataloader(self, set_index, n_way, n_samples, n_episodes, sampler='episodic',
                                include_silence=True, include_unknown=True, unique_speaker=False):
        class_list = []
        for item in self.words_list:
            if not include_silence and item == SILENCE_LABEL:
                continue
            if not include_unknown and item == UNKNOWN_WORD_LABEL:
                continue
            class_list.append(item)

        if sampler == 'episodic':
            sampler = self.get_episodic_fixed_sampler(len(class_list), n_way, n_episodes)

        dl_list = []
        if set_index in ['training', 'validation', 'testing']:
            for keyword in class_list:
                ts_ds = self.get_transform_dataset(self.data_set[set_index], [keyword])
                if n_samples <= 0:
                    n_samples = len(ts_ds)
                dl = torch.utils.data.DataLoader(ts_ds, batch_size=n_samples,
                                                 shuffle=True, num_workers=0)
                dl_list.append(dl)

            ds = SetDataset(dl_list)
            data_loader_params = dict(batch_sampler=sampler, num_workers=0,
                                      pin_memory=not self.cuda)
            dl = torch.utils.data.DataLoader(ds, **data_loader_params)
        else:
            raise ValueError("Set index = {} in episodic dataset is not correct.".format(set_index))

        return dl

    def get_iid_dataloader(self, set_index, batch_size, class_list=False,
                           include_silence=True, include_unknown=True, unique_speaker=False):
        if not class_list:
            class_list = []
            for item in self.words_list:
                if not include_silence and item == SILENCE_LABEL:
                    continue
                if not include_unknown and item == UNKNOWN_WORD_LABEL:
                    continue
                class_list.append(item)

        ts_ds = self.get_transform_dataset(self.data_set[set_index], class_list)
        dl = torch.utils.data.DataLoader(ts_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        return dl

    def dataset_filter_class(self, dslist, classes):
        return [item for item in dslist if item['label'] in classes]

    def get_transform_dataset(self, file_dict, classes, filters=None):
        transforms = compose([
            partial(self.load_audio, 'file', 'label', 'data'),
            partial(self.adjust_volume, 'data'),
            partial(self.shift_and_pad, 'data'),
            partial(self.mix_background, self.use_background, 'data', 'label'),
            partial(self.label_to_idx, 'label', 'label_idx'),
        ])
        file_dict = self.dataset_filter_class(file_dict, classes)
        ls_ds = ListDataset(file_dict)
        ts_ds = TransformDataset(ls_ds, transforms)
        return ts_ds

    def num_classes(self):
        return len(self.words_list)

    # ----------------------------------------------------------- transforms

    def label_to_idx(self, k, key_out, d):
        d[key_out] = torch.LongTensor([self.word_to_index[d[k]]]).squeeze()
        return d

    def mix_background(self, use_background, k, key_label, d):
        # _unknown_ samples ARE background slices already; don't mix more noise into them.
        if d[key_label] == UNKNOWN_WORD_LABEL:
            return d
        foreground = d[k]
        audio_len = foreground.size(1)   # matches desired_samples in fixed modes, variable in pad mode
        has_bg = len(self.background_data) > 0
        if has_bg and (use_background or d[key_label] == SILENCE_LABEL):
            background_index = np.random.randint(len(self.background_data))
            background_samples = self.background_data[background_index]
            if len(background_samples) <= audio_len:
                # too short — skip mixing for this sample
                background_reshaped = torch.zeros(1, audio_len)
                bg_vol = 0
            else:
                background_offset = np.random.randint(
                    0, len(background_samples) - audio_len)
                background_clipped = background_samples[background_offset:(
                    background_offset + audio_len)]
                background_reshaped = background_clipped.reshape([1, audio_len])
                if np.random.uniform(0, 1) < self.background_frequency:
                    bg_vol = np.random.uniform(0, self.background_volume)
                else:
                    bg_vol = 0
        else:
            background_reshaped = torch.zeros(1, audio_len)
            bg_vol = 0

        background_mul = background_reshaped * bg_vol
        background_add = background_mul + foreground
        d[k] = torch.clamp(background_add, -1.0, 1.0)
        return d

    def shift_and_pad(self, key, d):
        # Pad mode keeps samples at their native variable length — the random
        # time-shift trick depends on a fixed desired_samples window, so we skip
        # the augmentation entirely here. (Time-shift on already-variable
        # samples isn't meaningful: the wake word is already where it is.)
        if self.crop_strategy == 'pad':
            return d

        audio = d[key]
        time_shift = int((self.time_shift_ms * self.sample_rate) / 1000)
        if time_shift > 0:
            time_shift_amount = np.random.randint(-time_shift, time_shift)
        else:
            time_shift_amount = 0

        if time_shift_amount > 0:
            time_shift_padding = (time_shift_amount, 0)
            time_shift_offset = 0
        else:
            time_shift_padding = (0, -time_shift_amount)
            time_shift_offset = -time_shift_amount

        audio_len = audio.size(1)
        if audio_len < self.desired_samples:
            pad = (0, self.desired_samples - audio_len)
            audio = F.pad(audio, pad, 'constant', 0)

        padded = F.pad(audio, time_shift_padding, 'constant', 0)
        sliced = torch.narrow(padded, 1, time_shift_offset, self.desired_samples)
        d[key] = sliced
        return d

    def adjust_volume(self, key, d):
        d[key] = d[key] * self.foreground_volume
        return d

    def load_audio(self, key_path, key_label, out_field, d):
        # Shallow-copy so we don't permanently mutate the underlying record dict
        # (torchnet's ListDataset returns the same instance on each access).
        d = dict(d)
        # Fixed-length modes pre-allocate target_len = desired + max time-shift so
        # the subsequent shift_and_pad has slack to slide within. Pad mode keeps
        # samples native up to max_duration_samples (energy-cropped if longer).
        if self.crop_strategy == 'pad':
            target_len = self.max_duration_samples
        else:
            target_len = self.desired_samples + int(self.time_shift_ms * self.sample_rate / 1000)

        # Background slice path (GSC / LibriSpeech / wake-word backgrounds):
        # always read a fixed 1s-ish chunk regardless of crop_strategy — these
        # are the _unknown_ negatives and pretrained as ~1s.
        bg_offset_s = d.pop('bg_offset_seconds', None)
        if bg_offset_s is not None:
            bg_target_len = self.desired_samples + int(self.time_shift_ms * self.sample_rate / 1000)
            info = torchaudio.info(d[key_path])
            src_sr = info.sample_rate
            src_offset = int(bg_offset_s * src_sr)
            # Read a slightly larger window in source rate so resample lands on >= bg_target_len.
            src_n_frames = int((bg_target_len / self.sample_rate + 0.05) * src_sr)
            sound, sr = torchaudio.load(d[key_path], normalize=True,
                                        frame_offset=src_offset, num_frames=src_n_frames)
            if sound.size(0) > 1:
                sound = sound.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                sound = AF.resample(sound, sr, self.sample_rate)
            if sound.size(1) > bg_target_len:
                sound = sound[:, :bg_target_len]
            elif sound.size(1) < bg_target_len:
                sound = F.pad(sound, (0, bg_target_len - sound.size(1)))
            d[out_field] = sound
            return d

        sound, sr = torchaudio.load(d[key_path], normalize=True)
        if sound.size(0) > 1:
            sound = sound.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            sound = AF.resample(sound, sr, self.sample_rate)

        if self.crop_strategy == 'stretch':
            sound = self._stretch_to_target(sound, target_len)
        elif self.crop_strategy == 'pad':
            # Variable-length: only crop the tail (>3.1s) via energy peak;
            # keep native length otherwise.
            if sound.size(1) > target_len:
                start = self._energy_crop_start(sound, target_len)
                sound = sound[:, start:start + target_len]
        elif sound.size(1) > target_len:
            start = self._pick_crop_start(sound, target_len)
            sound = sound[:, start:start + target_len]

        if d[key_label] == SILENCE_LABEL:
            sound = torch.zeros(1, self.desired_samples)
        d[out_field] = sound
        return d

    def _pick_crop_start(self, sound, max_len):
        if self.crop_strategy == 'center':
            return (sound.size(1) - max_len) // 2
        return self._energy_crop_start(sound, max_len)

    def _energy_crop_start(self, sound, max_len):
        """Return the start sample of the max-RMS window of `max_len` samples.

        Coarse 10ms hop is enough — we only need rough peak localisation.
        """
        hop = max(1, int(0.01 * self.sample_rate))
        sq = sound.pow(2).sum(dim=0)  # (T,)
        csum = torch.cat([torch.zeros(1, dtype=sq.dtype), sq.cumsum(0)])
        T = sq.size(0)
        n_starts = (T - max_len) // hop + 1
        if n_starts <= 1:
            return 0
        starts = torch.arange(n_starts) * hop
        win_energy = csum[starts + max_len] - csum[starts]
        return int(starts[win_energy.argmax()].item())

    def _stretch_to_target(self, sound, target_len):
        """Time-stretch waveform to exactly target_len samples.

        Uses linear interpolation (F.interpolate) instead of sinc resampling —
        10-20x faster on CPU, sufficient quality for MFCC feature extraction.
        """
        orig_len = sound.size(1)
        if orig_len == target_len:
            return sound
        # F.interpolate expects (N, C, L); sound is (1, L)
        stretched = F.interpolate(
            sound.unsqueeze(0), size=target_len, mode='linear', align_corners=False
        ).squeeze(0)
        return stretched

    def load_background_data(self):
        background_data = []
        if not (self.use_background or self.silence):
            return background_data
        # Collect from <wake>/<*_background>/*.wav
        pattern = os.path.join(self.data_dir, '*', '*' + BACKGROUND_DIR_SUFFIX, '*.wav')
        for wav_path in glob.glob(pattern):
            try:
                sound, sr = torchaudio.load(wav_path)
                if sound.size(0) > 1:
                    sound = sound.mean(dim=0, keepdim=True)
                if sr != self.sample_rate:
                    sound = AF.resample(sound, sr, self.sample_rate)
                background_data.append(sound.flatten())
            except Exception as exc:
                print('[CompanyKWS] skip background {}: {}'.format(wav_path, exc))
        return background_data

    # ------------------------------------------------------- data dictionary

    def generate_data_dictionary(self, training_parameters):
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        global SILENCE_INDEX
        skip = 0
        if self.silence:
            skip += 1
        if self.unknown:
            skip += 1
        else:
            SILENCE_INDEX = SILENCE_INDEX - 1 if SILENCE_INDEX > 0 else SILENCE_INDEX

        wanted_words_index = {w: i + skip for i, w in enumerate(training_parameters['wanted_words'])}

        # First pass — gather records per (split TBD) so we can do speaker-disjoint split.
        records = []  # list of (wake, emp_id, take_name, wav_path)
        all_speakers = set()
        for wake in training_parameters['wanted_words']:
            wake_dir = os.path.join(self.data_dir, wake)
            for emp_id, take_name, wav_path in self._list_take_wavs(wake_dir):
                records.append((wake, emp_id, take_name, wav_path))
                all_speakers.add(emp_id)

        if not records:
            raise Exception('No wav files found for channel {} under {}'.format(
                self.channel, self.data_dir))

        train_set, val_set, test_set = self._split_speakers(
            all_speakers,
            training_parameters['validation_percentage'],
            training_parameters['testing_percentage'])

        self.data_set = {'validation': [], 'testing': [], 'training': []}
        for wake, emp_id, _take, wav_path in records:
            entry = {'label': wake, 'file': wav_path, 'speaker': emp_id}
            if emp_id in test_set:
                self.data_set['testing'].append(entry)
            elif emp_id in val_set:
                self.data_set['validation'].append(entry)
            else:
                self.data_set['training'].append(entry)

        # _unknown_ samples sliced from background wavs (open-set rejection class)
        if self.unknown:
            bg_pattern = os.path.join(self.data_dir, '*', '*' + BACKGROUND_DIR_SUFFIX, '*.wav')
            bg_meta = []
            min_dur = self.clip_duration_ms / 1000.0 + 0.2
            for f in sorted(glob.glob(bg_pattern)):
                try:
                    info = torchaudio.info(f)
                    dur = info.num_frames / info.sample_rate
                    if dur >= min_dur:
                        bg_meta.append((f, dur))
                except Exception as exc:
                    print('[CompanyKWS] skip background {} ({})'.format(f, exc))
            if not bg_meta:
                print('[CompanyKWS] WARNING: include_unknown=True but no usable '
                      '*_background/*.wav found; _unknown_ class will be empty.')
            else:
                rng_unk = random.Random(RANDOM_SEED + 7)
                n_wakes = max(1, len(training_parameters['wanted_words']))
                for split in ['training', 'validation', 'testing']:
                    set_size = len(self.data_set[split])
                    if set_size == 0:
                        continue
                    # Aim for one unknown sample per per-class average — keeps the
                    # _unknown_ class balanced against each wake-word class.
                    unk_count = int(math.ceil(set_size / n_wakes))
                    for _ in range(unk_count):
                        f, dur = rng_unk.choice(bg_meta)
                        max_offset = max(0.0, dur - self.clip_duration_ms / 1000.0 - 0.1)
                        bg_offset = rng_unk.uniform(0.0, max_offset) if max_offset > 0 else 0.0
                        self.data_set[split].append({
                            'label': UNKNOWN_WORD_LABEL,
                            'file': f,
                            'speaker': 'background',
                            'bg_offset_seconds': bg_offset,
                        })

        # silence samples
        if self.silence and len(self.data_set['training']) > 0:
            silence_wav_path = self.data_set['training'][0]['file']
            for split in ['validation', 'testing', 'training']:
                set_size = len(self.data_set[split])
                if set_size == 0:
                    continue
                silence_size = int(math.ceil(set_size * training_parameters['silence_percentage'] / 100))
                for _ in range(silence_size):
                    self.data_set[split].append({
                        'label': SILENCE_LABEL,
                        'file': silence_wav_path,
                        'speaker': 'None',
                    })

        # Inject GSC human-speech words as harder _unknown_ negatives
        if self.unknown and self.gsc_unknown_dir:
            gsc_entries = []
            for word in self.gsc_unknown_words:
                word_dir = os.path.join(self.gsc_unknown_dir, word)
                if not os.path.isdir(word_dir):
                    print('[CompanyKWS] GSC word dir not found, skipping: {}'.format(word_dir))
                    continue
                for wav in glob.glob(os.path.join(word_dir, '*.wav')):
                    gsc_entries.append({'label': UNKNOWN_WORD_LABEL, 'file': wav, 'speaker': 'gsc_unknown'})
            if gsc_entries:
                rng_gsc = random.Random(RANDOM_SEED + 13)
                rng_gsc.shuffle(gsc_entries)
                if self.gsc_unknown_splits == 'all':
                    # 80% train / 10% val / 10% test, mirroring CompanyKWS percentages
                    n = len(gsc_entries)
                    n_train = int(n * 0.8)
                    n_val   = int(n * 0.1)
                    self.data_set['training'].extend(gsc_entries[:n_train])
                    self.data_set['validation'].extend(gsc_entries[n_train:n_train + n_val])
                    self.data_set['testing'].extend(gsc_entries[n_train + n_val:])
                    print('[CompanyKWS] Added {} GSC unknown entries '
                          '(train={}, val={}, test={})'.format(
                              n, n_train, n_val, n - n_train - n_val))
                else:
                    self.data_set['testing'].extend(gsc_entries)
                    print('[CompanyKWS] Added {} GSC unknown entries to testing split'.format(
                        len(gsc_entries)))

        # Inject LibriSpeech continuous-speech slices as harder _unknown_ negatives.
        # Speaker-disjoint split (80/10/10): same speaker never appears in two splits,
        # otherwise the model can memorise speaker-specific cues and fake high val FAR.
        if self.unknown and self.librispeech_dir:
            min_dur = self.clip_duration_ms / 1000.0 + 0.2
            slice_dur = self.clip_duration_ms / 1000.0
            flac_paths = sorted(glob.glob(os.path.join(
                self.librispeech_dir, '**', '*.flac'), recursive=True))
            if self.librispeech_max_files > 0 and len(flac_paths) > self.librispeech_max_files:
                rng_cap = random.Random(RANDOM_SEED + 23)
                rng_cap.shuffle(flac_paths)
                flac_paths = flac_paths[:self.librispeech_max_files]

            # LibriSpeech path: <root>/<speaker_id>/<chapter_id>/<file>.flac
            #                                   ↑ third-from-last
            def _ls_speaker(path):
                parts = path.split(os.sep)
                return parts[-3] if len(parts) >= 3 else 'unknown_speaker'

            # Group files by speaker; split speakers (not files) into 80/10/10
            speaker_to_files = {}
            for f in flac_paths:
                speaker_to_files.setdefault(_ls_speaker(f), []).append(f)
            ls_speakers = sorted(speaker_to_files.keys())
            rng_split = random.Random(RANDOM_SEED + 31)
            rng_split.shuffle(ls_speakers)
            n_spk = len(ls_speakers)
            n_train_spk = int(n_spk * 0.8)
            n_val_spk   = int(n_spk * 0.1)
            spk_split = {
                'training':   set(ls_speakers[:n_train_spk]),
                'validation': set(ls_speakers[n_train_spk:n_train_spk + n_val_spk]),
                'testing':    set(ls_speakers[n_train_spk + n_val_spk:]),
            }

            rng_ls = random.Random(RANDOM_SEED + 17)
            per_split_entries = {'training': [], 'validation': [], 'testing': []}
            n_skipped_short = 0

            for f in flac_paths:
                spk = _ls_speaker(f)
                target_split = (
                    'training'   if spk in spk_split['training']   else
                    'validation' if spk in spk_split['validation'] else
                    'testing'
                )
                try:
                    info = torchaudio.info(f)
                    dur = info.num_frames / info.sample_rate
                except Exception as exc:
                    print('[CompanyKWS] LibriSpeech info failed {}: {}'.format(f, exc))
                    continue
                if dur < min_dur:
                    n_skipped_short += 1
                    continue
                max_offset = dur - slice_dur - 0.05
                for _ in range(self.librispeech_samples_per_file):
                    offset = rng_ls.uniform(0.0, max_offset)
                    per_split_entries[target_split].append({
                        'label': UNKNOWN_WORD_LABEL,
                        'file': f,
                        'speaker': 'librispeech_' + spk,
                        'bg_offset_seconds': offset,
                    })

            for split, entries in per_split_entries.items():
                self.data_set[split].extend(entries)
            total = sum(len(v) for v in per_split_entries.values())
            print('[CompanyKWS] LibriSpeech speaker-disjoint split '
                  '({} speakers → train/val/test = {}/{}/{}) → '
                  '{} slices (train={}, val={}, test={}, skipped_short={})'.format(
                      n_spk, n_train_spk, n_val_spk, n_spk - n_train_spk - n_val_spk,
                      total,
                      len(per_split_entries['training']),
                      len(per_split_entries['validation']),
                      len(per_split_entries['testing']),
                      n_skipped_short))

        # optionally absorb validation into another split (default: keep standalone)
        if self.merge_val == 'train' and self.data_set['validation']:
            self.data_set['training'].extend(self.data_set['validation'])
            self.data_set['validation'] = []
        elif self.merge_val == 'test' and self.data_set['validation']:
            self.data_set['testing'].extend(self.data_set['validation'])
            self.data_set['validation'] = []

        # deterministic shuffle
        for split in ['validation', 'testing', 'training']:
            random.Random(RANDOM_SEED).shuffle(self.data_set[split])

        self.words_list = prepare_words_list(training_parameters['wanted_words'],
                                             self.silence, self.unknown)
        self.word_to_index = dict(wanted_words_index)
        if self.silence:
            self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX
        if self.unknown:
            self.word_to_index[UNKNOWN_WORD_LABEL] = UNKNOWN_WORD_INDEX

        # surface split sizes for visibility
        def _count_unk(split):
            return sum(1 for r in self.data_set[split] if r['label'] == UNKNOWN_WORD_LABEL)
        print('[CompanyKWS] channel={} | crop={} | merge_val={} | unknown={} | wakes={} | '
              'speakers train/val/test = {}/{}/{} | '
              'samples train/val/test = {}/{}/{} | '
              '_unknown_ in train/val/test = {}/{}/{}'.format(
                  self.channel, self.crop_strategy, self.merge_val, self.unknown,
                  len(training_parameters['wanted_words']),
                  len(train_set), len(val_set), len(test_set),
                  len(self.data_set['training']),
                  len(self.data_set['validation']),
                  len(self.data_set['testing']),
                  _count_unk('training'), _count_unk('validation'), _count_unk('testing')))
