#!/usr/bin/env python3
"""
Build LibriSpeech phoneme hard-negative clips from alignment TextGrid files.

The script mines short snippets that are phonetically close to "Hey Camy" or
"Hey Reco" but should remain _unknown_ examples. It writes wav snippets,
manifests, and simple QA reports. Training code is intentionally not modified.

Example:
    python tools/build_librispeech_phoneme_hardneg.py \
        --alignment_root /mnt/vdb1/logic/librispeech_alignments \
        --librispeech_root /mnt/vdb1/logic/Librispeech/LibriSpeech \
        --out_dir /mnt/vdb1/logic/kws_hard_negative/librispeech_phoneme_hardneg_v1 \
        --splits train-clean-100 \
        --max_per_category 1000 \
        --num_workers 8
"""

import argparse
import csv
import hashlib
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None


MANIFEST_FIELDS = [
    'wav_path', 'label', 'category', 'target', 'source_dataset', 'source_split',
    'source_utt_id', 'speaker_id', 'chapter_id', 'start_sec', 'end_sec',
    'duration_sec', 'words', 'phones', 'normalized_phones', 'similarity',
    'edit_distance', 'matched_rule', 'blacklist_flag',
]

TARGETS = {
    'hey': ['HH', 'EY'],
    'hi_high': ['HH', 'AY'],
    'he': ['HH', 'IY'],
    'hey_camy': ['HH', 'EY', 'K', 'AE', 'M', 'IY'],
    'hey_reco': ['HH', 'EY', 'R', 'IY', 'K', 'OW'],
    'camy': ['K', 'AE', 'M', 'IY'],
    'reco': ['R', 'IY', 'K', 'OW'],
}

WAKE_TARGETS = {
    'hey_camy': TARGETS['hey_camy'],
    'hey_reco': TARGETS['hey_reco'],
}

TARGET_FOR_CATEGORY = {
    'hey_phoneme_similar': ('hey', 'hi_high', 'he'),
    'hey_nonwake': ('hey_camy', 'hey_reco'),
    'camy_phoneme_similar': ('camy',),
    'reco_phoneme_similar': ('reco',),
    'local_phoneme_confuser': ('hey_camy', 'hey_reco'),
}

HEY_WORDS = {'HEY', 'HAY', 'HI', 'HIGH'}

TEXT_BLACKLIST = {
    'HEY CAMY', 'HEY CAMI', 'HEY CAMMY', 'HEY KAMI', 'HEY KAMY',
    'HEY KEMI', 'HEY KEMY', 'HEY KAMEY', 'HEY KAMEE', 'HEY CAMEY',
    'HEY RECO', 'HEY REKO', 'HEY RICO', 'HEY RIKO', 'HEY RICKO',
    'HEY RECKO', 'HEY REECO',
    'CAMY', 'CAMI', 'CAMMY', 'KAMI', 'KAMY', 'KEMI', 'KEMY', 'KAMEY',
    'KAMEE', 'CAMEY',
    'RECO', 'REKO', 'RICO', 'RIKO', 'RICKO', 'RECKO', 'REECO',
}

NOISE_WORDS = {'', '<UNK>', '<SPOKEN_NOISE>', '{NS}', '[NOISE]'}
SILENCE_PHONES = {'', 'SIL', 'SP', 'SPN', 'NSN', 'LAU', 'BRTH', 'CGN'}
NOISE_PHONES = {'SPN', 'NSN', 'LAU', 'BRTH', 'CGN'}
PHONE_STRESS_RE = re.compile(r'\d')
TEXTGRID_TEXT_RE = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')
TEXTGRID_FLOAT_RE = re.compile(r'^\s*(xmin|xmax)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$')
TEXTGRID_NAME_RE = re.compile(r'^\s*name\s*=\s*"(.*)"\s*$')


@dataclass
class Interval:
    start: float
    end: float
    text: str


@dataclass
class Utterance:
    utt_id: str
    speaker_id: str
    chapter_id: str
    split: str
    words: List[Interval]
    phones: List[Interval]


@dataclass
class Candidate:
    category: str
    target: str
    source_split: str
    source_utt_id: str
    speaker_id: str
    chapter_id: str
    start_sec: float
    end_sec: float
    words: List[str]
    phones: List[str]
    normalized_phones: List[str]
    similarity: float
    edit_distance: int
    matched_rule: str
    blacklist_flag: str = ''
    wav_path: str = ''
    score: float = 0.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)

    @property
    def phrase(self) -> str:
        return ' '.join(self.words).upper()


@dataclass
class ProcessResult:
    candidates: List[Candidate] = field(default_factory=list)
    blacklist_hits: List[Candidate] = field(default_factory=list)
    missing_audio: List[Dict[str, str]] = field(default_factory=list)
    parse_errors: List[Dict[str, str]] = field(default_factory=list)


def strip_stress(phone: str) -> str:
    return PHONE_STRESS_RE.sub('', phone.upper().strip())


def normalize_phone(phone: str) -> str:
    phone = strip_stress(phone)
    return 'SIL' if phone in SILENCE_PHONES else phone


def scoring_phones(phones: Iterable[str]) -> List[str]:
    return [
        p for p in (normalize_phone(phone) for phone in phones)
        if p and p != 'SIL' and p not in NOISE_PHONES
    ]


def normalize_word(word: str) -> str:
    word = word.strip().upper()
    word = word.replace('’', "'")
    word = re.sub(r"^[^A-Z'<]+|[^A-Z'>]+$", '', word)
    return word


def clean_textgrid_text(text: str) -> str:
    return text.replace('""', '"').strip()


def safe_float(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return math.nan


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            sub_cost = 0 if x == y else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + sub_cost))
        prev = cur
    return prev[-1]


def phone_similarity(candidate: Sequence[str], target: Sequence[str]) -> Tuple[float, int]:
    if not candidate or not target:
        return 0.0, max(len(candidate), len(target))
    dist = levenshtein(candidate, target)
    denom = float(max(len(candidate), len(target), 1))
    return max(0.0, 1.0 - dist / denom), dist


def best_target_match(
    phones: Sequence[str],
    target_names: Sequence[str],
) -> Tuple[str, float, int]:
    best = ('', -1.0, 10 ** 9)
    for target_name in target_names:
        sim, dist = phone_similarity(phones, TARGETS[target_name])
        if (sim, -dist, target_name) > (best[1], -best[2], best[0]):
            best = (target_name, sim, dist)
    return best


def parse_textgrid(path: Path, split: str) -> Utterance:
    tiers: Dict[str, List[Interval]] = {}
    current_name: Optional[str] = None
    in_interval = False
    cur_start: Optional[float] = None
    cur_end: Optional[float] = None
    cur_text: Optional[str] = None

    with path.open('r', encoding='utf-8', errors='replace') as f:
        for line in f:
            name_match = TEXTGRID_NAME_RE.match(line)
            if name_match:
                current_name = name_match.group(1)
                tiers.setdefault(current_name, [])
                in_interval = False
                cur_start = cur_end = None
                cur_text = None
                continue

            if current_name is None:
                continue

            if re.match(r'^\s*intervals\s*\[\d+\]:\s*$', line):
                in_interval = True
                cur_start = cur_end = None
                cur_text = None
                continue

            if not in_interval:
                continue

            float_match = TEXTGRID_FLOAT_RE.match(line)
            if float_match:
                key, value = float_match.groups()
                if key == 'xmin':
                    cur_start = safe_float(value)
                else:
                    cur_end = safe_float(value)
                continue

            text_match = TEXTGRID_TEXT_RE.match(line)
            if text_match:
                cur_text = clean_textgrid_text(text_match.group(1))
                if cur_start is not None and cur_end is not None:
                    tiers.setdefault(current_name, []).append(
                        Interval(cur_start, cur_end, cur_text)
                    )
                in_interval = False

    if 'words' not in tiers or 'phones' not in tiers:
        raise ValueError('TextGrid missing required words/phones tiers')

    utt_id = path.stem
    parts = utt_id.split('-')
    speaker_id = parts[0] if len(parts) >= 1 else ''
    chapter_id = parts[1] if len(parts) >= 2 else ''
    return Utterance(
        utt_id=utt_id,
        speaker_id=speaker_id,
        chapter_id=chapter_id,
        split=split,
        words=tiers['words'],
        phones=tiers['phones'],
    )


def load_unaligned(alignment_root: Path, splits: Sequence[str]) -> set:
    unaligned = set()
    for split in splits:
        path = alignment_root / split / 'unaligned.txt'
        if not path.exists():
            continue
        with path.open('r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                unaligned.add(line.split()[0])
    return unaligned


def discover_textgrids(alignment_root: Path, splits: Sequence[str], unaligned: set) -> List[Tuple[str, str]]:
    jobs = []
    for split in splits:
        split_root = alignment_root / split
        if not split_root.exists():
            raise FileNotFoundError('Alignment split not found: {}'.format(split_root))
        for path in split_root.rglob('*.TextGrid'):
            if path.stem in unaligned:
                continue
            jobs.append((str(path), split))
    jobs.sort()
    return jobs


def build_audio_index(librispeech_root: Path, splits: Sequence[str]) -> Dict[str, Path]:
    index = {}
    for split in splits:
        split_root = librispeech_root / split
        if not split_root.exists():
            continue
        for path in split_root.rglob('*.flac'):
            index[path.stem] = path
    return index


def intervals_for_span(intervals: Sequence[Interval], start: float, end: float) -> List[Interval]:
    eps = 1e-5
    return [iv for iv in intervals if iv.end > start + eps and iv.start < end - eps]


def phones_for_span(utt: Utterance, start: float, end: float) -> Tuple[List[str], List[str]]:
    phones = [iv.text for iv in intervals_for_span(utt.phones, start, end)]
    normalized = [normalize_phone(p) for p in phones]
    return phones, normalized


def visible_words(words: Sequence[Interval]) -> List[str]:
    out = []
    for iv in words:
        word = normalize_word(iv.text)
        if word and word not in NOISE_WORDS:
            out.append(word)
    return out


def candidate_from_words(
    utt: Utterance,
    word_span: Sequence[Interval],
    category: str,
    target_names: Sequence[str],
    rule: str,
    pre_pad_sec: float,
    post_pad_sec: float,
) -> Optional[Candidate]:
    words = visible_words(word_span)
    if not words:
        return None
    raw_start = min(iv.start for iv in word_span)
    raw_end = max(iv.end for iv in word_span)
    start = max(0.0, raw_start - pre_pad_sec)
    end = raw_end + post_pad_sec
    phones, normalized = phones_for_span(utt, raw_start, raw_end)
    score_phones = scoring_phones(normalized)
    target, sim, dist = best_target_match(score_phones, target_names)
    return Candidate(
        category=category,
        target=target,
        source_split=utt.split,
        source_utt_id=utt.utt_id,
        speaker_id=utt.speaker_id,
        chapter_id=utt.chapter_id,
        start_sec=start,
        end_sec=end,
        words=words,
        phones=phones,
        normalized_phones=normalized,
        similarity=sim,
        edit_distance=dist,
        matched_rule=rule,
        score=sim,
    )


def candidate_from_phone_span(
    utt: Utterance,
    phones_span: Sequence[Interval],
    target_names: Sequence[str],
    pre_pad_sec: float,
    post_pad_sec: float,
) -> Optional[Candidate]:
    raw_start = min(iv.start for iv in phones_span)
    raw_end = max(iv.end for iv in phones_span)
    phones = [iv.text for iv in phones_span]
    normalized = [normalize_phone(p) for p in phones]
    score_phones = scoring_phones(normalized)
    target, sim, dist = best_target_match(score_phones, target_names)
    words = visible_words(intervals_for_span(utt.words, raw_start, raw_end))
    if not words:
        words = ['<PHONE_SPAN>']
    return Candidate(
        category='local_phoneme_confuser',
        target=target,
        source_split=utt.split,
        source_utt_id=utt.utt_id,
        speaker_id=utt.speaker_id,
        chapter_id=utt.chapter_id,
        start_sec=max(0.0, raw_start - pre_pad_sec),
        end_sec=raw_end + post_pad_sec,
        words=words,
        phones=phones,
        normalized_phones=normalized,
        similarity=sim,
        edit_distance=dist,
        matched_rule='local_phone_ngram',
        score=sim,
    )


def text_blacklist_flag(candidate: Candidate) -> str:
    phrase = candidate.phrase
    if phrase in TEXT_BLACKLIST:
        return 'text_blacklist'
    for item in TEXT_BLACKLIST:
        if phrase == item or phrase.startswith(item + ' ') or phrase.endswith(' ' + item):
            return 'text_blacklist'
    return ''


def phone_blacklist_flag(candidate: Candidate, max_edit_distance: int) -> str:
    phones = scoring_phones(candidate.normalized_phones)
    for name, target in WAKE_TARGETS.items():
        sim, dist = phone_similarity(phones, target)
        if dist <= max(1, min(max_edit_distance, 1)):
            return 'wake_phone_blacklist:{}'.format(name)
        if sim >= 0.92 and len(phones) >= len(target) - 1:
            return 'wake_phone_blacklist:{}'.format(name)
    return ''


def blacklist_flag(candidate: Candidate, max_edit_distance: int) -> str:
    return text_blacklist_flag(candidate) or phone_blacklist_flag(candidate, max_edit_distance)


def pass_candidate_filters(
    candidate: Candidate,
    min_sec: float,
    max_sec: float,
    similarity_threshold: float,
    max_edit_distance: int,
) -> bool:
    if candidate.duration_sec < min_sec or candidate.duration_sec > max_sec:
        return False
    if candidate.similarity < similarity_threshold:
        return False
    if candidate.edit_distance > max_edit_distance:
        return False
    if len(scoring_phones(candidate.normalized_phones)) < 2:
        return False
    return True


def pass_hey_prefix_filters(
    candidate: Candidate,
    min_sec: float,
    max_sec: float,
) -> bool:
    if candidate.duration_sec < min_sec or candidate.duration_sec > max_sec:
        return False
    phones = scoring_phones(candidate.normalized_phones)
    if len(phones) < 2 or len(phones) > 4:
        return False
    if candidate.edit_distance > 1:
        return False
    return candidate.similarity >= 0.66


def generate_candidates_for_utterance(
    utt: Utterance,
    min_sec: float,
    max_sec: float,
    pre_pad_sec: float,
    post_pad_sec: float,
    similarity_threshold: float,
    max_edit_distance: int,
) -> Tuple[List[Candidate], List[Candidate]]:
    candidates = []
    blacklist_hits = []
    word_items = [iv for iv in utt.words if normalize_word(iv.text) and normalize_word(iv.text) not in NOISE_WORDS]

    for i, word_iv in enumerate(word_items):
        word = normalize_word(word_iv.text)
        if word in HEY_WORDS:
            for next_count in (1, 2):
                end_idx = i + 1 + next_count
                if end_idx <= len(word_items):
                    cand = candidate_from_words(
                        utt, word_items[i:end_idx], 'hey_nonwake',
                        TARGET_FOR_CATEGORY['hey_nonwake'],
                        'hey_plus_{}_word{}'.format(next_count, '' if next_count == 1 else 's'),
                        pre_pad_sec, post_pad_sec,
                    )
                    if cand:
                        candidates.append(cand)

        for span_len in (1, 2):
            end_idx = i + span_len
            if end_idx <= len(word_items):
                cand = candidate_from_words(
                    utt, word_items[i:end_idx], 'hey_phoneme_similar',
                    TARGET_FOR_CATEGORY['hey_phoneme_similar'],
                    'hey_prefix_word_{}_gram'.format(span_len),
                    pre_pad_sec, post_pad_sec,
                )
                if cand:
                    candidates.append(cand)

        for span_len, category in ((1, 'camy_phoneme_similar'), (2, 'camy_phoneme_similar')):
            end_idx = i + span_len
            if end_idx <= len(word_items):
                cand = candidate_from_words(
                    utt, word_items[i:end_idx], category,
                    TARGET_FOR_CATEGORY[category],
                    'word_{}_gram'.format(span_len),
                    pre_pad_sec, post_pad_sec,
                )
                if cand:
                    candidates.append(cand)

        for span_len, category in ((1, 'reco_phoneme_similar'), (2, 'reco_phoneme_similar')):
            end_idx = i + span_len
            if end_idx <= len(word_items):
                cand = candidate_from_words(
                    utt, word_items[i:end_idx], category,
                    TARGET_FOR_CATEGORY[category],
                    'word_{}_gram'.format(span_len),
                    pre_pad_sec, post_pad_sec,
                )
                if cand:
                    candidates.append(cand)

    usable_phone_intervals = [iv for iv in utt.phones if normalize_phone(iv.text) not in ('', 'SIL')]
    for span_len in range(4, 8):
        for i in range(0, max(0, len(usable_phone_intervals) - span_len + 1)):
            span = usable_phone_intervals[i:i + span_len]
            cand = candidate_from_phone_span(
                utt, span, TARGET_FOR_CATEGORY['local_phoneme_confuser'],
                pre_pad_sec, post_pad_sec,
            )
            if cand:
                candidates.append(cand)

    kept = []
    for cand in candidates:
        flag = blacklist_flag(cand, max_edit_distance)
        cand.blacklist_flag = flag
        if flag:
            blacklist_hits.append(cand)
            continue
        if cand.category == 'hey_phoneme_similar':
            if pass_hey_prefix_filters(cand, min_sec, max_sec):
                kept.append(cand)
        elif pass_candidate_filters(cand, min_sec, max_sec, similarity_threshold, max_edit_distance):
            kept.append(cand)
    return dedupe_utterance_candidates(kept), blacklist_hits


def interval_iou(a: Candidate, b: Candidate) -> float:
    inter = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec)
    return inter / union if union > 0 else 0.0


def dedupe_utterance_candidates(candidates: Sequence[Candidate]) -> List[Candidate]:
    selected: List[Candidate] = []
    for cand in sorted(candidates, key=lambda c: (-c.score, c.edit_distance, c.start_sec)):
        if all(interval_iou(cand, prev) <= 0.5 for prev in selected):
            selected.append(cand)
    selected.sort(key=lambda c: (c.category, c.start_sec))
    return selected


def process_textgrid_job(
    job: Tuple[str, str],
    min_sec: float,
    max_sec: float,
    pre_pad_sec: float,
    post_pad_sec: float,
    similarity_threshold: float,
    max_edit_distance: int,
    audio_present: bool,
) -> ProcessResult:
    path_str, split = job
    result = ProcessResult()
    path = Path(path_str)
    try:
        utt = parse_textgrid(path, split)
        if not audio_present:
            result.missing_audio.append({
                'source_split': split,
                'source_utt_id': path.stem,
                'speaker_id': path.stem.split('-')[0] if '-' in path.stem else '',
                'chapter_id': path.stem.split('-')[1] if path.stem.count('-') >= 2 else '',
                'reason': 'flac_not_found',
            })
            return result
        candidates, blacklist_hits = generate_candidates_for_utterance(
            utt, min_sec, max_sec, pre_pad_sec, post_pad_sec,
            similarity_threshold, max_edit_distance,
        )
        result.candidates.extend(candidates)
        result.blacklist_hits.extend(blacklist_hits)
    except Exception as exc:
        result.parse_errors.append({
            'source_split': split,
            'source_utt_id': path.stem,
            'path': str(path),
            'error': repr(exc),
        })
    return result


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]


def limit_candidates(
    candidates: Sequence[Candidate],
    max_per_category: int,
    seed: int,
    max_per_speaker_category: int,
    max_per_phrase_category: int,
    max_per_hey_phrase_category: int,
) -> List[Candidate]:
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for cand in candidates:
        by_category[cand.category].append(cand)

    limited = []
    for category, items in sorted(by_category.items()):
        speaker_counts = Counter()
        phrase_counts = Counter()
        shuffled = list(items)
        rng.shuffle(shuffled)
        shuffled.sort(key=lambda c: (-c.score, c.edit_distance, c.duration_sec))
        selected = []
        for cand in shuffled:
            phrase_limit = (
                max_per_hey_phrase_category
                if cand.category == 'hey_phoneme_similar'
                else max_per_phrase_category
            )
            speaker_key = (cand.speaker_id, cand.category)
            phrase_key = (cand.phrase, cand.category)
            if speaker_counts[speaker_key] >= max_per_speaker_category:
                continue
            if phrase_counts[phrase_key] >= phrase_limit:
                continue
            selected.append(cand)
            speaker_counts[speaker_key] += 1
            phrase_counts[phrase_key] += 1
            if max_per_category > 0 and len(selected) >= max_per_category:
                break
        selected.sort(key=lambda c: (c.source_split, c.speaker_id, c.chapter_id, c.source_utt_id, c.start_sec))
        limited.extend(selected)
    return limited


def split_speakers(candidates: Sequence[Candidate], seed: int) -> Dict[str, str]:
    speakers = sorted({cand.speaker_id for cand in candidates})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n = len(speakers)
    if n == 0:
        return {}
    n_val = max(1, int(round(n * 0.05))) if n >= 3 else 0
    n_test = max(1, int(round(n * 0.05))) if n >= 3 else 0
    if n_val + n_test >= n:
        n_val = 1 if n >= 3 else 0
        n_test = 1 if n >= 3 else 0
    speaker_split = {}
    for i, speaker in enumerate(speakers):
        if i < n_val:
            speaker_split[speaker] = 'val'
        elif i < n_val + n_test:
            speaker_split[speaker] = 'test'
        else:
            speaker_split[speaker] = 'train'
    return speaker_split


def format_float(value: float, ndigits: int = 4) -> str:
    return ('{:.%df}' % ndigits).format(value)


def candidate_wav_relpath(candidate: Candidate) -> str:
    uid = stable_hash('{}:{:.3f}:{:.3f}:{}'.format(
        candidate.source_utt_id, candidate.start_sec, candidate.end_sec, candidate.category))
    name = '{}_{:.2f}_{:.2f}_{}.wav'.format(
        candidate.source_utt_id, candidate.start_sec, candidate.end_sec, uid)
    return str(Path('wavs') / candidate.category / name)


def candidate_to_row(candidate: Candidate) -> Dict[str, str]:
    return {
        'wav_path': candidate.wav_path,
        'label': '_unknown_',
        'category': candidate.category,
        'target': candidate.target,
        'source_dataset': 'LibriSpeech',
        'source_split': candidate.source_split,
        'source_utt_id': candidate.source_utt_id,
        'speaker_id': candidate.speaker_id,
        'chapter_id': candidate.chapter_id,
        'start_sec': format_float(candidate.start_sec),
        'end_sec': format_float(candidate.end_sec),
        'duration_sec': format_float(candidate.duration_sec),
        'words': ' '.join(candidate.words),
        'phones': ' '.join(candidate.phones),
        'normalized_phones': ' '.join(candidate.normalized_phones),
        'similarity': format_float(candidate.similarity),
        'edit_distance': str(candidate.edit_distance),
        'matched_rule': candidate.matched_rule,
        'blacklist_flag': candidate.blacklist_flag,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifests(out_dir: Path, candidates: Sequence[Candidate], seed: int) -> None:
    manifests_dir = out_dir / 'manifests'
    speaker_split = split_speakers(candidates, seed)
    rows = [candidate_to_row(cand) for cand in candidates]
    write_csv(manifests_dir / 'all.csv', MANIFEST_FIELDS, rows)
    for split in ('train', 'val', 'test'):
        split_rows = [
            candidate_to_row(cand)
            for cand in candidates
            if speaker_split.get(cand.speaker_id, 'train') == split
        ]
        write_csv(manifests_dir / '{}.csv'.format(split), MANIFEST_FIELDS, split_rows)


def write_reports(
    out_dir: Path,
    candidates: Sequence[Candidate],
    blacklist_hits: Sequence[Candidate],
    missing_audio: Sequence[Dict[str, str]],
    parse_errors: Sequence[Dict[str, str]],
    seed: int,
) -> None:
    reports_dir = out_dir / 'reports'
    count_rows = []
    counts = Counter(cand.category for cand in candidates)
    for category in sorted(set(TARGET_FOR_CATEGORY) | set(counts)):
        count_rows.append({'category': category, 'count': str(counts.get(category, 0))})
    write_csv(reports_dir / 'category_counts.csv', ['category', 'count'], count_rows)

    blacklist_rows = [candidate_to_row(cand) for cand in blacklist_hits]
    write_csv(reports_dir / 'blacklist_hits.csv', MANIFEST_FIELDS, blacklist_rows)

    missing_fields = ['source_split', 'source_utt_id', 'speaker_id', 'chapter_id', 'reason']
    write_csv(reports_dir / 'missing_audio.csv', missing_fields, missing_audio)

    parse_fields = ['source_split', 'source_utt_id', 'path', 'error']
    write_csv(reports_dir / 'parse_errors.csv', parse_fields, parse_errors)

    duration_rows = []
    for category in sorted(set(TARGET_FOR_CATEGORY) | {cand.category for cand in candidates}):
        durations = [cand.duration_sec for cand in candidates if cand.category == category]
        if durations:
            duration_rows.append({
                'category': category,
                'count': str(len(durations)),
                'min': format_float(min(durations)),
                'p50': format_float(statistics.median(durations)),
                'mean': format_float(statistics.mean(durations)),
                'p95': format_float(percentile(durations, 95)),
                'max': format_float(max(durations)),
            })
        else:
            duration_rows.append({
                'category': category,
                'count': '0',
                'min': '',
                'p50': '',
                'mean': '',
                'p95': '',
                'max': '',
            })
    write_csv(reports_dir / 'duration_stats.csv',
              ['category', 'count', 'min', 'p50', 'mean', 'p95', 'max'], duration_rows)

    rng = random.Random(seed)
    sample_rows = []
    for category in sorted({cand.category for cand in candidates}):
        items = [cand for cand in candidates if cand.category == category]
        rng.shuffle(items)
        for cand in items[: min(25, len(items))]:
            sample_rows.append(candidate_to_row(cand))
    write_csv(reports_dir / 'sample_check_list.csv', MANIFEST_FIELDS, sample_rows)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return math.nan
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def iter_progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def cut_audio(
    candidates: Sequence[Candidate],
    audio_index: Dict[str, Path],
    out_dir: Path,
    sample_rate: int,
    dry_run: bool,
    no_cut_audio: bool,
) -> None:
    for cand in candidates:
        relpath = candidate_wav_relpath(cand)
        cand.wav_path = relpath

    if dry_run or no_cut_audio:
        return

    import torch
    import torchaudio

    for cand in iter_progress(candidates, desc='Cutting audio', unit='clip'):
        src = audio_index.get(cand.source_utt_id)
        if src is None:
            continue
        dst = out_dir / cand.wav_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        waveform, sr = torchaudio.load(str(src))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
        start = max(0, int(round(cand.start_sec * sample_rate)))
        end = min(waveform.shape[1], int(round(cand.end_sec * sample_rate)))
        if end <= start:
            continue
        clip = waveform[:, start:end]
        torchaudio.save(str(dst), clip.cpu().to(torch.float32), sample_rate)


def parse_splits(text: str) -> List[str]:
    splits = []
    for item in re.split(r'[,\s]+', text.strip()):
        if item:
            splits.append(item)
    return splits


def collect_candidates(args, jobs: List[Tuple[str, str]], audio_index: Dict[str, Path]) -> ProcessResult:
    result = ProcessResult()
    audio_present_by_utt = {utt_id: True for utt_id in audio_index}
    task_args = [
        (
            job,
            args.min_sec,
            args.max_sec,
            args.pre_pad_sec,
            args.post_pad_sec,
            args.similarity_threshold,
            args.max_edit_distance,
            Path(job[0]).stem in audio_present_by_utt,
        )
        for job in jobs
    ]

    if args.num_workers <= 1:
        for packed in iter_progress(task_args, desc='Mining TextGrid', unit='utt'):
            merge_result(result, process_textgrid_job(*packed))
    else:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_textgrid_job, *packed) for packed in task_args]
            for future in iter_progress(
                as_completed(futures),
                total=len(futures),
                desc='Mining TextGrid',
                unit='utt',
            ):
                merge_result(result, future.result())
    return result


def merge_result(dst: ProcessResult, src: ProcessResult) -> None:
    dst.candidates.extend(src.candidates)
    dst.blacklist_hits.extend(src.blacklist_hits)
    dst.missing_audio.extend(src.missing_audio)
    dst.parse_errors.extend(src.parse_errors)


def validate_args(args) -> None:
    if args.min_sec <= 0 or args.max_sec <= 0 or args.min_sec > args.max_sec:
        raise ValueError('Invalid duration limits: min_sec={}, max_sec={}'.format(args.min_sec, args.max_sec))
    if args.max_edit_distance < 0:
        raise ValueError('--max_edit_distance must be non-negative')
    if args.max_per_category < 0:
        raise ValueError('--max_per_category must be non-negative')
    if args.max_per_speaker_category < 1:
        raise ValueError('--max_per_speaker_category must be >= 1')
    if args.max_per_phrase_category < 1:
        raise ValueError('--max_per_phrase_category must be >= 1')
    if args.max_per_hey_phrase_category < 1:
        raise ValueError('--max_per_hey_phrase_category must be >= 1')
    if args.num_workers < 1:
        raise ValueError('--num_workers must be >= 1')


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Build LibriSpeech phoneme-similar hard negatives from TextGrid alignments.')
    parser.add_argument('--alignment_root', required=True,
                        help='Root containing LibriSpeech alignment split directories.')
    parser.add_argument('--librispeech_root',
                        default='/mnt/vdb1/logic/Librispeech/LibriSpeech',
                        help='Root containing LibriSpeech audio split directories.')
    parser.add_argument('--out_dir', required=True,
                        help='Output directory for wavs, manifests, and reports.')
    parser.add_argument('--splits', default='train-clean-100',
                        help='Comma or whitespace separated LibriSpeech splits. Default: train-clean-100.')
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--min_sec', type=float, default=0.5)
    parser.add_argument('--max_sec', type=float, default=3.0)
    parser.add_argument('--pre_pad_sec', type=float, default=0.10)
    parser.add_argument('--post_pad_sec', type=float, default=0.15)
    parser.add_argument('--similarity_threshold', type=float, default=0.55)
    parser.add_argument('--max_edit_distance', type=int, default=2)
    parser.add_argument('--max_per_category', type=int, default=1000,
                        help='Maximum rows per category. Use 0 for unlimited.')
    parser.add_argument('--max_per_speaker_category', type=int, default=50,
                        help='Maximum rows per speaker/category after scoring. Default: 50.')
    parser.add_argument('--max_per_phrase_category', type=int, default=20,
                        help='Maximum rows per phrase/category after scoring. Default: 20.')
    parser.add_argument('--max_per_hey_phrase_category', type=int, default=5,
                        help='Phrase cap for hey_phoneme_similar to avoid only HE examples. Default: 5.')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dry_run', action='store_true',
                        help='Write manifests/reports but do not cut audio.')
    parser.add_argument('--no_cut_audio', action='store_true',
                        help='Skip wav writing while still producing manifests/reports.')
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)

    splits = parse_splits(args.splits)
    alignment_root = Path(args.alignment_root)
    librispeech_root = Path(args.librispeech_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unaligned = load_unaligned(alignment_root, splits)
    jobs = discover_textgrids(alignment_root, splits, unaligned)
    audio_index = build_audio_index(librispeech_root, splits)
    print('Found {} TextGrid files after skipping {} unaligned utterances.'.format(len(jobs), len(unaligned)))
    print('Indexed {} flac files.'.format(len(audio_index)))

    result = collect_candidates(args, jobs, audio_index)
    print('Mined {} raw accepted candidates, {} blacklist hits, {} missing audio, {} parse errors.'.format(
        len(result.candidates), len(result.blacklist_hits), len(result.missing_audio), len(result.parse_errors)))

    candidates = limit_candidates(
        result.candidates,
        args.max_per_category,
        args.seed,
        args.max_per_speaker_category,
        args.max_per_phrase_category,
        args.max_per_hey_phrase_category,
    )
    cut_audio(candidates, audio_index, out_dir, args.sample_rate, args.dry_run, args.no_cut_audio)
    write_manifests(out_dir, candidates, args.seed)
    write_reports(out_dir, candidates, result.blacklist_hits, result.missing_audio, result.parse_errors, args.seed)

    counts = Counter(cand.category for cand in candidates)
    print('Wrote {} final candidates to {}'.format(len(candidates), out_dir))
    for category in sorted(set(TARGET_FOR_CATEGORY) | set(counts)):
        print('  {}: {}'.format(category, counts.get(category, 0)))


if __name__ == '__main__':
    main()
