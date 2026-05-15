"""
Select ARPAbet hard-negative candidates from CMUdict.

This script uses CMUdict as a pronunciation dictionary, not as an audio
dataset. It scores candidate phrases such as "HEY CANDY" against target
wake-word pronunciations such as "HEY CAMY", then filters out unsafe variants
with a blacklist.

Example:
    python select_cmudict_hard_negatives.py \
        --cmudict /path/to/cmudict.dict \
        --target hey_camy --target hey_reco \
        --top_k 80 --out hard_negatives.csv
"""

import argparse
import csv
import re
from collections import defaultdict


DEFAULT_TARGETS = {
    'hey_camy': 'HH EY K AE M IY',
    'hey_reco': 'HH EY R IY K OW',
}

DEFAULT_BLACKLIST = {
    'hey_camy': {
        'CAMY', 'CAMI', 'CAMMY', 'KAMI', 'KAMY', 'KEMI', 'KEMY',
        'KAMEY', 'KAMEE', 'CAMEY',
    },
    'hey_reco': {
        'RECO', 'REKO', 'RICO', 'RIKO', 'RICKO', 'RECKO', 'REECO',
    },
}

PHONE_RE = re.compile(r'\d')
ALT_PRON_RE = re.compile(r'\(\d+\)$')


def strip_stress(phone):
    return PHONE_RE.sub('', phone)


def normalize_word(word):
    word = ALT_PRON_RE.sub('', word.upper())
    return word


def parse_phones(text, keep_stress=False):
    phones = text.strip().split()
    if keep_stress:
        return phones
    return [strip_stress(p) for p in phones]


def read_cmudict(path, keep_stress=False):
    entries = defaultdict(list)
    with open(path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;') or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = normalize_word(parts[0])
            phones = [p if keep_stress else strip_stress(p) for p in parts[1:]]
            entries[word].append(phones)
    return dict(entries)


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            sub_cost = 0 if x == y else 1
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + sub_cost,
            ))
        prev = cur
    return prev[-1]


def load_word_list(path):
    if not path:
        return set()
    words = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            for item in re.split(r'[,\s]+', line):
                if item:
                    words.add(item.upper())
    return words


def parse_inline_words(text):
    if not text:
        return set()
    return {w.strip().upper() for w in text.split(',') if w.strip()}


def word_allowed(word, args):
    if len(word) < args.min_word_len or len(word) > args.max_word_len:
        return False
    if args.alpha_only and not re.fullmatch(r"[A-Z']+", word):
        return False
    if args.exclude_possessive and word.endswith("'S"):
        return False
    return True


def score_candidates(cmudict, target_name, target_phones, prefix_phones, blacklist, args):
    rows = []
    for word, prons in cmudict.items():
        if word in blacklist:
            continue
        if not word_allowed(word, args):
            continue

        best = None
        for pron in prons:
            cand_phones = prefix_phones + pron
            dist = levenshtein(target_phones, cand_phones)
            norm = dist / float(max(len(target_phones), len(cand_phones), 1))
            if best is None or (dist, norm, len(cand_phones)) < (best[0], best[1], best[2]):
                best = (dist, norm, len(cand_phones), cand_phones)

        if best is None:
            continue
        dist, norm, cand_len, cand_phones = best
        if dist < args.min_distance or dist > args.max_distance:
            continue
        if norm < args.min_norm_distance or norm > args.max_norm_distance:
            continue

        rows.append({
            'target': target_name,
            'candidate': '{} {}'.format(args.prefix.upper(), word),
            'word': word,
            'distance': dist,
            'norm_distance': round(norm, 4),
            'target_phones': ' '.join(target_phones),
            'candidate_phones': ' '.join(cand_phones),
        })

    rows.sort(key=lambda r: (r['distance'], r['norm_distance'], r['word']))
    return rows[:args.top_k] if args.top_k > 0 else rows


def build_targets(args, keep_stress=False):
    targets = {}
    for name in args.target:
        if name not in DEFAULT_TARGETS:
            raise ValueError('Unknown built-in target: {}'.format(name))
        targets[name] = parse_phones(DEFAULT_TARGETS[name], keep_stress=keep_stress)
    for item in args.target_pron:
        if '=' not in item:
            raise ValueError('--target_pron must be NAME=PHONE PHONE ..., got {}'.format(item))
        name, phones = item.split('=', 1)
        targets[name.strip()] = parse_phones(phones, keep_stress=keep_stress)
    return targets


def main():
    parser = argparse.ArgumentParser(
        description='Select CMUdict ARPAbet hard-negative candidates for KWS.')
    parser.add_argument('--cmudict', required=True,
                        help='Path to CMUdict, e.g. cmudict.dict or cmudict-0.7b.')
    parser.add_argument('--target', action='append', default=[],
                        choices=sorted(DEFAULT_TARGETS.keys()),
                        help='Built-in target wake phrase. Can be repeated.')
    parser.add_argument('--target_pron', action='append', default=[],
                        help='Custom target pronunciation: NAME="HH EY K AE M IY". Can be repeated.')
    parser.add_argument('--prefix', default='HEY',
                        help='Prefix word prepended to every candidate word. Default: HEY.')
    parser.add_argument('--prefix_pron', default=None,
                        help='Override prefix pronunciation. Default: lookup --prefix in CMUdict.')
    parser.add_argument('--blacklist', default=None,
                        help='Optional file of words to exclude, one word per line or comma-separated.')
    parser.add_argument('--extra_blacklist', default='',
                        help='Comma-separated extra words to exclude.')
    parser.add_argument('--no_default_blacklist', action='store_true',
                        help='Do not apply built-in unsafe variant blacklists.')
    parser.add_argument('--min_distance', type=int, default=2,
                        help='Minimum raw phone edit distance. Default: 2.')
    parser.add_argument('--max_distance', type=int, default=5,
                        help='Maximum raw phone edit distance. Default: 5.')
    parser.add_argument('--min_norm_distance', type=float, default=0.0,
                        help='Minimum normalized phone edit distance. Default: 0.0.')
    parser.add_argument('--max_norm_distance', type=float, default=0.7,
                        help='Maximum normalized phone edit distance. Default: 0.7.')
    parser.add_argument('--min_word_len', type=int, default=2)
    parser.add_argument('--max_word_len', type=int, default=12)
    parser.add_argument('--top_k', type=int, default=100,
                        help='Candidates per target. Use 0 for all.')
    parser.add_argument('--out', default='',
                        help='CSV output path. If omitted, print CSV to stdout.')
    parser.add_argument('--keep_stress', action='store_true',
                        help='Keep ARPAbet stress digits during scoring.')
    parser.add_argument('--alpha_only', action='store_true', default=True,
                        help='Only keep alphabetic/apostrophe words. Default: true.')
    parser.add_argument('--allow_non_alpha', dest='alpha_only', action='store_false')
    parser.add_argument('--exclude_possessive', action='store_true', default=True,
                        help="Exclude words ending in 'S. Default: true.")
    parser.add_argument('--allow_possessive', dest='exclude_possessive', action='store_false')
    args = parser.parse_args()

    if not args.target and not args.target_pron:
        args.target = ['hey_camy', 'hey_reco']

    cmudict = read_cmudict(args.cmudict, keep_stress=args.keep_stress)
    targets = build_targets(args, keep_stress=args.keep_stress)

    if args.prefix_pron:
        prefix_phones = parse_phones(args.prefix_pron, keep_stress=args.keep_stress)
    else:
        prefix_word = args.prefix.upper()
        if prefix_word not in cmudict:
            raise ValueError('Prefix word {} not found in CMUdict; pass --prefix_pron.'.format(prefix_word))
        prefix_phones = cmudict[prefix_word][0]

    file_blacklist = load_word_list(args.blacklist)
    inline_blacklist = parse_inline_words(args.extra_blacklist)

    all_rows = []
    for target_name, target_phones in targets.items():
        blacklist = set(file_blacklist) | set(inline_blacklist)
        if not args.no_default_blacklist:
            blacklist |= DEFAULT_BLACKLIST.get(target_name, set())
        all_rows.extend(score_candidates(
            cmudict, target_name, target_phones, prefix_phones, blacklist, args))

    fieldnames = [
        'target', 'candidate', 'word', 'distance', 'norm_distance',
        'target_phones', 'candidate_phones',
    ]
    if args.out:
        with open(args.out, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    else:
        writer = csv.DictWriter(__import__('sys').stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == '__main__':
    main()
