"""
Evaluate a trained KWS classifier on the testing split and recalibrate
thr_far05 from real test data (not validation). Writes the updated
thr_far05 back to the checkpoint for downstream inference.
"""

import argparse
import json
import os

import torch

import models  # noqa: F401
from data.CompanyKWS import CompanyKWSDataset
from train_kws_classifier import KWSClassifier, evaluate, input_strategy_from_crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--classifier', required=True,
                        help='Checkpoint produced by train_kws_classifier.py')
    parser.add_argument('--datadir', default=None,
                        help='Override CompanyKWS data dir (default: from ckpt)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8,
                        help='DataLoader worker processes for pos/neg loaders '
                             '(default: 8). Drop to 0 only for debugging.')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no_writeback', action='store_true',
                        help='Compute metrics but do not modify the checkpoint')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('Device:', device)

    print('Loading classifier checkpoint …')
    ckpt = torch.load(args.classifier, map_location=device)
    data_dir = args.datadir or ckpt['data_dir']
    speech_args = ckpt['speech_args']
    crop_strategy = speech_args.get('crop_strategy', 'pad')
    input_strategy = ckpt.get('train_input_strategy', input_strategy_from_crop(crop_strategy))

    print(f'Data dir: {data_dir}')
    print(f'Task    : {ckpt["task"]}')
    print(f'Classes : {ckpt["class_list"]}')
    print('Input   : crop_strategy={} -> {}  window={}ms  hop={}ms  agg={}  window_batch={}'.format(
        crop_strategy,
        input_strategy,
        ckpt.get('sliding_window_ms', 1000),
        ckpt.get('sliding_hop_ms', 250),
        ckpt.get('sliding_agg', 'max_logit'),
        ckpt.get('sliding_window_batch', 1024)))

    # rebuild dataset using the exact same speech_args
    ds = CompanyKWSDataset(data_dir, ckpt['task'], device.type == 'cuda', speech_args)

    # rebuild model
    model = KWSClassifier(ckpt['encoder_ckpt'], ckpt['n_classes'],
                          freeze_encoder=ckpt['freeze_encoder'],
                          head=ckpt.get('head', 'linear'),
                          head_hidden=ckpt.get('head_hidden', 128),
                          head_dropout=ckpt.get('head_dropout', 0.2)).to(device)
    if hasattr(model.preprocessing, 'mfcc'):
        model.preprocessing.mfcc.to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    print('\nEvaluating on testing split …')
    test_metrics = evaluate(model, ds, 'testing', args.batch_size, device,
                             num_workers=args.num_workers,
                             pin_memory=device.type == 'cuda',
                             input_strategy=input_strategy,
                             sliding_window_ms=ckpt.get('sliding_window_ms', 1000),
                             sliding_hop_ms=ckpt.get('sliding_hop_ms', 250),
                             sliding_agg=ckpt.get('sliding_agg', 'max_logit'),
                             sliding_window_batch=ckpt.get('sliding_window_batch', 1024))

    print('\n=== Test metrics ===')
    print(json.dumps({k: float(v) for k, v in test_metrics.items()
                      if isinstance(v, (int, float))}, indent=2))

    if not args.no_writeback:
        ckpt['thr_far05'] = test_metrics['thr_far05']
        ckpt['acc_far05'] = test_metrics['acc_far05']
        ckpt['aucROC']    = test_metrics['aucROC']
        ckpt['test_metrics'] = {k: float(v) for k, v in test_metrics.items()
                                 if isinstance(v, (int, float))}
        torch.save(ckpt, args.classifier)
        print(f'\n✓ Updated thr_far05={test_metrics["thr_far05"]:.4f} → {args.classifier}')


if __name__ == '__main__':
    main()
