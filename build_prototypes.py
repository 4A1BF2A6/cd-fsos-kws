"""
Build final prototypes from ALL training samples using the best-episode
CKA-adapted model saved by target_adapting_querying.py.

Overwrites the 'muK' in best_prototypes.pt with full-data prototypes.

Usage:
    python build_prototypes.py \
        --adapted_model results/Pretrain_DSCNN_MSWC/best_adapted_model.pt \
        --prototypes    results/Pretrain_DSCNN_MSWC/best_prototypes.pt

Optional overrides (use original values from best_prototypes.pt by default):
    --datadir   /new/path/to/wakeword_dataset/
    --channel   ch07
    --batch_size 128
    --cpu
"""

import argparse
import torch
import torch.nn.functional as F

import models  # noqa: F401 — registers model builders
from models.utils import get_model
from models.CKAs_module import ReprModel_cka


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_model(adapted_model_path, ckpt, device):
    model_opt    = ckpt['model_opt']
    adapting_opt = ckpt['adapting_opt']
    criterion    = ckpt['criterion']
    x_dim        = ckpt['x_dim']

    full_opt = dict(adapting_opt)
    full_opt['data.cuda'] = device.type == 'cuda'

    base = get_model(model_opt)
    base.eval()
    base.to(device)
    if hasattr(base.preprocessing, 'mfcc'):
        base.preprocessing.mfcc.to(device)

    model = ReprModel_cka(base, full_opt, criterion, x_dim)
    state = torch.load(adapted_model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    if hasattr(model.preprocessing, 'mfcc'):
        model.preprocessing.mfcc.to(device)
    model.eval()
    return model


@torch.no_grad()
def compute_prototypes(model, ds, class_list, batch_size, device):
    """Return (muK, labels) where muK is L2-normalised, shape (N, D)."""
    prototypes = []
    labels_out = []

    for word in class_list:
        loader = ds.get_iid_dataloader(
            'training', batch_size,
            class_list=[word],
            include_silence=True,
            include_unknown=True,
        )
        embs = []
        for batch in loader:
            x = batch['data'].to(device)
            embs.append(model.get_embeddings(x).cpu())
        if not embs:
            print('[WARNING] No training samples found for class: {}'.format(word))
            continue
        class_emb = torch.cat(embs, dim=0)          # (N_i, D)
        proto = F.normalize(class_emb.mean(dim=0), dim=-1)  # (D,)
        prototypes.append(proto)
        labels_out.append(word)
        print('  {} : {} samples → prototype shape {}'.format(
            word, class_emb.size(0), proto.shape))

    muK = torch.stack(prototypes, dim=0)             # (N_classes, D)
    return muK, labels_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Rebuild prototypes from full training set using best adapted model')
    parser.add_argument('--adapted_model', required=True,
                        help='best_adapted_model.pt saved by target_adapting_querying.py')
    parser.add_argument('--prototypes', required=True,
                        help='best_prototypes.pt (will be overwritten)')
    parser.add_argument('--datadir', default=None,
                        help='Override data_dir (default: use value from best_prototypes.pt)')
    parser.add_argument('--channel', default=None,
                        help='Override audio channel, e.g. ch07 (default: from best_prototypes.pt)')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('Device:', device)

    # load metadata
    ckpt = torch.load(args.prototypes, map_location='cpu')
    speech_args = dict(ckpt['speech_args'])
    data_dir    = args.datadir  or ckpt['data_dir']
    task        = ckpt['task']
    cuda        = not args.cpu and ckpt.get('cuda', False)

    if args.channel:
        speech_args['channel'] = args.channel

    print('Data dir :', data_dir)
    print('Task     :', task)
    print('Channel  :', speech_args.get('channel', 'ch07'))
    print('Previous prototype was from ep={}, accuracy_pos={:.4f}'.format(
        ckpt.get('ep', '?'), ckpt.get('accuracy_pos', float('nan'))))

    # rebuild model
    print('\nLoading adapted model …')
    model = build_model(args.adapted_model, ckpt, device)

    # rebuild dataset (same args as original eval run)
    print('Loading dataset …')
    from data.CompanyKWS import CompanyKWSDataset
    ds = CompanyKWSDataset(data_dir, task, cuda, speech_args)

    # use same class_list order as original classifier
    class_list = ckpt['class_list']
    print('\nComputing prototypes from ALL training samples …')
    muK, labels_out = compute_prototypes(model, ds, class_list, args.batch_size, device)

    print('\nNew muK shape:', muK.shape)

    # rebuild word_to_index to match labels_out (in case a class had no samples)
    word_to_index = {l: i for i, l in enumerate(labels_out)}

    # overwrite prototypes file
    ckpt['muK']          = muK
    ckpt['class_list']   = labels_out
    ckpt['word_to_index'] = word_to_index
    ckpt['full_train_proto'] = True   # flag: this is a full-data prototype
    torch.save(ckpt, args.prototypes)
    print('\nSaved updated prototypes to', args.prototypes)
    print('Classes:', labels_out)


if __name__ == '__main__':
    main()
