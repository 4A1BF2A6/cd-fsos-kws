import argparse

parser = argparse.ArgumentParser(description='Train feature extractor')

#default values
default_model_name = 'e2e_conv'
default_encoding = 'DSCNNL'
default_model_prep = 'mfcc'

default_dataset = 'googlespeechcommand'
default_datadir = '' # FIXME: add here your path
default_split = 'GSC12'

parser.add_argument('--choose_cuda', type=str, default='0', 
                    help='trai/finetunen few shot learning: number of cuda(default: 0)')

# model args
parser.add_argument('--model.model_name', type=str, default=default_model_name, metavar='MODELNAME',
                    help="model name (default: {:s})".format(default_model_name))
parser.add_argument('--model.x_dim', type=str, default='1,49,10', metavar='XDIM',
                    help="dimensionality of input images (default: '1,28,28')")
parser.add_argument('--model.hid_dim', type=int, default=64, metavar='HIDDIM',
                    help="dimensionality of hidden layers (default: 64)")
parser.add_argument('--model.z_dim', type=int, default=64, metavar='ZDIM',
                    help="dimensionality of input images (default: 64)")
parser.add_argument('--model.encoding', type=str, default=default_encoding, metavar='MODELENC',
                    help="model encoding (default: {:s})".format(default_encoding))
parser.add_argument('--model.model_path', type=str, default="", metavar='MODELPATH',
                    help="location of pretrained model to evaluate (default: empty)")
parser.add_argument('--model.z_norm', action='store_true', default=True,
                    help="enable z_norm (default: True)")
parser.add_argument('--model.preprocessing', type=str, default=default_model_prep, metavar='MODELPREP',
                    help="model preprocessing (default: {:s})".format(default_model_prep))                   


# train args - used for train
parser.add_argument('--train.epochs', type=int, default=10000, metavar='NEPOCHS',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--train.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='optimization method (default: Adam)')
parser.add_argument('--train.learning_rate', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--train.decay_every', type=int, default=20, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
parser.add_argument('--train.weight_decay', type=float, default=0.0, metavar='WD',
                    help="weight decay (default: 0.0)")
parser.add_argument('--train.patience', type=int, default=20, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
parser.add_argument('--train.batch_size', type=int, default=128, metavar='BATCHSIZE',
                    help='batchsize when training with minibatch gradient descent (default: 128)')

# episodic train options - used for train
parser.add_argument('--train.n_episodes', type=int, default=200, 
                    help='train few shot learning: number of episodes  (default: 200)')
parser.add_argument('--train.n_way', type=int, default=12, 
                    help='train few shot learning: number of subclasses (default: 12)')
parser.add_argument('--train.n_support', type=int, default=0, 
                    help='train few shot learning: number of support samples (default: 5)')
parser.add_argument('--train.n_query', type=int, default=5, 
                    help='train few shot learning: number of query samples (default: 5)')
parser.add_argument('--train.n_way_u', type=int, default=0, 
                    help='train few shot learning: number of unknown subclasses (default: 0)')

# loss settings - used for train
parser.add_argument('--train.loss', type=str, default='metric',
                    help='FIXME')
parser.add_argument('--train.distance', type=str, default='euclidean',
                    help='FIXME')
parser.add_argument('--train.margin', type=float, default=0.5, 
                    help='FIXME')


# log args
default_fields = 'loss,acc'
parser.add_argument('--log.fields', type=str, default=default_fields, metavar='FIELDS',
                    help="fields to monitor during training (default: {:s})".format(default_fields))
default_exp_dir = 'results'
parser.add_argument('--log.exp_dir', type=str, default=default_exp_dir, metavar='EXP_DIR',
                    help="directory where experiments should be saved (default: {:s})".format(default_exp_dir))
parser.add_argument('--data.cuda', action='store_true', help="run in CUDA mode (default: False)")


# speech data args
parser.add_argument('--speech.dataset', type=str, default=default_dataset, metavar='DS',
                    help="data set name (default: {:s})".format(default_dataset))
parser.add_argument('--speech.task', type=str, default=default_split, metavar='SP',
                    help="split name (default: {:s})".format(default_split))
parser.add_argument('--speech.default_datadir', type=str, default=default_datadir, metavar='DIR',
                    help="path to the dataset")
parser.add_argument('--speech.include_silence', action='store_true', help="one of the classes out of n should be silence (default: False)")
parser.add_argument('--speech.include_unknown', action='store_true', help="one of the classes out of n should be unknown (default: False)")
parser.add_argument('--speech.sample_rate', type=int, default=16000, help='desired sampling rate of the input')
parser.add_argument('--speech.clip_duration', type=int, default=1000, help='clip duration in milliseconds')
parser.add_argument('--speech.time_shift', type=int, default=100, help='time shift the audio in milliseconds')
parser.add_argument('--speech.bg_volume', type=float, default=0.1, help='background volumen to mix in between 0 and 1')
parser.add_argument('--speech.bg_frequency', type=float, default=1.0, help='Amount of samples that should be mixed with background noise (between 0 and 1)')
parser.add_argument('--speech.num_silence', type=int, default=1000, help='Number of silence samples to generate')
parser.add_argument('--speech.foreground_volume', type=float, default=1)
parser.add_argument('--speech.channel', type=str, default='ch07',
                    help='which channel wav to load for CompanyKWS, e.g. ch01/ch07 (default: ch07)')
parser.add_argument('--speech.crop_strategy', type=str, default='center',
                    choices=['center', 'energy', 'pad'],
                    help='how to crop variable-length audio to clip_duration: '
                         'center (geometric middle), energy (1s window with max RMS energy), '
                         'or pad (keep native variable length up to --speech.max_duration_ms; '
                         'longer samples are energy-cropped). '
                         'Only used by CompanyKWS wrapper. Default: center')
parser.add_argument('--speech.max_duration_ms', type=int, default=3100,
                    help="Upper bound for variable-length samples under crop_strategy='pad'. "
                         'Samples longer than this get energy-cropped down to it; '
                         'shorter samples are kept native (zero-padded only at collate time). '
                         'Default: 3100 (CompanyKWS P99 ≈ 3.1s)')
parser.add_argument('--speech.merge_val', type=str, default='none',
                    choices=['none', 'train', 'test'],
                    help='how to absorb the validation split when it is otherwise unused: '
                         'none keeps it standalone; train merges into training; '
                         'test merges into testing. Only used by CompanyKWS wrapper. Default: none')
parser.add_argument('--speech.gsc_unknown_dir', type=str, default=None,
                    help='GSC root dir (speech_commands_v0.02/); if set, injects GSC words as '
                         'harder _unknown_ negatives into CompanyKWS testing split')
parser.add_argument('--speech.gsc_unknown_words', type=str,
                    default='backward,forward,visual,follow,learn,bed,bird,cat,dog',
                    help='comma-separated GSC words to use as unknown negatives '
                         '(only used when --speech.gsc_unknown_dir is set)')
parser.add_argument('--speech.gsc_unknown_splits', type=str, default='test',
                    choices=['test', 'all'],
                    help='test: inject GSC unknowns into testing only (default); '
                         'all: split 80/20 into training+testing so the support-set '
                         '_unknown_ prototype is built from human speech')
parser.add_argument('--speech.librispeech_dir', type=str, default=None,
                    help='LibriSpeech root (e.g. .../train-clean-100/). If set, '
                         'inject random 1s slices of continuous English speech as '
                         '_unknown_ negatives (always 80/20 train/test split)')
parser.add_argument('--speech.librispeech_samples_per_file', type=int, default=2,
                    help='How many random 1s slices to extract from each LibriSpeech '
                         'utterance (default: 2)')
parser.add_argument('--speech.librispeech_max_files', type=int, default=0,
                    help='Cap on number of LibriSpeech .flac files to use; 0 = no cap '
                         '(default: 0). Useful to control _unknown_ pool size.')
parser.add_argument('--speech.gsc_noise_dir', type=str, default=None,
                    help='GSC _background_noise_ dir; if set, randomly slice these '
                         'long background recordings into variable-length _unknown_ '
                         'negatives. E.g. <gsc>/speech_commands_v0.02/_background_noise_/')
parser.add_argument('--speech.gsc_noise_samples_per_file', type=int, default=50,
                    help='Random slices to extract from each GSC noise wav (default: 50). '
                         'Each file is ~60s so 50 slices gives ~50× more diversity '
                         'than per LibriSpeech utterance.')
parser.add_argument('--speech.bg_duration_min_ms', type=int, default=500,
                    help='Lower bound on random duration sampled for _unknown_ '
                         'background / LibriSpeech / GSC-noise slices, in ms. '
                         'Default 500 (matches wake-word P1).')
parser.add_argument('--speech.bg_duration_max_ms', type=int, default=3100,
                    help='Upper bound for random _unknown_ slice duration, in ms. '
                         'Default 3100 (matches --speech.max_duration_ms / wake P99).')

parser.add_argument('--speech.include_noise', action='store_true', default=True, help="one of the classes out of n should be unknown (default: False)")
parser.add_argument('--speech.noise_snr', type=int, default=5, help='time shift the audio in milliseconds')
parser.add_argument('--speech.noise_frequency', type=float, default=0.95, help='Amount of samples that should be mixed with background noise (between 0 and 1)')


# feature extraction
parser.add_argument('--speech.window_size', type=int, default=40)
parser.add_argument('--speech.window_stride', type=int,default=20)
parser.add_argument('--speech.num_features', type=int, default=10, help='Number of mfcc features to feed the model')
parser.add_argument('--speech.n_mfcc', type=int, default=40, help='Number of mfcc features to compute')

# classifier options - used for few-shot querying
parser.add_argument('--fsl.classifier', type=str, default='ncm', 
                    help='Type of the classifier')
parser.add_argument('--fsl.test.n_way', type=int, default=12, 
                    help='test few shot learning: number of subclasses (default: 12)')
parser.add_argument('--fsl.test.n_support', type=int, default=5, 
                    help='test few shot learning: number of support samples (default: 5)')
parser.add_argument('--fsl.test.n_episodes', type=int, default=100, 
                    help='test few shot learning: number of episodes  (default: 100)')
parser.add_argument('--fsl.test.fixed_silence_unknown', action='store_true',
                    help='force unknown and silence class to be present in every episode (default: False)')
parser.add_argument('--fsl.test.batch_size', type=int, default=128, 
                    help='test few shot batch size  (default: 128)')

# adapting settings
parser.add_argument('--adapting.n_episodes', type=int, default=50, 
                    help='adapting target few shot learning: number of episodes  (default: 50)')
parser.add_argument('--adapting.optim_method', type=str, default='Adam', metavar='OPTIM',
                    help='adapting optimization method (default: Adam)')
parser.add_argument('--adapting.learning_rate', type=float, default=0.1, metavar='LR',
                    help='adapting learning rate (default: 0.0001)')
parser.add_argument('--adapting.decay_every', type=int, default=5, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
parser.add_argument('--adapting.weight_decay', type=float, default=0.001, metavar='WD',
                    help="adapting weight decay (default: 0.0)")
parser.add_argument('--adapting.patience', type=int, default=20, metavar='PATIENCE',
                    help='number of epochs to wait before validation improvement (default: 1000)')
parser.add_argument('--adapting.para_select', type=str, default='cn_alpha',
                    help='adapting para select (option: cn_alpha, cn_alpha_trans, cn_alpha_lft, cn_alpha_trans_lft, ridge_regress)')
parser.add_argument('--adapting.episodes', type=int, default=50, 
                    help='number of adapting episodes  (default: 50)')


# finetune2 settings(prototype recalibration use ridge regression)
parser.add_argument('--adapting.optim_method_2', type=str, default='Adam', metavar='OPTIM',
                    help='adapting optimization method (default: Adam)')
parser.add_argument('--adapting.learning_rate_2', type=float, default=0.01, metavar='LR',
                    help='adapting learning rate (default: 0.0001)')
parser.add_argument('--adapting.decay_every_2', type=int, default=5, metavar='LRDECAY',
                    help='number of epochs after which to decay the learning rate')
parser.add_argument('--adapting.weight_decay_2', type=float, default=0.0005, metavar='WD',
                    help="adapting weight decay (default: 0.0)")

# custom keyword adapter settings
parser.add_argument('--adapting.cka-ad-type', type=str, choices=['residual', 'serial', 'none'], default='residual', metavar='TSA_AD_TYPE',
                    help="adapter type")
parser.add_argument('--adapting.cka-ad-form', type=str, choices=['matrix', 'vector', 'none'], default='matrix', metavar='TSA_AD_FORM',
                    help="adapter form")
parser.add_argument('--adapting.cka-opt', type=str, choices=['delta', 'beta', 'delta+beta'], default='delta+beta', metavar='TSA_OPT',
                    help="custom keyword adaptation option")
parser.add_argument('--adapting.cka-init', type=str, choices=['random', 'eye'], default='eye', metavar='TSA_INIT',
                    help="initialization for adapter")

# prototype reprojection settings
parser.add_argument('--querying.prototype_reprojection', action='store_true', help="eval with ridge regression (default: False)")
