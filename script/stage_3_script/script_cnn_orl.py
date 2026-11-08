import os
import sys
import time
import argparse
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy

DATA_DIR = os.path.join(REPO_ROOT, 'data', 'stage_3_data') + os.sep
RESULT_DIR = os.path.join(REPO_ROOT, 'result', 'stage_3_result', 'ORL') + os.sep

trial_runs = {
    'orl_baseline':       dict(),
    'orl_depth_2conv':    dict(conv_channels=(32, 64)),
    'orl_depth_4conv':    dict(conv_channels=(32, 64, 128, 256)),
    'orl_kernel_5':       dict(kernel_size=5, padding=2),
    'orl_no_padding':     dict(padding=0),
    'orl_stride2_nopool': dict(stride=2, pooling=None),
    'orl_avg_pool':       dict(pooling='avg'),
    'orl_hidden_64':      dict(hidden_sizes=(64,)),
    'orl_hidden_512':     dict(hidden_sizes=(512,)),
    'orl_no_hidden':      dict(hidden_sizes=()),
    'orl_no_batchnorm':   dict(batch_norm=False),
    'orl_no_dropout':     dict(dropout=0.0),
    'orl_augment':        dict(augment=True),
    'orl_loss_nll':       dict(loss_name='nll'),
    'orl_loss_mse':       dict(loss_name='mse'),
    'orl_optimizer_sgd':  dict(optimizer_name='sgd'),
    'orl_sgd_lr_1e-3':    dict(optimizer_name='sgd', learning_rate=1e-3),
    'orl_sgd_lr_1e-4':    dict(optimizer_name='sgd', learning_rate=1e-4),
}


def save_run_log(path, name, configuration, seed, history, scores, train_time, num_params, device):
    with open(path, 'w') as f:
        f.write('Run: ' + name + '\n')
        f.write('Configuration: ' + str(configuration) + '\n')
        f.write(f'Seed: {seed}\nParameters: {num_params:,}\nDevice: {device}\nTraining time: {train_time:.1f}s\n\n')
        f.write('Training per epoch\n')
        for epoch, loss, acc in zip(history['epoch'], history['loss'], history['accuracy']):
            f.write(f'Epoch {epoch:<3}  Accuracy {acc:.4f}  Loss {loss:.4f}\n')
        f.write('\nTest results\n')
        f.write(f"Accuracy:           {scores['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {scores['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall:    {scores['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1:        {scores['f1_weighted']:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[21, 42, 7])
    parser.add_argument('--only', nargs='+', default=None)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--device', default=None, choices=['cuda', 'mps', 'cpu'])
    args = parser.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)

    loader = Dataset_Loader(dName='ORL', dDescription='ORL face dataset')
    loader.dataset_source_folder_path = DATA_DIR
    loader.dataset_source_file_name = 'ORL'
    data = loader.load()
    print('train X', data['train']['X'].shape, 'test X', data['test']['X'].shape)

    runs = {k: v for k, v in trial_runs.items() if args.only is None or k in args.only}
    summary = []

    for name, configuration in runs.items():
        per_seed = []
        for i, seed in enumerate(args.seeds):
            print(f'\n===== {name}  seed {seed} =====')
            model = Method_CNN_ORL(name, '', seed=seed, max_epoch=args.max_epoch, patience=args.patience, device=args.device,
                                   verbose=(i == 0), **configuration)
            model.data = data
            start = time.time()
            results = model.run()
            train_time = time.time() - start

            evaluator = Evaluate_Accuracy(name + ' evaluator', '')
            evaluator.data = results
            if i == 0:
                scores = evaluator.evaluate_weighted_suite(
                    plot_save_path=RESULT_DIR + name + '_convergence.png',
                    loss_name=configuration.get('loss_name', 'cross_entropy'))
                plt.close('all')
                save_run_log(RESULT_DIR + name + '_results.txt', name, configuration, seed,
                             results['history'], scores, train_time, model.num_params, model.device)
            else:
                scores = evaluator.evaluate_weighted_suite(plot_save_path=None)

            true_y = np.asarray(results['true_y']).reshape(-1)
            pred_y = np.asarray(results['pred_y']).reshape(-1)
            wrong = [f'{t + 1}->{p + 1}' for t, p in zip(true_y, pred_y) if t != p]
            print(f'{name} seed {seed}: misclassified (true->pred): {wrong}')
            with open(RESULT_DIR + 'misclassified.txt', 'a') as f:
                f.write(f'{name}\tseed {seed}\t{wrong}\n')

            scores['epochs'] = len(results['history']['epoch'])
            scores['time'] = train_time
            per_seed.append(scores)
            print(f"{name} seed {seed}: test acc {scores['accuracy']:.4f}  ({scores['epochs']} epochs, {train_time:.1f}s)")

        agg = {k: (np.mean([s[k] for s in per_seed]), np.std([s[k] for s in per_seed])) for k in per_seed[0]}
        summary.append((name, configuration, model.num_params, agg))

    lines = [f'# ORL CNN results (mean +- std over seeds {args.seeds})', '',
             '| Run | Change vs baseline | Params | Epochs | Accuracy | Precision (w) | Recall (w) | F1 (w) |',
             '|---|---|---|---|---|---|---|---|']
    for name, configuration, params, agg in summary:
        change = ', '.join(f'{k}={v}' for k, v in configuration.items()) or '-'
        cell = lambda k: f'{agg[k][0]:.4f} +- {agg[k][1]:.4f}'
        lines.append(f"| {name} | {change} | {params:,} | {agg['epochs'][0]:.0f} | {cell('accuracy')} | "
                     f"{cell('precision_weighted')} | {cell('recall_weighted')} | {cell('f1_weighted')} |")
    table = '\n'.join(lines)
    print('\n' + table)
    out = 'ablation_summary.md' if args.only is None else 'summary_' + '_'.join(runs) + '.md'
    if args.patience is not None:
        out = out.replace('.md', f'_patience{args.patience}.md')
    with open(RESULT_DIR + out, 'w') as f:
        f.write(table + '\n')


if __name__ == '__main__':
    main()