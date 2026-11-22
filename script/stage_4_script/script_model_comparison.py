
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from script.stage_4_script import script_rnn_classification as classification
from script.stage_4_script import script_rnn_generation as generation

RESULT_DIR = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result' / 'model_comparison_4.5'

# Keys that Method_RNN takes directly; any other key overrides a training argument
# (hidden_dim, embedding_dim, dropout, lr, batch_size, max_epoch, patience).
CLASSIFIER_MODEL_KEYS = {'cell', 'bidirectional', 'num_layers'}

CLASSIFICATION_RUNS = {
    'rnn': dict(cell='rnn'),
    'lstm': dict(cell='lstm'),
    'gru': dict(cell='gru'),
    'bilstm': dict(cell='lstm', bidirectional=True),
    'lstm_2layer': dict(cell='lstm', num_layers=2),
    'gru_hidden256': dict(cell='gru', hidden_dim=256),
}

# Every key overrides a script_rnn_generation argument (cell, num_layers, hidden_dim,
# embedding_dim, dropout, lr, max_epoch, checkpoint, temperature, top_k).
GENERATION_RUNS = {
    'rnn': dict(cell='rnn'),
    'lstm': dict(cell='lstm'),
    'gru': dict(cell='gru'),
    'gru_1layer': dict(cell='gru', num_layers=1),
    'lstm_dropout50': dict(cell='lstm', dropout=0.5),
    'lstm_best_ckpt': dict(cell='lstm', checkpoint='best'),
}

# Fixed colour per slot, in order, so a run keeps its colour whichever subset is plotted.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
SURFACE, INK, INK_SECONDARY, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e3e2dd'


def with_overrides(base_args, overrides):
    args = copy.copy(base_args)
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise ValueError(f'Unknown setting {key!r}')
        setattr(args, key, value)
    return args


def describe(configuration):
    return ', '.join(f'{k}={v}' for k, v in configuration.items())


def merge_results(path, new_results):

    results = json.loads(path.read_text()) if path.exists() else {}
    results.update(new_results)
    path.write_text(json.dumps(results, indent=2, default=float))
    return results


def plot_comparison(results, order, panels, title, path):

    names = [n for n in order if n in results] + [n for n in results if n not in order]
    # Runs that train identically (e.g. only the generation checkpoint differs) would hide
    # each other, so they share one line and one legend entry.
    labels = {}
    for name in names:
        twin = next((n for n in labels if results[n]['history'] == results[name]['history']), None)
        if twin is None:
            labels[name] = name
        else:
            labels[twin] += ' / ' + name
    names = list(labels)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 4.5), facecolor=SURFACE)
    for ax, (key, label, log_scale) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for name in names:
            slot = order.index(name) if name in order else len(order)
            history = results[name]['history']
            ax.plot(history['epoch'], history[key], color=SERIES_COLORS[slot % len(SERIES_COLORS)],
                    linewidth=2, marker='o', markersize=3, label=labels[name])
        if log_scale:
            ax.set_yscale('log')
        ax.set_title(label, color=INK, fontsize=11, loc='left')
        ax.set_xlabel('Epoch', color=INK_SECONDARY)
        ax.set_ylabel(label, color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY)
        ax.grid(True, which='both', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color(GRID)
    axes[-1].legend(frameon=False, labelcolor=INK, loc='center left', bbox_to_anchor=(1.01, 0.5))
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def compare_classification(args):
    out_dir = RESULT_DIR / 'classification'
    out_dir.mkdir(parents=True, exist_ok=True)
    base = classification.build_parser().parse_args(
        ['--data-root', args.data_root, '--device', args.device, '--seed', str(args.seed)])
    if args.max_epoch is not None:
        base.max_epoch = args.max_epoch
    data = classification.Dataset_Loader(data_root=base.data_root).load(
        val_fraction=base.val_fraction, seed=base.seed)
    print(f"train {len(data['train'])}, val {len(data['val'])}, test {len(data['test'])}, "
          f"vocabulary {len(data['vocabulary'])}")

    new_results = {}
    for name in args.classification_runs:
        configuration = CLASSIFICATION_RUNS[name]
        model_settings = {k: v for k, v in configuration.items() if k in CLASSIFIER_MODEL_KEYS}
        run_args = with_overrides(base, {k: v for k, v in configuration.items()
                                         if k not in CLASSIFIER_MODEL_KEYS})
        scores, history = classification.train_and_evaluate(
            name, model_settings, data, run_args, result_dir=out_dir, return_history=True)
        best = history['epoch'].index(scores['best_epoch'])
        new_results[name] = dict(scores, settings=describe(configuration), history=history,
                                 val_accuracy_at_best=history['val_accuracy'][best],
                                 device=args.device, seed=args.seed)
        # Save after every run so a long comparison can be interrupted without losing work.
        results = merge_results(out_dir / 'comparison.json', new_results)

    order = list(CLASSIFICATION_RUNS)
    lines = ['# Stage 4.5 sentiment classification: model comparison (test set, 25,000 reviews)', '',
             f'Train {len(data["train"])} / validation {len(data["val"])} split of the training set, '
             f'vocabulary {len(data["vocabulary"])}. Unlisted settings: embedding {base.embedding_dim}, '
             f'hidden {base.hidden_dim}, 1 layer, dropout {base.dropout}, Adam lr {base.lr}, '
             f'batch {base.batch_size}, early stopping patience {base.patience}. '
             'Test metrics use the epoch with the lowest validation loss.', '',
             '| Run | Settings | Params | Best epoch | Val accuracy | Test accuracy | Precision | Recall | F1 | Train time | Device |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for name in [n for n in order if n in results] + [n for n in results if n not in order]:
        s = results[name]
        lines.append(f"| {name} | {s['settings']} | {s['num_params']:,} | {s['best_epoch']} / {s['epochs_run']} | "
                     f"{s['val_accuracy_at_best']:.4f} | {s['accuracy']:.4f} | {s['precision']:.4f} | "
                     f"{s['recall']:.4f} | {s['f1']:.4f} | {s['train_time']:.0f}s | {s['device']} |")
    table = '\n'.join(lines)
    print('\n' + table)
    (out_dir / 'comparison.md').write_text(table + '\n')
    plot_comparison(results, order,
                    [('val_loss', 'Validation loss', False), ('val_accuracy', 'Validation accuracy', False)],
                    'Sentiment classification: validation curves by model',
                    out_dir / 'comparison_learning_curves.png')


def story_statistics(stories):
    count = len(stories)
    return {'story_count': count,
            'exact_copies': sum(bool(s['exact_copy']) for s in stories),
            'mean_nearest_similarity': sum(s['nearest_similarity'] for s in stories) / count,
            'mean_3gram_in_training': sum(s['3gram_in_training'] for s in stories) / count}


def compare_generation(args):
    out_dir = RESULT_DIR / 'generation'
    out_dir.mkdir(parents=True, exist_ok=True)
    base = generation.build_parser().parse_args(
        ['--data-root', args.data_root, '--device', args.device, '--seed', str(args.seed)])
    if args.max_epoch is not None:
        base.max_epoch = args.max_epoch
    corpus = generation.load_corpus(base)

    new_results = {}
    for name in args.generation_runs:
        configuration = GENERATION_RUNS[name]
        print(f'\n===== {name}: {configuration} =====')
        result = generation.train_and_generate(name, with_overrides(base, configuration), corpus,
                                               result_dir=out_dir)
        history = result['history']
        new_results[name] = dict(result, settings=describe(configuration),
                                 final_train_perplexity=history['train_perplexity'][-1],
                                 final_val_perplexity=history['val_perplexity'][-1],
                                 best_val_perplexity=min(history['val_perplexity']),
                                 device=args.device, seed=args.seed,
                                 **story_statistics(result['stories']))
        results = merge_results(out_dir / 'comparison.json', new_results)

    order = list(GENERATION_RUNS)
    names = [n for n in order if n in results] + [n for n in results if n not in order]
    lines = ['# Stage 4.5 joke generation: model comparison', '',
             f'Unlisted settings: embedding {base.embedding_dim}, hidden {base.hidden_dim}, '
             f'{base.num_layers} layers, dropout {base.dropout}, Adam lr {base.lr}, batch {base.batch_size}, '
             f'{base.max_epoch} epochs, stories generated from the final epoch. Story columns cover every '
             f'generated story (greedy plus {base.samples} samples at temperature {base.temperature}, '
             f'top-k {base.top_k}, for each start).', '',
             '| Run | Settings | Params | Final train ppl | Best val ppl (epoch) | Final val ppl | '
             'Generated from epoch | Exact copies | Mean nearest sim. | Mean 3-grams in train | Train time | Device |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for name in names:
        r = results[name]
        lines.append(f"| {name} | {r['settings']} | {r['num_params']:,} | {r['final_train_perplexity']:.2f} | "
                     f"{r['best_val_perplexity']:.2f} ({r['best_epoch']}) | {r['final_val_perplexity']:.2f} | "
                     f"{r['generated_from_epoch']} | {r['exact_copies']} / {r['story_count']} | "
                     f"{r['mean_nearest_similarity']:.2f} | {r['mean_3gram_in_training']:.2f} | "
                     f"{r['train_time']:.0f}s | {r['device']} |")
    lines += ['', '## Greedy story per start', '', '| Start | Run | Story | Exact copy | 3-grams in train |',
              '|---|---|---|---|---|']
    greedy = {name: {s['start']: s for s in results[name]['stories'] if s['decoding'] == 'greedy'}
              for name in names}
    starts = list(dict.fromkeys(start for name in names for start in greedy[name]))
    for start in starts:
        for name in names:
            s = greedy[name].get(start)
            if s is not None:
                story = s['story'].replace('|', '\\|')
                lines.append(f"| {start} | {name} | {story} | {'yes' if s['exact_copy'] else 'no'} | "
                             f"{s['3gram_in_training']:.2f} |")
    table = '\n'.join(lines)
    print('\n' + table)
    (out_dir / 'comparison.md').write_text(table + '\n', encoding='utf-8')
    plot_comparison(results, order,
                    [('train_perplexity', 'Training perplexity (log scale)', True),
                     ('val_perplexity', 'Validation perplexity (log scale)', True)],
                    'Joke language model: perplexity by model',
                    out_dir / 'comparison_learning_curves.png')


# (result key, panel title, which direction is better) for the summary figure.
SUMMARY_PANELS = {
    'classification': ('Sentiment classification', CLASSIFICATION_RUNS, [
        ('accuracy', 'Test accuracy', 'higher'),
        ('f1', 'Test F1', 'higher'),
        ('test_loss', 'Test loss', 'lower'),
        ('train_time', 'Training time (s)', 'lower'),
    ]),
    'generation': ('Joke generation', GENERATION_RUNS, [
        ('best_val_perplexity', 'Best validation perplexity', 'lower'),
        ('final_val_perplexity', 'Final validation perplexity', 'lower'),
        ('final_train_perplexity', 'Final training perplexity', None),
        ('mean_3gram_in_training', 'Story 3-grams copied from training', None),
    ]),
}


def plot_summary(path=None):

    path = path or RESULT_DIR / 'architecture_summary.png'
    tasks = []
    for task, (title, runs, panels) in SUMMARY_PANELS.items():
        saved = RESULT_DIR / task / 'comparison.json'
        if saved.exists():
            results = json.loads(saved.read_text())
            order = list(runs)
            names = [n for n in order if n in results] + [n for n in results if n not in order]
            tasks.append((title, runs, panels, results, order, names))
    if not tasks:
        print('No comparison results yet, so no summary figure was drawn')
        return
    columns = max(len(panels) for _, _, panels, _, _, _ in tasks)
    fig, grid = plt.subplots(len(tasks), columns, figsize=(4.4 * columns, 3.6 * len(tasks)),
                             facecolor=SURFACE, squeeze=False)
    for row, (title, runs, panels, results, order, names) in zip(grid, tasks):
        positions = range(len(names), 0, -1)  # First run at the top.
        for column, ax in enumerate(row):
            ax.set_facecolor(SURFACE)
            if column >= len(panels):
                ax.set_visible(False)
                continue
            key, label, better = panels[column]
            values = [results[name][key] for name in names]
            best = None if better is None else (max(values) if better == 'higher' else min(values))
            span = (max(values) - min(values)) or 1.0
            ax.set_xlim(min(values) - 0.12 * span, max(values) + 0.42 * span)
            ax.set_ylim(0.4, len(names) + 0.6)
            for name, value, y in zip(names, values, positions):
                slot = order.index(name) if name in order else len(order)
                ax.plot([ax.get_xlim()[0], value], [y, y], color=GRID, linewidth=1.5, zorder=1)
                ax.plot(value, y, 'o', markersize=9, color=SERIES_COLORS[slot % len(SERIES_COLORS)],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=2)
                text = f'{value:.0f}' if abs(value) >= 100 else f'{value:.3f}' if abs(value) < 1 else f'{value:.2f}'
                ax.annotate(text, (value, y), xytext=(8, 0), textcoords='offset points', va='center',
                            color=INK, fontsize=9, fontweight='bold' if value == best else 'normal')
            ax.set_title(label, color=INK, fontsize=10, loc='left')
            ax.set_yticks(list(positions))
            if column == 0:
                described = [name if describe(runs.get(name, {})) in ('', f'cell={name}')
                             else name + '\n' + describe(runs[name]) for name in names]
                ax.set_yticklabels(described, color=INK, fontsize=9)
            else:
                ax.set_yticklabels([])
            ax.tick_params(axis='x', colors=INK_SECONDARY, labelsize=8)
            ax.tick_params(axis='y', length=0)
            ax.grid(True, axis='x', color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ['top', 'right', 'left']:
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(GRID)
        row[0].annotate(title, (0, 1), xytext=(0, 26), xycoords='axes fraction',
                        textcoords='offset points', color=INK, fontsize=12, fontweight='bold')
    fig.suptitle('Testing results', color=INK, fontsize=14, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=3.0)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f'Summary figure written to {path}')


def plot_training_convergence(path=None):

    path = path or RESULT_DIR / 'training_convergence.png'
    tasks = []
    for task, (title, runs, _) in SUMMARY_PANELS.items():
        saved = RESULT_DIR / task / 'comparison.json'
        if saved.exists():
            tasks.append((title, list(runs), json.loads(saved.read_text())))
    if not tasks:
        print('No comparison results yet, so no training convergence figure was drawn')
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(7 * len(tasks), 4.8), facecolor=SURFACE,
                             squeeze=False)
    for ax, (title, order, results) in zip(axes[0], tasks):
        ax.set_facecolor(SURFACE)
        # Runs that train identically share one line and one legend entry.
        labels = {}
        for name in [n for n in order if n in results] + [n for n in results if n not in order]:
            twin = next((n for n in labels if results[n]['history'] == results[name]['history']), None)
            if twin is None:
                labels[name] = name
            else:
                labels[twin] += ' / ' + name
        for name, label in labels.items():
            slot = order.index(name) if name in order else len(order)
            history = results[name]['history']
            ax.plot(history['epoch'], history['train_loss'], linewidth=2, marker='o', markersize=3,
                    color=SERIES_COLORS[slot % len(SERIES_COLORS)], label=label)
        ax.set_title(title, color=INK, fontsize=12, fontweight='bold', loc='left')
        ax.set_xlabel('Epoch', color=INK_SECONDARY)
        ax.set_ylabel('Training loss (cross-entropy)', color=INK_SECONDARY)
        ax.tick_params(colors=INK_SECONDARY)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
        for side in ['left', 'bottom']:
            ax.spines[side].set_color(GRID)
        ax.legend(frameon=False, labelcolor=INK, loc='upper right')
    fig.suptitle('Training convergence', color=INK, fontsize=14, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)
    print(f'Training convergence figure written to {path}')


def main():
    parser = argparse.ArgumentParser(description='Stage 4.5 RNN / LSTM / GRU model comparison')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--tasks', nargs='+', default=['classification', 'generation'],
                        choices=['classification', 'generation'])
    parser.add_argument('--classification-runs', nargs='+', default=list(CLASSIFICATION_RUNS),
                        choices=list(CLASSIFICATION_RUNS))
    parser.add_argument('--generation-runs', nargs='+', default=list(GENERATION_RUNS),
                        choices=list(GENERATION_RUNS))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--max-epoch', type=int, default=None,
                        help='Override the epoch limit of every run (e.g. 1 for a quick check)')
    parser.add_argument('--summary-only', action='store_true',
                        help='Train nothing; redraw the two cross-task figures from saved results')
    args = parser.parse_args()

    if not args.summary_only:
        if 'generation' in args.tasks:
            compare_generation(args)
        if 'classification' in args.tasks:
            compare_classification(args)
    plot_summary()
    plot_training_convergence()
    print(f'\nResults written to {RESULT_DIR}')


if __name__ == '__main__':
    main()
