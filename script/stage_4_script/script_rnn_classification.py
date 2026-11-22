
import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader import Dataset_Loader, collate_reviews
from local_code.stage_4_code.Evaluate_Text import Evaluate_Text, plot_learning_curves
from local_code.stage_4_code.Method_RNN import Method_RNN

RESULT_DIR = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result' / 'classification'

trial_runs = {
    'rnn': dict(cell='rnn'),
    'lstm': dict(cell='lstm'),
    'gru': dict(cell='gru'),
    'bilstm': dict(cell='lstm', bidirectional=True),
}


class BucketBatchSampler(torch.utils.data.Sampler):
    """Shuffled batches of similar-length reviews, so padding (and CPU time) stays small."""
    def __init__(self, lengths, batch_size, seed):
        self.lengths, self.batch_size = lengths, batch_size
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        self.rng.shuffle(idx)
        pool = self.batch_size * 50
        batches = []
        for i in range(0, len(idx), pool):
            chunk = sorted(idx[i:i + pool], key=self.lengths.__getitem__)
            batches += [chunk[j:j + self.batch_size] for j in range(0, len(chunk), self.batch_size)]
        self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def run_epoch(model, loader, device, loss_function, optimizer=None, clip=5.0):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    preds, trues = [], []
    with torch.set_grad_enabled(training):
        for tokens, lengths, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens, lengths)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            pred = logits.argmax(1)
            total_loss += loss.item() * len(labels)
            correct += (pred == labels).sum().item()
            count += len(labels)
            preds.append(pred.cpu())
            trues.append(labels.cpu())
    return total_loss / count, correct / count, torch.cat(trues).numpy(), torch.cat(preds).numpy()


def train_and_evaluate(name, configuration, data, args, result_dir=RESULT_DIR, return_history=False):
    result_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    model = Method_RNN(name, '', vocab_size=len(data['vocabulary']),
                       embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim,
                       dropout=args.dropout, **configuration).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'\n===== {name}: {configuration}, {num_params:,} parameters, device {device} =====')

    lengths = [len(seq) for seq in data['train'].sequences]
    train_loader = DataLoader(data['train'], collate_fn=collate_reviews,
                              batch_sampler=BucketBatchSampler(lengths, args.batch_size, args.seed))
    eval_loader = lambda split: DataLoader(data[split], batch_size=256, collate_fn=collate_reviews)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {k: [] for k in ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']}
    best_loss, best_state, best_epoch, stale = float('inf'), None, -1, 0
    start = time.time()
    for epoch in range(args.max_epoch):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, device, loss_function, optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, eval_loader('val'), device, loss_function)
        for key, value in zip(history, [epoch + 1, train_loss, train_acc, val_loss, val_acc]):
            history[key].append(value)
        print(f'Epoch {epoch + 1:<3} train loss {train_loss:.4f} acc {train_acc:.4f} | '
              f'val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - start:.0f}s')
        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = val_loss, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print(f'Early stopping: validation loss has not improved for {args.patience} epochs')
                break
    train_time = time.time() - start

    # Test once, with the checkpoint chosen on validation loss.
    model.load_state_dict(best_state)
    test_loss, _, true_y, pred_y = run_epoch(model, eval_loader('test'), device, loss_function)
    evaluator = Evaluate_Text(name + ' evaluator', '')
    evaluator.data = {'true_y': true_y, 'pred_y': pred_y}
    scores = evaluator.classification_suite()
    scores.update(test_loss=test_loss, best_epoch=best_epoch, epochs_run=len(history['epoch']),
                  train_time=train_time, num_params=num_params)
    print(f"{name}: test accuracy {scores['accuracy']:.4f}, F1 {scores['f1']:.4f} (best epoch {best_epoch})")

    plot_learning_curves(history, result_dir / f'{name}_learning_curves.png',
                         f'{name.upper()} sentiment', test_accuracy=scores['accuracy'])
    save_run_log(result_dir / f'{name}_results.txt', name, configuration, args, history, scores)
    if args.save_model:
        torch.save({'state_dict': best_state, 'vocabulary': data['vocabulary'].tokens,
                    'configuration': configuration}, result_dir / f'{name}_model.pt')
    return (scores, history) if return_history else scores


def save_run_log(path, name, configuration, args, history, scores):
    with open(path, 'w') as f:
        f.write(f'Run: {name}\nConfiguration: {configuration}\n')
        f.write(f'Hyperparameters: embedding_dim={args.embedding_dim}, hidden_dim={args.hidden_dim}, '
                f'dropout={args.dropout}, lr={args.lr}, batch_size={args.batch_size}, '
                f'max_epoch={args.max_epoch}, patience={args.patience}, grad_clip=5.0\n')
        f.write(f"Seed: {args.seed}\nParameters: {scores['num_params']:,}\nDevice: {args.device}\n"
                f"Training time: {scores['train_time']:.1f}s\n\nPer epoch\n")
        for row in zip(*history.values()):
            f.write('Epoch {:<3} Train loss {:.4f}  Train acc {:.4f}  Val loss {:.4f}  Val acc {:.4f}\n'.format(*row))
        f.write(f"\nBest epoch (lowest validation loss): {scores['best_epoch']}\n\nTest results (25,000 reviews)\n")
        for key in ['test_loss', 'accuracy', 'precision', 'recall', 'f1', 'precision_macro',
                    'recall_macro', 'f1_macro', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            f.write(f'{key + ":":<20}{scores[key]:.4f}\n')
        (tn, fp), (fn, tp) = scores['confusion_matrix']
        f.write(f'\nConfusion matrix (rows true, cols predicted; 0=neg, 1=pos)\n'
                f'          pred neg  pred pos\ntrue neg  {tn:>8}  {fp:>8}\ntrue pos  {fn:>8}  {tp:>8}\n')


def build_parser():
    parser = argparse.ArgumentParser(description='Stage 4.3 RNN sentiment classification')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--runs', nargs='+', default=['rnn', 'lstm'], choices=list(trial_runs))
    parser.add_argument('--max-epoch', type=int, default=15)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--embedding-dim', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--save-model', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = Dataset_Loader(data_root=args.data_root).load(val_fraction=args.val_fraction, seed=args.seed)
    print(f"train {len(data['train'])}, val {len(data['val'])}, test {len(data['test'])}, "
          f"vocabulary {len(data['vocabulary'])}")

    summary = {name: train_and_evaluate(name, trial_runs[name], data, args) for name in args.runs}

    lines = ['# Stage 4.3 sentiment classification (test set, 25,000 reviews)', '',
             f'Train {len(data["train"])} / validation {len(data["val"])} split of the training set, '
             f'vocabulary {len(data["vocabulary"])}, seed {args.seed}.', '',
             '| Run | Params | Best epoch | Accuracy | Precision | Recall | F1 | Train time |',
             '|---|---|---|---|---|---|---|---|']
    for name, s in summary.items():
        lines.append(f"| {name} | {s['num_params']:,} | {s['best_epoch']} / {s['epochs_run']} | "
                     f"{s['accuracy']:.4f} | {s['precision']:.4f} | {s['recall']:.4f} | {s['f1']:.4f} | "
                     f"{s['train_time']:.0f}s |")
    table = '\n'.join(lines)
    print('\n' + table)
    out = 'summary.md' if args.runs == ['rnn', 'lstm'] else 'summary_' + '_'.join(args.runs) + '.md'
    (RESULT_DIR / out).write_text(table + '\n')
    (RESULT_DIR / out.replace('.md', '.json')).write_text(json.dumps(summary, indent=2, default=float))


if __name__ == '__main__':
    main()
