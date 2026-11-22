"""Stage 4.4: train a word-level RNN language model on the jokes, generate stories from
three starting words, and compare them against the training data.

Run from repository root:
    python -m script.stage_4_script.script_rnn_generation --data-root PATH [--start "what did the"]
"""
import argparse
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from local_code.stage_4_code.Dataset_Loader import (Dataset_Loader, collate_jokes, detokenize,
                                                    tokenize_joke)
from local_code.stage_4_code.Evaluate_Text import TrainingCorpusIndex, plot_learning_curves
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator

RESULT_DIR = Path(__file__).resolve().parents[2] / 'result' / 'stage_4_result' / 'generation'

DEFAULT_STARTS = ['what did the', 'why did the', 'did you hear', 'i went to', 'my wife asked']


def run_epoch(model, loader, device, loss_function, optimizer=None, clip=5.0):
    training = optimizer is not None
    model.train(training)
    total_loss, total_tokens = 0.0, 0
    with torch.set_grad_enabled(training):
        for inputs, lengths, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits, _ = model(inputs, lengths)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()
            tokens = (targets != 0).sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description='Stage 4.4 RNN text generation')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--start', nargs='+', default=DEFAULT_STARTS,
                        help='Starting phrases of exactly three words each')
    parser.add_argument('--cell', default='lstm', choices=['rnn', 'lstm', 'gru'])
    parser.add_argument('--max-epoch', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--embedding-dim', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--samples', type=int, default=3, help='Sampled stories per start, besides greedy')
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    starts = [tokenize_joke(s) for s in args.start]
    for phrase, tokens in zip(args.start, starts):
        if len(tokens) != 3:
            parser.error(f'start phrase must be three words: {phrase!r} -> {tokens}')

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    corpus = Dataset_Loader(data_root=args.data_root).load_generation_corpus(
        val_fraction=args.val_fraction, seed=args.seed)
    vocab = corpus['vocabulary']
    eos = vocab.word_to_id['<eos>']
    print(f"train jokes {len(corpus['train'])}, val jokes {len(corpus['val_jokes'])}, vocabulary {len(vocab)}")

    name = args.cell + '_generator'
    model = Method_RNN_Generator(name, '', vocab_size=len(vocab), embedding_dim=args.embedding_dim,
                                 hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                                 dropout=args.dropout, cell=args.cell).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'{name}: {num_params:,} parameters, device {device}')

    train_loader = DataLoader(corpus['train'], batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_jokes)
    val_loader = (DataLoader(corpus['val'], batch_size=256, collate_fn=collate_jokes)
                  if corpus['val'] is not None else None)
    loss_function = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {k: [] for k in ['epoch', 'train_loss', 'train_perplexity', 'val_loss', 'val_perplexity']}
    start_time = time.time()
    for epoch in range(args.max_epoch):
        run_epoch(model, train_loader, device, loss_function, optimizer)
        # Re-measure without dropout so train and validation numbers are comparable.
        train_loss = run_epoch(model, train_loader, device, loss_function)
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['train_perplexity'].append(math.exp(train_loss))
        line = f'Epoch {epoch + 1:<3} train loss {train_loss:.4f} ppl {math.exp(train_loss):8.2f}'
        if val_loader is not None:
            val_loss = run_epoch(model, val_loader, device, loss_function)
            history['val_loss'].append(val_loss)
            history['val_perplexity'].append(math.exp(val_loss))
            line += f' | val loss {val_loss:.4f} ppl {math.exp(val_loss):8.2f}'
        print(line + f' | {time.time() - start_time:.0f}s')
    train_time = time.time() - start_time
    plot_learning_curves(history, RESULT_DIR / f'{name}_learning_curves.png',
                         f'{args.cell.upper()} joke language model', metric='perplexity')

    index = TrainingCorpusIndex([tokenize_joke(j) for j in corpus['train_jokes']])
    rng = torch.Generator().manual_seed(args.seed)
    stories = []
    for prefix in starts:
        prefix_ids = [vocab.word_to_id.get(w, vocab.UNK) for w in prefix]
        unknown = [w for w in prefix if w not in vocab.word_to_id]
        decodes = [('greedy', 0.0)] + [(f'sample {i + 1}', args.temperature) for i in range(args.samples)]
        for label, temperature in decodes:
            ids = model.generate(prefix_ids, eos, temperature=temperature, top_k=args.top_k, generator=rng)
            story = prefix + vocab.decode(ids)
            comparison = index.compare(prefix, story)
            stories.append({'start': ' '.join(prefix), 'decoding': label,
                            'unknown_start_words': unknown, 'story': detokenize(story),
                            **comparison, 'nearest_joke': detokenize(comparison['nearest_joke'])})
            print(f"\n[{' '.join(prefix)} | {label}] {detokenize(story)}\n"
                  f"   nearest training joke ({comparison['nearest_similarity']:.2f}): "
                  f"{detokenize(comparison['nearest_joke'])}")

    save_report(name, args, num_params, train_time, history, stories, len(vocab), len(corpus['train']))


def save_report(name, args, num_params, train_time, history, stories, vocab_size, train_count):
    with open(RESULT_DIR / f'{name}_results.txt', 'w') as f:
        f.write(f'Run: {name}\nHyperparameters: cell={args.cell}, embedding_dim={args.embedding_dim}, '
                f'hidden_dim={args.hidden_dim}, num_layers={args.num_layers}, dropout={args.dropout}, '
                f'lr={args.lr}, batch_size={args.batch_size}, epochs={args.max_epoch}, grad_clip=5.0\n')
        f.write(f'Decoding: greedy, plus {args.samples} samples at temperature {args.temperature}, '
                f'top-k {args.top_k}\nSeed: {args.seed}\nParameters: {num_params:,}\n'
                f'Vocabulary: {vocab_size}\nTraining jokes: {train_count}\nDevice: {args.device}\n'
                f'Training time: {train_time:.1f}s\n\nPer epoch\n')
        for i, epoch in enumerate(history['epoch']):
            row = f"Epoch {epoch:<3} Train loss {history['train_loss'][i]:.4f}  Train ppl {history['train_perplexity'][i]:.2f}"
            if history['val_loss']:
                row += f"  Val loss {history['val_loss'][i]:.4f}  Val ppl {history['val_perplexity'][i]:.2f}"
            f.write(row + '\n')
        f.write('\nGenerated stories vs training data\n')
        for s in stories:
            f.write(f"\n[{s['start']} | {s['decoding']}]\n  Story:   {s['story']}\n"
                    f"  Nearest: {s['nearest_joke']}  (similarity {s['nearest_similarity']:.2f})\n"
                    f"  Start begins {s['prefix_starts_jokes']} training jokes, appears in "
                    f"{s['prefix_appears_in_jokes']}; exact copy: {s['exact_copy']}; "
                    f"longest verbatim span {s['longest_copied_span']}/{s['length']} tokens; "
                    f"n-grams seen in training: 2={s['2gram_in_training']:.2f} "
                    f"3={s['3gram_in_training']:.2f} 4={s['4gram_in_training']:.2f}\n")

    lines = ['# Stage 4.4 joke generation vs training data', '',
             f'{args.cell.upper()} language model, {num_params:,} parameters, {args.max_epoch} epochs. '
             f"Final train perplexity {history['train_perplexity'][-1]:.2f}"
             + (f", validation perplexity {history['val_perplexity'][-1]:.2f}" if history['val_loss'] else '')
             + '.', '',
             '| Start | Decoding | Story | Exact copy | Nearest sim. | Longest copied span | 3-grams in train |',
             '|---|---|---|---|---|---|---|']
    for s in stories:
        story = s['story'].replace('|', '\\|')
        lines.append(f"| {s['start']} | {s['decoding']} | {story} | {'yes' if s['exact_copy'] else 'no'} | "
                     f"{s['nearest_similarity']:.2f} | {s['longest_copied_span']}/{s['length']} | "
                     f"{s['3gram_in_training']:.2f} |")
    (RESULT_DIR / f'{name}_summary.md').write_text('\n'.join(lines) + '\n')
    (RESULT_DIR / f'{name}_stories.json').write_text(json.dumps(stories, indent=2))


if __name__ == '__main__':
    main()
