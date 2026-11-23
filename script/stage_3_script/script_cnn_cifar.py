"""Train on the course CIFAR pickle. Select settings on validation before --test."""
import argparse
import csv
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from local_code.stage_3_code.Method_CNN_CIFAR import CIFARCNN as CNN

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    result = {}
    for split in ('train', 'test'):
        x = np.stack([row['image'] for row in data[split]])
        y = np.asarray([row['label'] for row in data[split]], dtype=np.int64)
        assert x.shape[1:] == (32, 32, 3) and x.dtype == np.uint8
        assert y.min() == 0 and y.max() == 9
        result[split] = (x, y)
    return result


class Images(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.transform(self.x[i]), int(self.y[i])




def evaluate(model, loader, device):
    model.eval()
    loss_sum, targets, predictions = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += nn.functional.cross_entropy(logits, y, reduction='sum').item()
            targets.extend(y.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    p, r, f, _ = precision_recall_fscore_support(targets, predictions, average='macro', zero_division=0)
    return {'loss': loss_sum / len(targets), 'accuracy': accuracy_score(targets, predictions),
            'macro_precision': float(p), 'macro_recall': float(r), 'macro_f1': float(f)}, targets, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=str(REPO_ROOT / 'data/stage_3_data/CIFAR'))
    parser.add_argument('--out', default=str(REPO_ROOT / 'result/stage_3_result/CIFAR/proposed'))
    parser.add_argument('--variant', choices=['proposed', 'no_augmentation', 'no_batchnorm', 'no_dropout', 'shallower'], default='proposed')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--test', action='store_true', help='Evaluate existing best.pt only; do this after model selection.')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    out = Path(args.out)
    if not args.test and (out / 'best.pt').exists():
        parser.error('Output already contains a checkpoint; choose a new --out folder.')
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(args.data)
    x, y = data['train']
    rng = np.random.default_rng(args.seed)
    train_idx, val_idx = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(y == label))
        val_idx.extend(indices[:500]); train_idx.extend(indices[500:])
    train_idx, val_idx = np.array(train_idx), np.array(val_idx)
    # Statistics come only from the 45,000 training images.
    mean = x[train_idx].mean(axis=(0, 1, 2)) / 255
    std = x[train_idx].std(axis=(0, 1, 2)) / 255
    config = dict(vars(args), device_used=device, mean=mean.tolist(), std=std.tolist())
    if args.test:
        config = json.loads((out / 'config.json').read_text())
        mean, std = config['mean'], config['std']
    depth = 2 if config['variant'] == 'shallower' else 3
    model = CNN(depth, config['variant'] != 'no_batchnorm', 0 if config['variant'] == 'no_dropout' else 0.3).to(device)
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    plain = transforms.Compose(normalize)
    def loader(images, labels, transform, shuffle=False):
        return DataLoader(Images(images, labels, transform), batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    if args.test:
        model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=True))
        metrics, truth, predicted = evaluate(model, loader(*data['test'], plain), device)
        (out / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
        np.savetxt(out / 'confusion_matrix.csv', confusion_matrix(truth, predicted, labels=list(range(10))), fmt='%d', delimiter=',')
        print('FINAL TEST:', metrics)
        return
    config['parameters'] = sum(p.numel() for p in model.parameters())
    config['torch_version'] = torch.__version__
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    np.savez(out / 'split_indices.npz', train=train_idx, validation=val_idx)
    augmentation = [] if args.variant == 'no_augmentation' else [transforms.ToPILImage(), transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    training = loader(x[train_idx], y[train_idx], transforms.Compose(augmentation + normalize), True)
    validation = loader(x[val_idx], y[val_idx], plain)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history, best, start = [], -1, time.time()
    print('Device:', device, 'Parameters:', config['parameters'], flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, count = 0., 0, 0
        for images, labels in training:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            loss.backward(); optimizer.step()
            count += len(labels)
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
        metrics, _, _ = evaluate(model, validation, device)
        row = dict(epoch=epoch, train_loss=total_loss/count, train_accuracy=correct/count,
                   **{'val_' + k: v for k, v in metrics.items()})
        history.append(row)
        if metrics['accuracy'] > best:
            best = metrics['accuracy']
            torch.save(model.state_dict(), out / 'best.pt')
            (out / 'best_validation.json').write_text(json.dumps(dict(epoch=epoch, **metrics), indent=2))
        scheduler.step()
        with open(out / 'history.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerows(history)
        print(f"Epoch {epoch}/{args.epochs}: loss={row['train_loss']:.4f}, val_accuracy={metrics['accuracy']:.2%}", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Training')
    ax.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='Validation')
    ax.set(xlabel='Epoch', ylabel='Cross-entropy loss'); ax.legend(); fig.tight_layout()
    fig.savefig(out / 'loss_curve.png', dpi=180); plt.close(fig)
    (out / 'runtime.json').write_text(json.dumps({'seconds': time.time()-start}, indent=2))
    print('Best validation accuracy:', best, 'Test set not evaluated.')


if __name__ == '__main__':
    main()
