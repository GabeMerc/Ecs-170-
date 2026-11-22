"""Stage 4.3/4.4 evaluation: classification metrics, learning curves, and
comparison of generated stories against the training jokes."""
from difflib import SequenceMatcher

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from local_code.base_class.evaluate import evaluate


class Evaluate_Text(evaluate):
    data = None

    def evaluate(self):
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def classification_suite(self):
        true_y, pred_y = self.data['true_y'], self.data['pred_y']
        scores = {'accuracy': accuracy_score(true_y, pred_y)}
        for average in ['binary', 'macro', 'weighted']:
            suffix = '' if average == 'binary' else '_' + average
            scores['precision' + suffix] = precision_score(true_y, pred_y, average=average, zero_division=0)
            scores['recall' + suffix] = recall_score(true_y, pred_y, average=average, zero_division=0)
            scores['f1' + suffix] = f1_score(true_y, pred_y, average=average, zero_division=0)
        scores['confusion_matrix'] = confusion_matrix(true_y, pred_y, labels=[0, 1]).tolist()
        return scores


def plot_learning_curves(history, path, title, test_accuracy=None, metric='accuracy'):
    """Train vs validation loss and metric per epoch. metric may be accuracy or perplexity."""
    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = history['epoch']
    ax_loss.plot(epochs, history['train_loss'], marker='o', markersize=3, label='Train')
    if history.get('val_loss'):
        ax_loss.plot(epochs, history['val_loss'], marker='o', markersize=3, label='Validation')
    ax_loss.set_title(title + ': loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Cross-Entropy Loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_metric.plot(epochs, history['train_' + metric], marker='o', markersize=3, label='Train')
    if history.get('val_' + metric):
        ax_metric.plot(epochs, history['val_' + metric], marker='o', markersize=3, label='Validation')
    if test_accuracy is not None:
        ax_metric.axhline(test_accuracy, linestyle='--', color='gray',
                          label=f'Test accuracy ({test_accuracy:.4f})')
    ax_metric.set_title(title + ': ' + metric)
    ax_metric.set_xlabel('Epoch')
    ax_metric.set_ylabel(metric.capitalize())
    if metric == 'perplexity':
        ax_metric.set_yscale('log')
    ax_metric.legend()
    ax_metric.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _longest_copied_span(tokens, corpus_ngrams, max_n):
    # Longest run of the story that appears verbatim somewhere in the training jokes.
    best = 0
    for n in range(1, max_n + 1):
        if any(gram in corpus_ngrams[n] for gram in _ngrams(tokens, n)):
            best = n
        else:
            break
    return best


class TrainingCorpusIndex:
    """N-gram index over the training jokes for checking generated stories."""
    def __init__(self, token_lists, max_n=64):
        self.token_lists = token_lists
        self.max_n = max_n
        self.ngrams = {n: set() for n in range(1, max_n + 1)}
        for tokens in token_lists:
            for n in range(1, max_n + 1):
                self.ngrams[n].update(_ngrams(tokens, n))
        self.exact = {tuple(tokens) for tokens in token_lists}

    def compare(self, prefix, story):
        """prefix: the three starting tokens. story: prefix + generated tokens."""
        prefix_t = tuple(prefix)
        starts = [t for t in self.token_lists if tuple(t[:len(prefix)]) == prefix_t]
        contains = sum(1 for t in self.token_lists
                       if prefix_t in set(_ngrams(t, len(prefix))))
        match_scores = [SequenceMatcher(None, story, t, autojunk=False).ratio()
                        for t in self.token_lists]
        best = max(range(len(match_scores)), key=match_scores.__getitem__)
        result = {
            'prefix_starts_jokes': len(starts),
            'prefix_appears_in_jokes': contains,
            'exact_copy': tuple(story) in self.exact,
            'nearest_joke': self.token_lists[best],
            'nearest_similarity': match_scores[best],
            'longest_copied_span': _longest_copied_span(story, self.ngrams, self.max_n),
            'length': len(story),
        }
        # Share of the story's n-grams seen in training: high = locally fluent/copied.
        for n in [2, 3, 4]:
            grams = _ngrams(story, n)
            result[f'{n}gram_in_training'] = (sum(g in self.ngrams[n] for g in grams) / len(grams)
                                              if grams else float('nan'))
        return result
