'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'], self.data['pred_y'])

    def evaluate_weighted_suite(self, plot_save_path=None, loss_name='cross_entropy'):
        #Print accuracy + weighted precision/recall/F1
        #loss_name: same names as the methods' loss_name option, used for the loss plot's y-axis label
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        scores = {
            'accuracy': accuracy_score(true_y, pred_y),
            'precision_weighted': precision_score(true_y, pred_y, average='weighted', zero_division=0),
            'recall_weighted': recall_score(true_y, pred_y, average='weighted', zero_division=0),
            'f1_weighted': f1_score(true_y, pred_y, average='weighted', zero_division=0),
        }

        print('Evaluation Suite: ')
        for name, value in scores.items():
            print(f'{name:>20}: {value:.4f}')

        # convergence plots, training loss and accuracy per epoch
        history = self.data.get('history')
        if history is not None:
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

            ax_loss.plot(history['epoch'], history['loss'], marker='o', markersize=3)
            ax_loss.set_title('Training Loss Convergence')
            ax_loss.set_xlabel('Epoch')
            loss_labels = {'cross_entropy': 'Cross-Entropy Loss', 'nll': 'NLL Loss', 'mse': 'MSE Loss'}
            ax_loss.set_ylabel(loss_labels.get(loss_name, str(loss_name)))
            ax_loss.grid(True, alpha=0.3)

            ax_acc.plot(history['epoch'], history['accuracy'], marker='o', markersize=3)
            ax_acc.axhline(scores['accuracy'], linestyle='--', color='gray', label=f"Test accuracy ({scores['accuracy']:.4f})")
            ax_acc.set_title('Training Accuracy Convergence')
            ax_acc.set_xlabel('Epoch')
            ax_acc.set_ylabel('Accuracy')
            ax_acc.legend()
            ax_acc.grid(True, alpha=0.3)

            fig.tight_layout()
            if plot_save_path is not None:
                fig.savefig(plot_save_path, dpi=150)
                print('convergence plot saved to', plot_save_path)
            plt.show()

        return scores
