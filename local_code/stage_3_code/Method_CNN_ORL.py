'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Method_CNN_ORL(method, nn.Module):
    data = None
    num_classes = 40
    input_shape = (1, 112, 92)

    max_epoch = 100
    learning_rate = 1e-3
    batch_size = 32
    patience = 8
    min_delta = 1e-4

    def __init__(self, mName, mDescription,
                 conv_channels=(32, 64, 128),
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 pooling='max',
                 hidden_sizes=(256,),
                 batch_norm=True,
                 dropout=0.5,
                 optimizer_name='adam',
                 loss_name='cross_entropy',
                 weight_decay=0.0,
                 augment=False,
                 seed=21,
                 max_epoch=None,
                 patience=None,
                 learning_rate=None,
                 device=None,
                 verbose=True):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.weight_decay = weight_decay
        self.augment = augment
        self.verbose = verbose
        if max_epoch is not None:
            self.max_epoch = max_epoch
        if patience is not None:
            self.patience = patience
        self.custom_lr = learning_rate

        conv_layers = []
        in_ch = self.input_shape[0]
        for out_ch in conv_channels:
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding))
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_ch))
            conv_layers.append(nn.ReLU())
            if pooling == 'max':
                conv_layers.append(nn.MaxPool2d(2))
            elif pooling == 'avg':
                conv_layers.append(nn.AvgPool2d(2))
            elif pooling is not None:
                raise ValueError('unknown pooling: ' + str(pooling))
            in_ch = out_ch
        self.features = nn.Sequential(*conv_layers)

        with torch.no_grad():
            self.flat_dim = self.features(torch.zeros(1, *self.input_shape)).numel()

        fc_layers = [nn.Flatten()]
        in_size = self.flat_dim
        for size in hidden_sizes:
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            fc_layers.append(nn.Linear(in_size, size))
            fc_layers.append(nn.ReLU())
            in_size = size
        if dropout > 0:
            fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(in_size, self.num_classes))
        if loss_name == 'nll':
            fc_layers.append(nn.LogSoftmax(dim=1))
        self.classifier = nn.Sequential(*fc_layers)

        self.num_params = sum(p.numel() for p in self.parameters())

        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = torch.device(device)
        self.to(self.device)

        if self.verbose:
            print(f'{mName}: flatten dim {self.flat_dim}, {self.num_params:,} parameters, device {self.device}')

    def forward(self, x):
        return self.classifier(self.features(x))

    # train(X, y) overrides nn.Module.train(mode), so self.eval() can't be used.
    # Set the training flag on each module directly instead.
    def _set_training_mode(self, mode):
        for m in self.modules():
            m.training = mode

    @staticmethod
    def _prepare(X):
        return torch.FloatTensor(np.asarray(X)) / 255.0

    def _augment_batch(self, X_batch):
        flip = torch.rand(X_batch.size(0), device=X_batch.device) < 0.5
        X_batch = X_batch.clone()
        X_batch[flip] = X_batch[flip].flip(-1)
        pad = 4
        padded = F.pad(X_batch, (pad, pad, pad, pad), mode='replicate')
        dx, dy = np.random.randint(0, 2 * pad + 1, size=2)
        return padded[:, :, dy:dy + X_batch.size(2), dx:dx + X_batch.size(3)]

    def _make_loss(self):
        if self.loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        if self.loss_name == 'nll':
            return nn.NLLLoss()
        if self.loss_name == 'mse':
            mse = nn.MSELoss()
            def loss_function(y_pred, y_true):
                return mse(torch.softmax(y_pred, dim=1),
                           F.one_hot(y_true, num_classes=self.num_classes).float())
            return loss_function
        raise ValueError('unknown loss_name: ' + str(self.loss_name))

    def _make_optimizer(self):
        lr = self.custom_lr
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=lr or self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_name == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=lr or self.learning_rate,
                                     weight_decay=self.weight_decay or 1e-2)
        if self.optimizer_name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=lr or 1e-2, momentum=0.9, weight_decay=self.weight_decay)
        raise ValueError('unknown optimizer_name: ' + str(self.optimizer_name))

    def train(self, X, y):
        optimizer = self._make_optimizer()
        loss_function = self._make_loss()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = self._prepare(X).to(self.device)
        y_tensor = torch.LongTensor(np.asarray(y)).to(self.device)
        y_cpu = y_tensor.cpu()
        n = X_tensor.size(0)

        best_loss = float('inf')
        epochs_without_improvement = 0
        self.history = {'epoch': [], 'loss': [], 'accuracy': []}

        for epoch in range(self.max_epoch):
            self._set_training_mode(True)
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_batch, y_batch = X_tensor[idx], y_tensor[idx]
                if self.augment:
                    X_batch = self._augment_batch(X_batch)

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

            self._set_training_mode(False)
            with torch.no_grad():
                full_pred = self.forward(X_tensor)
                full_loss = loss_function(full_pred, y_tensor).item()
            pred_cpu = full_pred.argmax(1).cpu()
            accuracy_evaluator.data = {'true_y': y_cpu, 'pred_y': pred_cpu}
            train_accuracy = accuracy_evaluator.evaluate() if self.verbose else \
                (pred_cpu == y_cpu).float().mean().item()
            if self.verbose:
                print('Epoch:', epoch, 'Accuracy:', train_accuracy, 'Loss:', full_loss)

            self.history['epoch'].append(epoch)
            self.history['loss'].append(full_loss)
            self.history['accuracy'].append(train_accuracy)

            if full_loss < best_loss - self.min_delta:
                best_loss = full_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print('Early stopping at epoch', epoch, '- no improvement in', self.patience, 'epochs')
                    break

    def test(self, X):
        self._set_training_mode(False)
        with torch.no_grad():
            y_pred = self.forward(self._prepare(X).to(self.device))
        return y_pred.argmax(1).cpu()

    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}