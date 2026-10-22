'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import torch
from torch import nn
import numpy as np


class Method_MLP(method, nn.Module):
    data = None
    # max training rounds
    max_epoch = 500

    learning_rate = 1e-3
    # num in each mini batch
    batch_size = 128
    # how many epochs without improvement are allowed before stopping early
    patience = 5
    # minimum decrease in loss that counts as an improvement, interacts with patience.
    min_delta = 1e-4

    seed = 21

    # architecture options are passed here
    #   hidden_sizes: one hidden layer per entry, (512, 256, 128) = 784 -> 512 -> 256 -> 128 -> 10, 784 is required with our dataset.
    #   activation: hidden-layer activation ('relu', 'leaky_relu', 'tanh')
    #   output_activation: applied after the last layer ('leaky_relu', 'relu', 'log_softmax', or None)
    #   optimizer_name: 'adam', 'adamw', or 'sgd'.
    #   loss_name: 'cross_entropy', 'nll' (needs output_activation='log_softmax'),
    #              or 'mse' (softmax probabilities vs one-hot labels)
    #   label_smoothing:
    #   normalize: scale values from 0-255 to 0-1
    # defaults reproduce first successful run above 95% test accuracy.
    def __init__(self, mName, mDescription,
                 hidden_sizes=(256,),
                 activation='relu',
                 output_activation='leaky_relu',
                 optimizer_name='adam',
                 loss_name='cross_entropy',
                 label_smoothing=0.0,
                 normalize=False):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.label_smoothing = label_smoothing
        self.normalize = normalize

        activations = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'tanh': nn.Tanh}
        output_activations = {'relu': nn.ReLU(), 'leaky_relu': nn.LeakyReLU(), 'log_softmax': nn.LogSoftmax(dim=1)}

        layers = []
        in_size = 784
        for size in hidden_sizes:
            layers.append(nn.Linear(in_size, size))
            layers.append(activations[activation]())
            in_size = size
        layers.append(nn.Linear(in_size, 10))
        if output_activation is not None:
            layers.append(output_activations[output_activation])

        self.layers = nn.Sequential(*layers)

    # it defines the forward propagation function for input x
    # this function will calculate the output layer by layer

    def forward(self, x):
        '''Forward propagation'''
        # runs every layer in self.layers in order; output is a (n, 10) tensor, one score per class
        return self.layers(x)

    # backward error propagation will be implemented by pytorch automatically
    # so we don't need to define the error backpropagation function here

    def train(self, X, y):
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError('unknown optimizer_name: ' + str(self.optimizer_name))

        if self.loss_name == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        elif self.loss_name == 'nll':
            loss_function = nn.NLLLoss()
        elif self.loss_name == 'mse':
            mse = nn.MSELoss()
            def loss_function(y_pred, y_true):
                return mse(torch.softmax(y_pred, dim=1), nn.functional.one_hot(y_true, num_classes=10).float())
        else:
            raise ValueError('unknown loss_name: ' + str(self.loss_name))
        #Getting accuracy metrics
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        X_tensor = torch.FloatTensor(np.array(X))
        y_tensor = torch.LongTensor(np.array(y))
        if self.normalize:
            X_tensor = X_tensor / 255.0
        n = X_tensor.size(0)

        # early stopping
        best_loss = float('inf')
        epochs_without_improvement = 0

        # per-epoch history
        self.history = {'epoch': [], 'loss': [], 'accuracy': []}

        #  mini batch gradient descent
        for epoch in range(self.max_epoch):
            # shuffle the instance order each epoch so batches differ
            perm = torch.randperm(n)

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_batch = X_tensor[idx]
                y_batch = y_tensor[idx]

                # get the output for this mini batch
                y_pred = self.forward(X_batch)
                # calculate the training loss
                train_loss = loss_function(y_pred, y_batch)


                optimizer.zero_grad() #clear old grad

                # do backpropagation to calculate the gradients
                train_loss.backward()

                # update the variables according to the optimizer and the gradients
                optimizer.step()

            # evaluate on the full training set at the end of each epoch
            with torch.no_grad():
                full_pred = self.forward(X_tensor)
                full_loss = loss_function(full_pred, y_tensor)
            accuracy_evaluator.data = {'true_y': y_tensor, 'pred_y': full_pred.max(1)[1]}
            train_accuracy = accuracy_evaluator.evaluate()
            print('Epoch:', epoch, 'Accuracy:', train_accuracy, 'Loss:', full_loss.item())

            self.history['epoch'].append(epoch)
            self.history['loss'].append(full_loss.item())
            self.history['accuracy'].append(train_accuracy)

            # early stopping: stop if loss hasn't improved by min_delta for `patience` epochs
            if full_loss.item() < best_loss - self.min_delta:
                best_loss = full_loss.item()
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print('Early stopping at epoch', epoch, '- no improvement in', self.patience, 'epochs')
                    break
    
    def test(self, X):
        # do the testing, and result the result
        X_tensor = torch.FloatTensor(np.array(X))

        if self.normalize:
            X_tensor = X_tensor / 255.0
        with torch.no_grad():
            y_pred = self.forward(X_tensor)

        return y_pred.argmax(1) #return highest prob classification
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
            