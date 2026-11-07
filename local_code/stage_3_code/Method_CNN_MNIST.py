

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
import torch
from torch import nn
import numpy as np



class Method_CNN_MNIST(method, nn.Module):
    data = None
    max_epoch = 500
    learning_rate = 1e-3
    batch_size = 128
    patience = 3
    min_delta = 1e-4
    seed = 21


    def __init__(self, mName, mDescription,
                 conv_layers = None,
                 fc_layers = None,
                 optimizer_name = 'adam',
                 loss_name = 'cross_entropy',
                 normalize = True):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)


        torch.manual_seed(self.seed)

        np.random.seed(self.seed)

        self.optimizer_name = optimizer_name
        self.loss_name = loss_name
        self.normalize = normalize

        if conv_layers is None:
            raise ValueError("Conv layers is unpopulated")
        if fc_layers is None:
            raise ValueError("fc_layers is unpopulated")

        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(*fc_layers)

        #testing it runs
        with torch.no_grad():
            self.forward(torch.zeros(1,1,28,28))

        #use the Apple GPU, otherwise cpu
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        #forward pass
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.fc_layers(x)

    def train_model(self, X, y):
        #Optimizer

        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        elif self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr = self.learning_rate)
        #elif self.optimizer_name == 'sgd':
         #   optimizer = torch.optim.SGD(self.parameters(), lr = self.learning_rate, momentum= .9)
        else:
            raise ValueError("Unknown optmizer" + str(self.optimizer_name))

        #Loss function

        if self.loss_name == 'cross_entropy':
            loss_function = nn.CrossEntropyLoss()
        elif self.loss_name == 'nll':
            loss_function = nn.NLLLoss()
        elif self.loss_name == 'mse':
            mse = nn.MSELoss()
            def loss_function(y_pred, y_true):
                return mse(torch.softmax(y_pred, dim = 1), nn.functional.one_hot(y_true,num_classes=10).float()) #just the mse between predicted probability and 1.
        else:
            raise ValueError('unknown loss name' + str(self.loss_name))



        X_tensor = torch.from_numpy(X)

        y_tensor = torch.from_numpy(y)

        if self.normalize:
            X_tensor = X_tensor / 255.0 #scale between 1 - 0
        n = X_tensor.size(0) #gets the first shape so in this case the num of images.

        train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tensor, y_tensor), batch_size = self.batch_size, shuffle = True)

        best_loss = float('inf')

        epochs_without_improvement = 0

        self.history = {'epoch': [], 'loss': [], 'accuracy': []}

        for epoch in range(self.max_epoch):
            self.train() #train mode is on
            total_loss = 0.0
            total_correct = 0

            for X_batch, y_batch in train_loader:
                #move batch onto the same device as the model
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_pred = self.forward(X_batch)
                train_loss = loss_function(y_pred, y_batch)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_loss += train_loss.item() * X_batch.size(0)
                total_correct += (y_pred.argmax(1) == y_batch).sum().item()

            epoch_loss = total_loss / n
            epoch_accuracy = total_correct / n
            print("Epoch: ", epoch, "Accuracy: ", epoch_accuracy, "Loss: ", epoch_loss)

            self.history['epoch'].append(epoch)
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_accuracy)

            if epoch_loss < best_loss - self.min_delta:
                best_loss = epoch_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print('Early stopping at epoch', epoch, ' no improvement in', self.patience, 'epochs')
                    break

    def test(self, X):
        self.eval()

        X_tensor = torch.from_numpy(X)

        if self.normalize:
            X_tensor = X_tensor / 255.0

        preds = []

        with torch.no_grad():
            for i in range(0, X_tensor.size(0), self.batch_size):
                X_batch = X_tensor[i:i + self.batch_size].to(self.device)
                #.cpu() so the evaluator (sklearn) can read the predictions
                preds.append(self.forward(X_batch).argmax(1).cpu())
        return torch.cat(preds)


    def run(self):
        print('method running...')
        print('start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('start testing...')
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y'], 'history': self.history}
