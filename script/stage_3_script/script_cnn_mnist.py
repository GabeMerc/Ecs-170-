from torch import nn

from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST
from local_code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy


result_path = "../../result/stage_3_result/MNIST/"

#data import

dataset_loader = Dataset_Loader(dName = "MNIST", dDescription = "Stage 3 MNIST")
dataset_loader.dataset_source_folder_path = "../../data/stage_3_data/"
dataset_loader.dataset_source_file_name = "MNIST"

data = dataset_loader.load()

trial_runs = {
    # Conv(1->32,kernel = 3,padding = 1) -> ReLU -> MaxPool2 -> Conv(32->64,kernel = 3,padding = 1) -> ReLU -> MaxPool2 -> Flatten -> Linear(128) -> linear(10), Loss = CrossEntropy, opt = Adam
    "mnist_baseline": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)]),

    # kernel size 5 instead of 3, padding 2 keeps the same output size
    "mnist_kernel5": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)]),

    # one conv block instead of two
    "mnist_1block": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)]),

    # average pooling instead of max pooling
    "mnist_avgpool": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.AvgPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)]),

    # stride 2, all convs no pooling
    "mnist_stride2": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
                       nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU()],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)]),

    # hidden layer dimension 512 instead of 128
    "mnist_hidden512": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(512), nn.ReLU(), nn.Linear(512, 10)]),

    # loss function MSE instead of CrossEntropy
    "mnist_mse": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)],
        loss_name = 'mse'),

    # three conv blocks instead of two
    "mnist_3block": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, 10)]),

    # activation LeakyReLU instead of ReLU
    "mnist_leakyrelu": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.LeakyReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.LeakyReLU(), nn.Linear(128, 10)]),

    # BatchNorm after each conv dropout before the output layer
    "mnist_bn_dropout": dict(
        conv_layers = [nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                       nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)],
        fc_layers = [nn.LazyLinear(128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 10)]),

}

summary_metrics = []

for name, configuration in trial_runs.items():
    CNN = Method_CNN_MNIST(name, "", **configuration)
    CNN.data = data
    results = CNN.run()

    loss_name = configuration.get('loss_name', 'cross_entropy')

    evaluator = Evaluate_Accuracy(name + " evaluator", "")
    evaluator.data = results
    scores = evaluator.evaluate_weighted_suite(plot_save_path = result_path + name + "_convergence.png", loss_name = loss_name)

    summary_metrics.append((name, len(results['history']['epoch']), scores))


    history = results['history']

    with open(result_path + name + "_results.txt", "w") as f:
        f.write("Run: " + name + "\n")
        f.write("Loss: " + loss_name + ", Optimizer: " + CNN.optimizer_name + "\n")
        f.write("Conv layers:\n" + str(CNN.conv_layers) + "\n")
        f.write("FC layers:\n" + str(CNN.fc_layers) + "\n\n")

        f.write("Training per epoch\n")
        for epoch, loss, acc in zip(history['epoch'], history['loss'], history['accuracy']):
            f.write(f"Epoch {epoch:<3}  Accuracy {acc:.4f}  Loss {loss:.4f}\n")

        f.write("\nTest results\n")
        f.write(f"Accuracy:           {scores['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {scores['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall:    {scores['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1:        {scores['f1_weighted']:.4f}\n")

#Summary table of every run
with open(result_path + "summary.txt", "w") as f:
    f.write(f"{'Run':<20}{'Epochs':>8}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}\n")
    for name, epochs, scores in summary_metrics:
        f.write(f"{name:<20}{epochs:>8}{scores['accuracy']:>10.4f}{scores['precision_weighted']:>11.4f}"
                f"{scores['recall_weighted']:>9.4f}{scores['f1_weighted']:>8.4f}\n")
