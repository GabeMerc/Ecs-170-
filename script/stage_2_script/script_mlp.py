from locale import normalize

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy

#Training Import:
dataset_train_loader = Dataset_Loader(dName = "stage2_train",dDescription = "Stage 2 train dataset")
dataset_train_loader.dataset_source_folder_path = "../../data/stage_2_data/"
dataset_train_loader.dataset_source_file_name = "train.csv"

dataset_train = dataset_train_loader.load()


#Testing Import
dataset_test_loader = Dataset_Loader(dName = "stage2_test",dDescription = "Stage 2 test dataset")
dataset_test_loader.dataset_source_folder_path = "../../data/stage_2_data/"
dataset_test_loader.dataset_source_file_name = "test.csv"

dataset_test = dataset_test_loader.load()


#Declaring the MLP & setting up data:


#First test, now set to default settings.
data = {"train" : dataset_train, "test" : dataset_test}

'''
MLP = Method_MLP(mName = "Stage 2 MLP", mDescription = "MLP trained on stage 2 provided data")

MLP.data = data

results = MLP.run()

evaluator = Evaluate_Accuracy("test evaluator", "")
evaluator.data = results

print("Test Accuracy:", evaluator.evaluate())

# eval suite
evaluator.evaluate_weighted_suite(plot_save_path = "../../result/stage_2_result/convergence.png")

'''

trial_runs = {
    # normalized -> Linear(784,256) -> ReLU -> Linear(256,10) -> LeakyReLU, Loss = CrossEntropy, opt = Adam
    "stage2_baseline": dict(),
    # normalized -> Linear(784,256) -> ReLU -> Linear(256,10), Loss = CrossEntropy, opt = Adam
    "stage2_normalize": dict(normalize=True, output_activation = None),
    # normalized -> Linear(784,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,10), Loss = CrossEntropy, opt = Adam
    "stage2_2layers": dict(normalize=True, output_activation = None, hidden_sizes=(512,256)),
    # normalized -> Linear(784,512) -> ReLU -> Linear(512,256) -> ReLU -> Linear(256,128) -> ReLU -> Linear(128,10), Loss = CrossEntropy, opt = Adam
    "stage2_3layers": dict(normalize=True, output_activation=None, hidden_sizes=(512,256,128)),
    # normalized -> Linear(784,256) -> ReLU -> Linear(256,10), Loss = CrossEntropy, opt = AdamW
    "stage2_adamw": dict(normalize = True, output_activation=None, optimizer_name = 'adamw'),
    # normalized -> Linear(784,256) -> ReLU -> Linear(256,10) -> LogSoftmax, Loss = NLL, opt = Adam
    "stage2_nll": dict(normalize=True, output_activation='log_softmax', loss_name='nll'),
    # normalized -> Linear(784,256) -> ReLU -> Linear(256,10), Loss = MSE, opt = Adam
    "stage2_mse": dict(normalize=True, output_activation=None, loss_name='mse')
}

summary_metrics = []

for name, configuration in trial_runs.items():
    MLP = Method_MLP(name, "", **configuration)
    MLP.data = data
    results = MLP.run()

    evaluator = Evaluate_Accuracy(name + " evaluator ", "")
    evaluator.data = results
    scores = evaluator.evaluate_weighted_suite(plot_save_path = "../../result/stage_2_result/" + name + "_convergence.png")

    summary_metrics.append((name,configuration,len(results['history']['epoch']),scores))

    #Save results
    history = results['history']

    with open("../../result/stage_2_result/" + name + "_results.txt", "w") as f:
        f.write("Run: " + name + "\n")
        f.write("Configuration: " + str(configuration) + "\n\n")

        f.write("Training per epoch\n")
        for epoch, loss, acc in zip(history['epoch'], history['loss'], history['accuracy']):
            f.write(f"Epoch {epoch:<3}  Accuracy {acc:.4f}  Loss {loss:.4f}\n")

        f.write("\nTest results\n")
        f.write(f"Accuracy:           {scores['accuracy']:.4f}\n")
        f.write(f"Weighted Precision: {scores['precision_weighted']:.4f}\n")
        f.write(f"Weighted Recall:    {scores['recall_weighted']:.4f}\n")
        f.write(f"Weighted F1:        {scores['f1_weighted']:.4f}\n")


