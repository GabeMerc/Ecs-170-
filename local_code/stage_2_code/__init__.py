from Dataset_Loader import Dataset_Loader
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




