import torch

# paths to the training, validation and test data sets
train_data_dir = ('C:\\Users\\Meet Sinojia\\Downloads\\Eye-Disease-Classification-Few-Shot-Learning-master\\Eye-Disease-Classification-Few-Shot-Learning-master//training_data')
validation_data_dir = 'C:\\Users\\Meet Sinojia\\Downloads\\Eye-Disease-Classification-Few-Shot-Learning-master\\Eye-Disease-Classification-Few-Shot-Learning-master//validation_data'
test_data_dir = 'C:\\Users\\Meet Sinojia\\Downloads\\Eye-Disease-Classification-Few-Shot-Learning-master\\Eye-Disease-Classification-Few-Shot-Learning-master//training_data'


MEAN = torch.tensor([0.5507, 0.4053, 0.3529]) 
STD = torch.tensor([0.2550, 0.2261, 0.2246])