import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
from einops import rearrange
from augumentation import Rotate
from torch.utils.data import random_split

# Class read npy and pickle file to make data and label in couple
class FeederINCLUDE(Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        window_size: The length of the output sequence
    """
    def __init__(self, data_path: Path, label_path: Path, transform = None):
        super(FeederINCLUDE, self).__init__
        self.data_path = data_path
        self.label_path = label_path
        self.transform = transform
        self.load_data()
    
    def load_data(self):
        # data: N C V T M
        # Load label with numpy
        self.label = np.load(self.label_path)
        # load data
        self.data = np.load(self.data_path)     
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __getitem__(self, index):
        """
        Input shape (N, C, V, T, M)
        N : batch size
        C : numbers of features
        V : numbers of joints (as nodes)
        T : numbers of frames
        M : numbers of people (should delete)
        
        Output shape (C, V, T, M)
        C : numbers of features
        V : numbers of joints (as nodes)
        T : numbers of frames
        label : label of videos
        """
        data_numpy = torch.tensor(self.data[index]).float()
        # Delete one dimension
        # data_numpy = data_numpy[:, :, :2]
        # data_numpy = rearrange(data_numpy, ' t v c 1 -> c t v 1')
        label = self.label[index]
        p = random.random()
        if self.transform and p > 0.5: 
            data_numpy, label = self.transform(data_numpy, label)
        return data_numpy, label
    
    def __len__(self):
        return len(self.label)
    
if __name__ == '__main__':
    file, label = np.load("wsl100_train_data_preprocess.npy"), np.load("wsl100_train_label_preprocess.npy")
    print(file.shape, label.shape)
    data = FeederINCLUDE(data_path=f"wsl100_train_data_preprocess.npy", label_path=f"wsl100_train_data_preprocess.npy",
                            transform=None)
    # test_dataset = FeederINCLUDE(data_path=f"data/vsl100_test_data_preprocess.npy", label_path=f"data/vsl100_test_label_preprocess.npy")
    # valid_dataset = FeederINCLUDE(data_path=f"data/vsl100_valid_data_preprocess.npy", label_path=f"data/vsl100_valid_label_preprocess.npy")
    # data = FeederINCLUDE(data_path=f"data/vsl100_test_data_preprocess.npy", label_path=f"data/vsl100_test_label_preprocess.npy", 
                        # transform=None)
    print(data.N, data.C, data.T, data.V, data.M)
    print(data.data.shape)
    print(data.__len__())
