from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image

class CIFAR100DIRTY_TEST(Dataset):
    def __init__(self, csv_path):
        self.transformations = transforms.Compose([transforms.CenterCrop(32),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
        self.data_info = pd.read_csv(csv_path, header=None)
        self.image_arr = np.asarray(self.data_info.iloc[:, 0])
        self.label_arr = np.asarray(self.data_info.iloc[:, 1])
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):  # returns the data and labels. This function is called from dataloader like this
        single_image_name = self.image_arr[index]
        img_as_img = Image.open(single_image_name)
        img_as_tensor = self.transformations(img_as_img)
        single_image_label = np.int(self.label_arr[index])

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len

if __name__ == '__main__':
    cifar100_dirty = CIFAR100DIRTY_TEST('/home/mhha/A2S/cifar100_test_targets.csv')
""""""