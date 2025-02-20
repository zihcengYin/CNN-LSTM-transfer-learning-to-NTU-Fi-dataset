import os
import glob
import torch
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from NTU_Fi_CNN_LSTM import NTU_Fi_CNN_LSTM


class CSI_Dataset(Dataset):
    def __init__(self, root_dir, modal='CSIamp', transform=None):
        """初始化数据集

        Args:
            root_dir (str): 数据集根目录，包含子文件夹（每个子文件夹代表一个类别）
            modal (str): 数据模态，例如 'CSIamp'
            transform (callable, optional): 数据预处理函数
        """
        self.root_dir = root_dir
        self.modal = modal
        self.transform = transform

        # 递归搜索子文件夹中的.mat文件
        self.data_list = []
        self.labels = []
        self.category = {}  # 类别名称到类别索引的映射

        # 遍历所有子文件夹
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue  # 跳过文件，只处理文件夹

            # 将该类别添加到映射
            self.category[class_name] = idx

            # 遍历该类别文件夹中的所有.mat文件
            for mat_file in glob.glob(os.path.join(class_dir, '*.mat')):
                self.data_list.append(mat_file)
                self.labels.append(idx)

        # 检查是否为空
        if len(self.data_list) == 0:
            raise ValueError(f"No .mat files found in {root_dir}")

        print(f"Found {len(self.data_list)} .mat files in {root_dir} with {len(self.category)} classes.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 加载.mat文件
        mat_file = self.data_list[idx]
        label = self.labels[idx]
        x = sio.loadmat(mat_file)[self.modal]

        # 检查数据是否为空
        if x is None:
            raise ValueError(f"Data is empty in {mat_file}")

        # 归一化
        x = (x - 42.3199) / 4.9802

        # 采样：2000 -> 500
        x = x[:, ::4]
        x = x.reshape(3, 114, 500)

        if self.transform:
            x = self.transform(x)

        x = torch.FloatTensor(x)
        return x, label

def create_dataloader(dataset_name, root):
    print(f"Root absolute path: {os.path.abspath(root)}")
    classes = {'NTU-Fi-HumanID': 14, 'NTU-Fi_HAR': 6}

    if dataset_name == 'NTU-Fi-HumanID':
        print('using dataset: NTU-Fi-HumanID')
        num_classes = classes['NTU-Fi-HumanID']
        train_path = os.path.join(root, 'train_amp')  # 直接指向 train_amp 文件夹
        test_path = os.path.join(root, 'test_amp')  # 直接指向 test_amp 文件夹
        print(f"Train path: {train_path}")
        print(f"Test path: {test_path}")

        train_loader = torch.utils.data.DataLoader(
            dataset=CSI_Dataset(train_path),
            batch_size=64, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=CSI_Dataset(test_path),
            batch_size=64, shuffle=False
        )

        print("using finetune")
        model = NTU_Fi_CNN_LSTM(num_classes=num_classes)
        train_epoch = 50
        return train_loader, test_loader

    elif dataset_name == 'NTU-Fi_HAR':
        print('using dataset: NTU-Fi_HAR')
        num_classes = classes['NTU-Fi_HAR']
        train_path = os.path.join(root, 'train_amp')  # 直接指向 train_amp 文件夹
        test_path = os.path.join(root, 'test_amp')  # 直接指向 test_amp 文件夹
        print(f"Train path: {train_path}")
        print(f"Test path: {test_path}")

        train_loader = torch.utils.data.DataLoader(
            dataset=CSI_Dataset(train_path),
            batch_size=64, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=CSI_Dataset(test_path),
            batch_size=64, shuffle=False
        )
        print("using model: CNN+LSTM")
        model = NTU_Fi_CNN_LSTM(num_classes=num_classes)
        train_epoch = 50
        return train_loader, test_loader
