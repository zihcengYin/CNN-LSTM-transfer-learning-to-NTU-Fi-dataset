import torch
import torch.nn as nn
from NTU_Fi_CNN_LSTM import NTU_Fi_CNN_LSTM
from torch.optim import AdamW
from tqdm import tqdm
from DataLoader import create_dataloader


class TransferLearner:
    def __init__(self, source_model_path, target_classes, freeze_cnn=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化基础模型
        self.base_model = NTU_Fi_CNN_LSTM(num_classes=6)  # 源任务类别数
        self.base_model.load_state_dict(torch.load(source_model_path))
        self.base_model = self.base_model.to(self.device)

        # 冻结CNN层（假设CNN部分是encoder）
        if freeze_cnn:
            for param in self.base_model.encoder.parameters():  # 修改为冻结encoder
                param.requires_grad = False

        # 替换分类层
        in_features = self.base_model.classifier[1].in_features  # 获取原始分类层的输入维度
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(in_features, target_classes),  # 调整输出层以适应新任务的类别数
            nn.Softmax(dim=1)  # 添加Softmax以输出概率
        )

        # 优化器只训练非冻结层
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.base_model.parameters()),  # 只优化需要梯度的参数
            lr=1e-4,
            weight_decay=1e-5
        )

        self.criterion = nn.CrossEntropyLoss()

    def finetune(self, train_loader, val_loader, epochs=50):
        best_acc = 0.0
        for epoch in range(epochs):
            # 训练阶段
            self.base_model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1} Training"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.base_model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * inputs.size(0)

            # 验证阶段
            val_acc = self.evaluate(val_loader)
            train_loss = train_loss / len(train_loader.dataset)

            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

            # 保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.base_model.state_dict(), "best_transfer_model.pth")

        print(f"Best Validation Accuracy: {best_acc:.2f}%")
        return self.base_model

    def evaluate(self, test_loader):
        self.base_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.base_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total
