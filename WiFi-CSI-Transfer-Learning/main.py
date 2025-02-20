import argparse
from DataLoader import create_dataloader
from NTU_Fi_CNN_LSTM import NTU_Fi_CNN_LSTM
from Transfer_Learning_CNN_LSTM import TransferLearner
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pretrain_har():
    """在HAR数据集上预训练模型"""
    # 加载HAR数据
    train_loader, test_loader = create_dataloader(
        'NTU-Fi_HAR',
        root="C:\\Users\\25442\\OneDrive\\Desktop\\study\\dataset"
    )

    # 初始化模型
    model = NTU_Fi_CNN_LSTM(num_classes=6).to(device)  # HAR有6类
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练参数
    best_acc = 0.0
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        # 训练阶段
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total

        print(f"Epoch [{epoch + 1}/50] Loss: {total_loss / len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "har_pretrained.pth")

    print(f"Pre-training finished. Best Accuracy: {best_acc:.2f}%")


def transfer_learning():
    """迁移到HumanID数据集"""
    # 加载HumanID数据
    train_loader, test_loader = create_dataloader(
        'NTU-Fi-HumanID',
        root="C:\\Users\\25442\\OneDrive\\Desktop\\study\\dataset\\NTU-Fi-HumanID"
    )

    # 初始化迁移学习器
    transfer = TransferLearner(
        source_model_path="har_pretrained.pth",
        target_classes=14  # HumanID有14类
    )

    # 执行微调
    transfer.finetune(train_loader, test_loader, epochs=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "transfer"], required=True,
                        help="选择模式：pretrain（在HAR上预训练）或 transfer（迁移到HumanID）")
    args = parser.parse_args()

    if args.mode == "pretrain":
        pretrain_har()
    elif args.mode == "transfer":
        transfer_learning()