import os


class Config:
    # 路径配置
    ROOT_DIR = r"C:\Users\25442\OneDrive\Desktop\study\dataset"
    HAR_PATH = os.path.join(ROOT_DIR, "NTU-Fi_HAR")
    HUMANID_PATH = os.path.join(ROOT_DIR, "NTU-Fi-HumanID")

    # 训练参数
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    NUM_EPOCHS_PRETRAIN = 50
    NUM_EPOCHS_FINETUNE = 30

    # 模型参数
    CNN_FILTERS = [16, 32, 64]
    LSTM_HIDDEN_SIZE = 128
    DROPOUT_RATE = 0.3

    @staticmethod
    def setup():
        # 创建必要目录
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("logs", exist_ok=True)