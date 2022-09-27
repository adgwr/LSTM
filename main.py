import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from dataset import MetalPriceDataset
from model import LSTMClassifier
import os


def set_random_seed(seed_value=1024):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def main(pred_day, seq_len, hidden_size, epoch, lr, num_layers=1, check_lstm=False, verbose=False, dropout=None, extra_fea=False):
    # 读取数据, 利用训练集的数据归一化测试集和验证集
    train_data = MetalPriceDataset(data_type='train', pred_day=pred_day, seq_len=seq_len,
                                   data_scaler=StandardScaler())
    valid_data = MetalPriceDataset(data_type='valid', pred_day=pred_day, seq_len=seq_len,
                                   data_scaler=train_data.data_scaler)
    test_data = MetalPriceDataset(data_type='test', pred_day=pred_day, seq_len=seq_len,
                                  data_scaler=train_data.data_scaler)

    # 初始化模型
    model = LSTMClassifier(input_size=train_data[0][0].shape[1], hidden_size=hidden_size, target_size=6, num_layers=num_layers,
                           check_lstm=check_lstm, dropout=dropout)
    device = torch.device('cuda')
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.5)

    # 训练
    model.fit(train_data, valid_data, optimizer=optimizer, batch_size=64, epoch=epoch,
              scheduler=scheduler, verbose=verbose)
    # 计算并打印测试集上的准确率
    _, acc = model.predict(test_data)
    print('predict_{}d: test acc: {}'.format(pred_day, acc))
    return acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    set_random_seed()

    # acc_1d = main(pred_day=1, seq_len=20, hidden_size=100, epoch=1, lr=0.01,
    #               num_layers=3, check_lstm=True, verbose=True)

    acc_1d = main(pred_day=1, seq_len=20, hidden_size=100, epoch=100, lr=0.01,
                  num_layers=3, check_lstm=False, verbose=False, dropout=0.3)
    acc_20d = main(pred_day=20, seq_len=20, hidden_size=100, epoch=100, lr=0.01,
                   num_layers=2, check_lstm=False, verbose=False, dropout=0.3)
    acc_60d = main(pred_day=60, seq_len=60, hidden_size=100, epoch=100, lr=0.01,
                   num_layers=3, check_lstm=False, verbose=False, dropout=0.3)

    print('acc on test data: 1d: {}, 20d: {}, 60d:{}'.format(acc_1d, acc_20d, acc_60d))
    print('average acc on test data: {}'.format((acc_1d + acc_20d + acc_60d) / 3))
