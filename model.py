import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy
from layers import MyLSTM


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, target_size, num_layers=1, check_lstm=False, dropout=None):
        super(LSTMClassifier, self).__init__()
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__target_size = target_size
        self.__num_layers = num_layers
        self.__check_lstm = check_lstm
        self.lstm = MyLSTM(input_size, hidden_size, num_layers=num_layers, batch_first=False, dropout=dropout)
        self.fc = nn.Linear(hidden_size, target_size)
        # 调库的lstm
        self.lib_lstm = None
        self.lib_fc = None
        # 若需要检查手动实现的lstm, 则初始化lib_lstm和lib_fc
        if check_lstm:
            self.lib_lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=False)
            self.lib_fc = nn.Linear(hidden_size, target_size)
            self.__sync_initial_parameter_values()

        # 用于保存最佳参数
        self.best_valid_acc = 0.0
        self.best_lstm = None
        self.best_fc = None

        self.__batch_size = 0
        self.__train_loss = []
        self.__valid_loss = []

    def forward(self, x, train=False):
        out, (h, c) = self.lstm(x, train=train)
        out = self.fc(out)
        return torch.sigmoid(out)

    def lib_forward(self, x):
        # 利用nn.LSTM进行前向传播
        out, (h, c) = self.lib_lstm(x)
        out = self.lib_fc(out)
        return torch.sigmoid(out)

    def __sync_initial_parameter_values(self):
        # 将lib_lstm, lib_fc和lstm, fc的初始参数设定为相同值
        with torch.no_grad():
            for i in range(self.__num_layers):
                exec('self.lib_lstm.weight_ih_l{} = nn.Parameter(copy.deepcopy(self.lstm.weight_ih{}))'.format(i, i))
                exec('self.lib_lstm.weight_hh_l{} = nn.Parameter(copy.deepcopy(self.lstm.weight_hh{}))'.format(i, i))
                exec('self.lib_lstm.bias_ih_l{} = nn.Parameter(copy.deepcopy(self.lstm.bias_ih{}))'.format(i, i))
                exec('self.lib_lstm.bias_hh_l{} = nn.Parameter(copy.deepcopy(self.lstm.bias_hh{}))'.format(i, i))

            self.lib_fc.weight = nn.Parameter(copy.deepcopy(self.fc.weight))
            self.lib_fc.bias = nn.Parameter(copy.deepcopy(self.fc.bias))

    def __check_result(self, x, target, y_score, loss, iter):
        # 对比MyLSTM的结果y_score, loss与nn.LSTM的结果y_score2, loss2
        y_score2 = self.lib_forward(x).transpose(0, 1)
        bce = nn.BCELoss()
        loss2 = bce(y_score2, target)
        loss2.backward()
        with torch.no_grad():
            print('iter {}, check: y_score err: {}, loss err: {}'.format(iter, (y_score - y_score2).norm(2).pow(2).data,
                                                                         (loss2 - loss).data))

    def predict(self, test_dataset):
        pred_list, loss, acc = self.__evaluate_dataset(test_dataset)
        return pred_list, acc

    def __evaluate_dataset(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.__batch_size)
        loss = 0.0
        acc = 0.0
        pred_list = []
        with torch.no_grad():
            for _, (data, target) in enumerate(data_loader):
                data, target = data.cuda(), target.cuda()
                y_score = self.forward(data.transpose(0, 1)).transpose(0, 1)
                bce = nn.BCELoss()
                loss = bce(y_score, target)
                loss += float(loss) * len(data) / len(dataset)
                y_score = y_score[:, -1]
                y_pred = torch.where(y_score >= 0.6, torch.ones_like(y_score), torch.zeros_like(y_score))
                pred_list.append(y_pred)
                y_target = target[:, -1]
                acc += torch.eq(y_pred, y_target).sum().item() / self.__target_size
            acc /= len(dataset)
        return pred_list, loss, acc

    def __plot(self):
        train_len = len(self.__train_loss)
        valid_len = len(self.__valid_loss)
        plt.plot(range(train_len), self.__train_loss, label='train loss')
        plt.plot(np.linspace(0, train_len, valid_len), self.__valid_loss, label='valid loss')
        plt.legend()
        plt.xlabel('iter times')
        plt.ylabel('loss')
        plt.savefig('loss.png')
        plt.show()

    def __updata_best_para(self, acc):
        if acc > self.best_valid_acc:
            self.best_valid_acc = acc
            self.__best_lstm = copy.deepcopy(self.lstm)
            self.__best_fc = copy.deepcopy(self.fc)

    def fit(self, train_data, valid_data, optimizer, batch_size, epoch, scheduler=None, verbose=False):
        self.__batch_size = batch_size
        self.__train_loss.clear()
        self.__valid_loss.clear()
        train_loader = DataLoader(train_data, batch_size=batch_size)
        for e in range(epoch):
            for idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                y_score = self.forward(data.transpose(0, 1), train=True).transpose(0, 1)
                bce = nn.BCELoss()
                loss = bce(y_score, target)
                # 打印每次迭代的loss
                if verbose:
                    print('iter {}: loss: {}'.format(idx, loss.data))
                # 反向传播，梯度下降
                optimizer.zero_grad()
                # 检查MyLSTM的结果
                if self.__check_lstm:
                    self.__check_result(data.transpose(0, 1), target, y_score.detach(), loss.detach(), idx)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            _, train_loss, train_acc = self.__evaluate_dataset(train_data)
            _, valid_loss, valid_acc = self.__evaluate_dataset(valid_data)
            self.__train_loss.append(train_loss)
            self.__valid_loss.append(valid_loss)
            self.__updata_best_para(valid_acc)
            print('Epoch {}: train loss:{}, train acc:{}, valid loss:{}, valid acc:{}'.format(
                e + 1, self.__train_loss[-1], train_acc, self.__valid_loss[-1], valid_acc))

        self.__plot()
        self.lstm = self.__best_lstm
        self.fc = self.__best_fc