import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False, dropout=None):
        super(MyLSTM, self).__init__()
        self.__input_size = input_size
        self.__hidden_size = hidden_size
        self.__num_layers = num_layers
        self.__batch_first = batch_first
        self.__dropout = None
        if dropout is not None:
            self.__dropout = nn.Dropout(dropout)

        # 定义参数
        gate_size = 4 * hidden_size
        self.__all_weights = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            w_ih = nn.Parameter(torch.Tensor(gate_size, layer_input_size))
            w_hh = nn.Parameter(torch.Tensor(gate_size, hidden_size))
            b_ih = nn.Parameter(torch.Tensor(gate_size))
            b_hh = nn.Parameter(torch.Tensor(gate_size))
            # 将权重参数添加至类属性，并将参数名称逐层记录在__all_weights中
            param_names = ['weight_ih{}', 'weight_hh{}', 'bias_ih{}', 'bias_hh{}']
            param_names = [x.format(layer) for x in param_names]
            layer_params = (w_ih, w_hh, b_ih, b_hh)
            for name, param in zip(param_names, layer_params):
                setattr(self, name, param)
            self.__all_weights.append(param_names)

        # 初始化参数
        self.__init_weights()

    def __init_weights(self):
        stdv = 1.0 / math.sqrt(self.__hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hx=None, train=False):
        # 转化为(seq_len, batch_size, feature_num)维度的数据
        if self.__batch_first:
            input = input.transpose(0, 1)
        seq_len = input.shape[0]
        batch_size = input.shape[1]
        if hx is None:
            zeros = torch.zeros(self.__num_layers, batch_size, self.__hidden_size, device=input.device)
            hx = (zeros, zeros)
        output = []
        # i0代表前一时刻，i1代表当前时刻
        # h_i0, c_i0, h_i1, c_i1 维度均为 (layer_num, batch_size, hidden_size)
        h_i0, c_i0 = hx
        h_i0 = [x for x in h_i0]
        c_i0 = [x for x in c_i0]
        # 对序列进行遍历
        for x_i, step_i in zip(input, range(seq_len)):
            # 初始化为空Tensor，对layer遍历时逐层拼接至(layer_num, batch_size, hidden_size)
            h_i1 = []
            c_i1 = []
            for layer in range(self.__num_layers):
                # 获取当前层对应的单元的权重参数
                w_ih, w_hh, b_ih, b_hh = (getattr(self, i) for i in self.__all_weights[layer])
                # 若当前是第一层，则输入为input，否则为当前时刻i1的上一层输出h
                x = x_i if layer == 0 else h_i1[-1]
                # 计算四个门的输出
                gates_out = F.linear(x, w_ih, b_ih) + F.linear(h_i0[layer], w_hh, b_hh)
                i, f, g, o = gates_out.chunk(4, dim=1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                # 计算当前时刻当前单元的c和h
                c_i1_layer = f * c_i0[layer] + i * g
                h_i1_layer = o * torch.tanh(c_i1_layer)
                if train and self.__dropout is not None and layer != self.__num_layers - 1:
                    h_i1_layer = self.__dropout(h_i1_layer)
                h_i1.append(h_i1_layer)
                c_i1.append(c_i1_layer)
            h_i0, c_i0 = h_i1, c_i1
            output.append(h_i0[-1])
        # 将output, h_i0, c_i0拼接为一个Tensor
        output = torch.stack(output, dim=0)
        h_i0 = torch.stack(h_i0, dim=0)
        c_i0 = torch.stack(c_i0, dim=0)
        if self.__batch_first:
            output.transpose_(0, 1)
        return output, (h_i0, c_i0)