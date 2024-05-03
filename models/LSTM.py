import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from typing import List
from utils import show_parameter


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super(ConvLSTMCell, self).__init__()

        # assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = 3
        self.num_features = 4

        self.padding = int((self.kernel_size - 1) / 2)
        # 遗忘门的 neural network layer 构成
        self.Wfx = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wfh = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # 输入门的 neural network layer 构成
        self.Wix = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wih = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # 激活单元的 neural network layer 构成
        self.Wcx = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Wch = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        # 输出门的 neural network layer 构成
        self.Wox = nn.Conv1d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Woh = nn.Conv1d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wic = None
        self.Wfc = None
        self.Woc = None

    def forward(self, x, h, c):
        # 遗忘门
        cf = torch.sigmoid(self.Wfx(x) + self.Wfh(h) + c * self.Wfc)
        # 更新门
        ci = torch.sigmoid(self.Wix(x) + self.Wih(h) + c * self.Wic)
        # 前一时刻*forget+现一时刻*input
        cc = cf * c + ci * torch.tanh(self.Wcx(x) + self.Wch(h))
        # 输出门
        co = torch.sigmoid(self.Wox(x) + self.Woh(h) + cc * self.Woc)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wic is None:
            self.Wic = nn.Parameter(torch.zeros(1, hidden, shape[0])).cuda()
            self.Wfc = nn.Parameter(torch.zeros(1, hidden, shape[0])).cuda()
            self.Woc = nn.Parameter(torch.zeros(1, hidden, shape[0])).cuda()
        # else:
        #  assert shape[0] == self.Wic.size()[2], 'Input Length Mismatched!'
        return (torch.zeros((batch_size, hidden, shape[0])).cuda(),
                torch.zeros((batch_size, hidden, shape[0])).cuda())


class ConvLSTM(nn.Module):
    # input_channels corresponds to the first input feature map
    # hidden state is 2 list of succeeding lstm layers.
    def __init__(self, batch_size, input_channels, effective_step):
        super(ConvLSTM, self).__init__()
        hidden_channels = [32]
        self.input_channels = [input_channels] + hidden_channels
        # print("input:", [input_channels] + hidden_channels)
        self.hidden_channels = hidden_channels
        self.num_layers = len(hidden_channels)
        self.batch_size = batch_size
        self.step = 10
        self.effective_step = effective_step
        self._all_layers = []
        # num_layers: 纵向有多少个时间单元
        for i in range(self.num_layers):
            name = 'cell{}'.format(i)
            cell = ConvLSTMCell(self.input_channels[i], self.hidden_channels[i])
            # 将name的值变成cell
            setattr(self, name, cell)
            self._all_layers.append(cell)
        self.final_layer = nn.Sequential(
            nn.Conv1d(hidden_channels[-1], out_channels=1,
                      kernel_size=3, padding=1),
            nn.Dropout(0.5),
            nn.Linear(1 * 3840, 64),
            nn.Dropout(0.5),
            nn.Linear(64, 2)
        )

    def get_reconstruction_loss(self, *args, labels) -> dict:
        recons = args[0]
        recons_loss_ = nn.SmoothL1Loss()
        # recons_loss = recons_loss_(recons.view(4, -1), gt.view(4, -1))  # reshape成20行，列数由行数决定
        recons_loss = recons_loss_(recons, labels)
        loss = recons_loss
        return loss

    def forward(self, input):
        internal_state = []
        # outputs = torch.zeros(self.batch_size, self.hidden_channels[-1], input.shape[2]).cuda()
        outputs = []
        # step:横向有多少个时间步
        for step in range(self.step):
            # print(step)
            x = input
            for i in range(self.num_layers):
                # all cells are initialized in the first step
                name = 'cell{}'.format(i)
                if step == 0:
                    bsize, _, length = x.size()
                    (h, c) = getattr(self, name).init_hidden(batch_size=bsize, hidden=self.hidden_channels[i],
                                                             shape=[length])
                    internal_state.append((h, c))  # 纵向传播
                # do forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)
            # only record effective steps
            if step < self.effective_step:
                # print(x.shape())
                outputs.append(x)
            # outputs = np.array(outputs)
            # outputs.to(torch.device("cuda"))
        outputs = self.final_layer(outputs[-1])
        return outputs, (x, new_c)


if __name__ == '__main__':
    # gradient check
    batch_size = 10
    model = ConvLSTM(batch_size, input_channels=1, effective_step=2).cuda()
    loss_fn = torch.nn.MSELoss()
    show_parameter(model)
    input = Variable(torch.randn(batch_size, 1, 3840)).cuda()
    target = Variable(torch.randn(batch_size, 1, 2)).double().cuda()

    output = model(input)
    output = output[0][0].double()
    res = torch.autograd.gradcheck(loss_fn, (output, target), eps=1e-6, raise_exception=True)
    print(model)
    print(output.size())
