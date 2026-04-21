import os
import sys
import time
import yaml
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data.dataset import ConcatDataset
# from torch.utils.tensorboard import SummaryWriter
from models.MobileNet.MobileNetV1 import MobileNetV1
from models.MobileNet.MobileNetV2 import MobileNetV2
from models.MobileNet.MobileNetV3 import MobileNetV3
from models.Abaltion.DeepLabV3_6.DeepLabV3Plus import DeepLab6
from models.ae_cnn import VanillaVAE
# from models.ae_cnn import VanillaAE
from utils.ecg_dataset import *
from torchsummary import summary

workdir = os.path.abspath('.')
# print(workdir)
sys.setrecursionlimit(2000)
config_filename = os.path.join(workdir, 'configs/ecg_ae.yml')
# print(config_filename)
with open(config_filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

torch.manual_seed(config['exp_params']['manual_seed'])
device = torch.device("cuda")
# model = MobileNetV3(1, config['model_params']['latent_dim']).to(device)
model = DeepLab6(backbone='mobilenet', output_stride=16).to(device)
concat_dataset = []
for test in os.listdir(os.path.join(workdir, 'dataset/ecg_train_set/')):
    train_set = MyDataset(os.path.join(workdir, 'dataset/ecg_train_set/') + test)
    concat_dataset = ConcatDataset([train_set, concat_dataset])
dataloader = torch.utils.data.DataLoader(concat_dataset, batch_size=128
                                         , shuffle=True, num_workers=0)


def train(model, dataloader, save_name):
    epoch_start_time = time.time()
    # writer = SummaryWriter('./runs/batch_128')
    for epoch in range(config['trainer_params']['max_epochs']):
        running_loss = 0
        if epoch < 70:
            my_lr = 1e-3
        elif epoch < 100:
            my_lr = 1e-4
        elif epoch < 200:
            my_lr = 1e-5
        elif epoch < 400:
            my_lr = 1e-6
        elif epoch < 800:
            my_lr = 1e-7
        else:
            my_lr = 1e-8

        optimizer = optim.Adam(model.parameters(), lr=my_lr)
        steps = 10
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
        for i, data in enumerate(dataloader, 0):
            # get the inputs;
            inputs_i, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            inputs_i = inputs_i.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(inputs_i)
            loss = model.loss_function(*outputs, gt=labels)
            loss.backward()
            optimizer.step()

            # visual loss
            running_loss += loss.detach().item()
            if (i + 1) % 5 == 0:
                scheduler.step()
        #  writer.add_scalar('Loss', loss, epoch * steps + steps)
        # print statistics
        print('[%d, %5d] loss: %.4f learning_rate: %.4f ' % (epoch + 1, i + 1, running_loss, my_lr))
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(workdir, 'evaluate/') + save_name + '.pth')
            print('time elapsed: ', (time.time() - epoch_start_time) // 3600, 'h ',
                  (time.time() - epoch_start_time) % 3600)
        if loss < 1e-6:
            break
    torch.save(model.state_dict(), os.path.join(workdir, 'evaluate/') + save_name + '.pth')
    print('Finished Training')


if __name__ == "__main__":
    # pip install --upgrade git+https://github.com/Lyken17/pytorch-OpCounter.git
    # from thop import profile
    # flops, params = profile(model, input_size=(1, 640))
    # print('Total params: %.2fM' % (params / 1000000.0))
    # print('Total flops: %.2fM' % (flops / 1000000.0))
    # summary(model, input_size=(1, 640))
    train(model, dataloader, 'ecg_deeplab_lqz9')
    # print(model)

