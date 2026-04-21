import argparse
import os.path
import re
from matplotlib import pyplot as plt
from scipy.io import savemat
import torch
from models.Abaltion.DeepLabV3_6.DeepLabV3Plus import DeepLab6
from utils import MyDataset, get_network
from score import evaluate
# from utils.split import merge_excel
import time
from models.Autoencoder import Autoencoder
from models.resnet import resnet34, resnet50
from utils import merge_excel
from models.rnn import lstm, gru

parser = argparse.ArgumentParser()

parser.add_argument("--net", default='ae128', type=str,
                    help="display a square of a given number")
parser.add_argument('-gpu', action='store_true',
                    default=False, help='use gpu or not')
args = parser.parse_args()


def plot_result(i, test_set, user, scg_label, scg_noisy, *arg):
    test_length = test_set[0].shape[-1]
    scg_label = scg_label.reshape(test_length, -1)
    scg_noisy = scg_noisy.reshape(test_length, -1)
    plt.plot(range(len(scg_label)), scg_label + 1, color="orange", linewidth=1.0, label="GroundTruth")
    plt.plot(range(test_length), arg[0] - 0.1, color="blue", linewidth=1.0, label="ECG_recovery")
    # plt.plot(range(test_length), args[1] - 1, color="green", linewidth=1.0, label="dplab_rec")
    plt.plot(range(len(scg_label)), scg_noisy - 1, color="red", linewidth=1.0, label="SCG")
    plt.legend(loc='upper right')

    plt.title(user + str(i))
    image_path = './result/ecg_image/' + user + "/"
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    # plt.savefig(image_path + str(i) + '.jpg')
    # plt.show()


def tst(model1, model2, user):
    test_dir = 'dataset/ecg_test_set/'
    listdir = os.listdir(test_dir)
    listdir.reverse()
    patten = "all"
    excel_path = "result/" + patten + "/rec_data/" + user + '/'
    if not os.path.exists(excel_path):
        os.makedirs(excel_path)
    for test in listdir:
        test_user = re.sub("_test", '', test)
        # print(test_user)
        if test_user == user:
            test_set = MyDataset(test_dir + test)
            savePath = "result/" + patten + "/recovery_data/gru/"
            evalPath = re.sub("recovery", 'rec', savePath)
            savePath = savePath + user + '/'
            dataloader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
            arg = []
            for i, data in enumerate(dataloader, 0):
                scg, scg_label = data
                scg_label = scg_label.squeeze().detach().numpy()
                outputs = model1(scg)
                scg_rec = outputs[0].squeeze().detach().numpy()
                arg.append(scg_rec)
                outputs2 = model2(scg)
                scg_rec2 = outputs2[0].squeeze().detach().numpy()
                arg.append(scg_rec)
                # assert scg_rec.shape == scg_rec2.shape
                plot_result(i, data, user, scg_label, scg, *arg)
                # print(i)
                arg = []
                mkdic = {"ecg": scg_label,
                         'scg': scg.squeeze().detach().numpy(),
                         'scg_rec': scg_rec}
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                print(savePath)
                savemat(savePath + "/" + str(i) + ".mat", mkdic)
            evaluate(savePath, evalPath, test)
    # merge_excel(excel_path)


if __name__ == "__main__":

    user_list = ['wxw', 'chy', 'lc', 'hh', 'liushilan', 'yq', 'gxc', 'sy2', 'jx', 'zyj']
    blank_list = ['ws']
    patient_list = ['20-m_train', '21-m_train', '22-f_train', '23-m_train',
                    '27-f_train', '34-f_train', '35-m-2_train',
                    '52-f_train', '54-m_train', '58-m_train',
                    'AMI_train']
    test_list = ["wxw"]
    for user in blank_list:
        PATH1 = 'eval/wxw_train_2.pth'
        PATH2 = 'evaluate/ecg_deeplab_dkh9.pth'

        # model1 = Autoencoder(1, 128)
        model1 = get_network(args)
        # model1 = DeepLab6()
        model2 = DeepLab6()
        model1.load_state_dict(torch.load(PATH1, map_location=lambda storage, loc: storage))
        model2.load_state_dict(torch.load(PATH2, map_location=lambda storage, loc: storage))
        model1.eval()
        model2.eval()
        tst(model1, model2, user)
