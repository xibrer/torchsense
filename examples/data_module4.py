from torchsense.datasets.folder import ImageFolder
from torch.utils.data import DataLoader

list1 = ["acc", "mix_mic", "mic"]  #, "sisnr", "speakernum", "text"]
list2 = ["PY_orbit2_RTN"]
data = ImageFolder(root="data", params=list1)
train_set, test_set = data.train_test_split(0.5)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

for i, batch in enumerate(train_loader):
    acc, mix, mic = batch[0]
    # acc = batch[0]
    print(f"Index: {i}, Acc: {acc},\nshape:{acc.shape}")

    break
