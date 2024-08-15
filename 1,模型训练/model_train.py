import numpy as np
import os
import re
from sklearn.model_selection import train_test_split
from model_net import MyConvNet
import torch
import torch.nn as nn
import torch.utils.data as Data


DEF_FILE_FORMAT = '.txt'
DEF_FILE_NAME_SEPERATOR = '_'
DEF_N_ROWS = 150
DEF_FILE_MAX = 100


motion_names = ['RightAngle', 'SharpAngle', 'Lightning', 'Triangle', 'Letter_h', 'letter_R', 'letter_W', 'letter_phi', 'Circle', 'UpAndDown', 'Horn', 'Wave', 'NoMotion']
motion_to_label = {name: idx for idx, name in enumerate(motion_names)}

# 1,导入数据集，数据集预处理
# 数据路径
dataset_file_path = r"F:\python\python program\electronic_wand\CyberryPotter_ElectromagicWand_Basic_Project-main\CNN\TraningData_8_2"


def load_dataset(root_dir, max_rows=None):
    file_list = []
    labels = []
    for filename in os.listdir(root_dir):
        if filename.endswith(DEF_FILE_FORMAT):
            match = re.match(rf'^([\w]+)_([\d]+){DEF_FILE_FORMAT}$', filename)
            if match:
                motion_name = match.group(1)
                number_str = match.group(2)
                number = int(number_str)
                if 0 <= number <= DEF_FILE_MAX:
                    if motion_name in motion_to_label:
                        file_path = os.path.join(root_dir, filename)
                        # 使用max_rows参数限制读取的行数
                        data = np.loadtxt(file_path, delimiter=' ', usecols=(0, 1, 2), max_rows=max_rows)
                        file_list.append(data)
                        labels.append(motion_to_label[motion_name])
                    else:
                        print(f"Motion name not recognized: {filename}")
                else:
                    print(f"Number out of range: {filename}")
            else:
                print(f"Invalid file name format: {filename}")
    return file_list, labels


# 读取数据，转成tensor
data_list, labels = load_dataset(dataset_file_path, max_rows=DEF_N_ROWS)
data_tensor = torch.tensor(data_list, dtype=torch.float32).permute(0, 2, 1)
labels = torch.tensor(labels, dtype=torch.long)


X_train, X_test, Y_train, Y_test = train_test_split(data_tensor, labels, test_size=0.2, random_state=123)
train_data = Data.TensorDataset(X_train, Y_train)
test_data = Data.TensorDataset(X_test, Y_test)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=32,
    shuffle=True,
    # num_workers=4
)

val_loader = Data.DataLoader(
    dataset=test_data,
    batch_size=32,
    shuffle=True,
    # num_workers=4
)


# 3，模型训练
modelnet = MyConvNet()

optimizer = torch.optim.SGD(modelnet.parameters(), lr=0.003)
loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数，

train_loss = []
train_c = []

val_loss = []
val_c = []

for epoch in range(2000):
    train_loss_epoch = 0
    val_loss_epoch = 0
    train_corrects = 0
    val_correct = 0
    modelnet.train()
    for step, (b_x, b_y) in enumerate(train_loader):
        float_tensor = b_x.to(torch.float32)
        output = modelnet(b_x)
        loss = loss_func(output, b_y)
        pre_lab = torch.argmax(output, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_epoch += loss.item() * b_x.size(0)
        train_corrects += torch.sum(pre_lab == b_y)

    print('***' * 10, '已完成', epoch + 1, '轮')

    train_loss.append(train_loss_epoch / len(train_data))
    train_c.append(train_corrects.double() / len(train_data))
    print('train_loss:', train_loss[-1])
    print('train_accurary:', train_c[-1])

    modelnet.eval()

    for step, (val_x, val_y) in enumerate(val_loader):
        output = modelnet(val_x)
        loss = loss_func(output, val_y)
        pre_lab = torch.argmax(output, 1)
        val_loss_epoch += loss.item() * val_x.size(0)
        val_correct += torch.sum(pre_lab == val_y)

    val_loss.append(val_loss_epoch / len(test_data))
    val_c.append(val_correct / len(test_data))
    print("val_loss:", val_loss[-1])
    print("val_accurary:", val_c[-1].item())


# 4，保存模型
torch.save(modelnet, 'modelnet.pth')

