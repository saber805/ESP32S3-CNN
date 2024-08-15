import torch
import torch.nn as nn

# 关闭科学计数法
torch.set_printoptions(sci_mode=False)

# 2，模型构建
class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(3, out_channels=30, kernel_size=3, stride=3),  # 1*150
            nn.ReLU(),
            nn.Conv1d(30, out_channels=15, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(3, 3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(75, 13),
            nn.Dropout(p=0.3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


model = torch.load('modelnet.pth')
model.eval()
print(model)

# 把模型所有权重转为整形,存到txt里，RELU，池化和sigmoid没有参数，直接用C写个函数就行
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.data}')
        # 把每个参数都弄成整形,先保留一位小数试试
        # param.data = torch.round(param.data * 10) / 10
        str_weight = str(param.data)
        # 去掉tensor字符串和（）
        str_weight = str_weight[7:-1]
        # 替换[]为{}
        str_weight = str_weight.replace('[', '{')
        str_weight = str_weight.replace(']', '}')

        # name中的.替换为_
        name = name.replace('.', '_')

        # 要判断len(param.data.shape)
        if len(param.data.shape) == 3:
            Variable_Name = name + '[' + str(param.data.shape[0]) + ']' + '[' + str(param.data.shape[1]) + ']' + '[' + str(param.data.shape[2]) +']'
        elif len(param.data.shape) == 2:
            Variable_Name = name + '[' + str(param.data.shape[0]) + ']' + '[' + str(param.data.shape[1]) + ']'
        elif len(param.data.shape) == 1:
            Variable_Name = name + '[' + str(param.data.shape[0]) + ']'

        with open('C_h/' + name+'.h', 'w') as f:
            f.write('#ifndef ' + name + '\n')
            if name == 'conv_2_weight':
                # 这里特殊情况特殊处理，有的层权重太多，直接输出有省略号
                pass
            else:
                f.write('float ' + Variable_Name + ' = ' + str_weight + ';')  # 运行慢的话可以改为int，
            f.write('\n#endif ')
            f.close()






