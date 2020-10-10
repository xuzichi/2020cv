import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math


class alexnet(nn.Module):
    def __init__(self, init_weights=True):
        super(alexnet, self).__init__()

        def conv_pool(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=(3, 3), stride=stride, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            )

        self.base_net = nn.Sequential(
            conv_pool(3, 128, 1),
            conv_pool(128, 256, 1),
            conv_pool(256, 384, 1),
            conv_pool(384, 384, 1),
            conv_pool(384, 256, 1),
        )

        self.conv_1 = nn.Sequential(  # 输入cifar时为32 * 32
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=2, padding=2),  # 输出15 * 15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # 输出7 * 7
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5), stride=1, padding=2),  # 输出7 * 7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0)  # 输出3 * 3
        )

        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv_4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.conv_5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        # x = nn.functional.avg_pool2d(x, 1)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


AlexNet = models.alexnet(pretrained=True)
print(AlexNet)
model = alexnet()
print(model)

image_transforms = {
    'train': transforms.Compose([  # 进行数据增强（把以下步骤整理成一个步骤）（下同）
        # transforms.Resize(size=32),
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        # transforms.CenterCrop(size=224),  # 中心裁剪到224*224
        transforms.ColorJitter(brightness=(0, 36), contrast=(0, 10), saturation=(0, 25), hue=(-0.5, 0.5)),
        transforms.RandomGrayscale(p=0.6),
        transforms.ToTensor(),  # 转成Tensor
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # 正规化
    ]),
    'valid': transforms.Compose([
        # transforms.Resize(size=32),
        # transforms.CenterCrop(size=224),
        transforms.ColorJitter(brightness=(0, 36), contrast=(0, 10), saturation=(0, 25), hue=(-0.5, 0.5)),
        transforms.RandomGrayscale(p=0.6),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

epochs = 501
batch_size = 600
train_error = []
test_acc = []

dataset_train = datasets.CIFAR10(root='./dataset_method_3', train=True, transform=image_transforms['train'],
                                 download=True)
dataset_test = datasets.CIFAR10(root='./dataset_method_4', train=False, transform=image_transforms['valid'],
                                download=True)

train_data = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)  # 将数据封装成dataset类
valid_data = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

for param in model.parameters():
    param.requires_grad = True

loss_func = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99, dampening=0, weight_decay=5e-4, nesterov=False)
optimizer = optim.Adam(model.parameters())

for epoch in range(epochs):
    loss_ep = 0

    for i, (inputs, labels) in enumerate(train_data):
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_func(output, labels)
        loss.backward()  # 反向传播回传损失
        optimizer.step()  # 根据反向传播得到的梯度更新参数
        loss_ep += loss
    loss_ep = loss_ep / 50000
    train_error.append(loss_ep)

    with torch.no_grad():
        counter = 0
        for j, (inputs, labels) in enumerate(valid_data):
            output = model(inputs)
            ret, predictions = torch.max(output.data, 1)

            for i in range(len(predictions)):
                if predictions[i] == labels[i]:
                    counter += 1
        test_acc.append(counter / 10000)
    if epoch % 1 == 0:
        print('epoch:', epoch, 'loss:', loss_ep, 'test_acc:', counter / 10000, '\n')

plt.title(" training error curves")
plt.xlabel("epochs")
plt.ylabel("loss_average")
x = np.arange(0, epochs)
plt.plot(x, train_error)
plt.show()

plt.title(" test acc curves")
plt.xlabel("epochs")
plt.ylabel("loss_average")
x = np.arange(0, epochs)
plt.plot(x, test_acc)
plt.show()
