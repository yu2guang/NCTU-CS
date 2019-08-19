import torch as t
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt

# plot
def plot_line(n, y, type_color):
    x = [i for i in range(1,n+1)]

    if (type_color == 'relu_train'):
        c = 'orange'
    elif (type_color == 'relu_test'):
        c = 'blue'
    elif (type_color == 'leaky_relu_train'):
        c = 'green'
    elif (type_color == 'leaky_relu_test'):
        c = 'red'
    elif (type_color == 'elu_train'):
        c = 'purple'
    elif (type_color == 'elu_test'):
        c = 'brown'

    plt.plot(x, y, color=c, linewidth=1.0, label=type_color)


def show_result_start(net_name):
    print('~~~ start:',net_name,'~~~')
    plt.figure()
    plt.title('Activation function comparision({name})'.format(name=net_name), fontsize=18)
    plt.xlim((-10, 510))
    plt.xticks(np.arange(0, 510, 50))
    plt.ylim((62, 103))
    plt.yticks(np.arange(65, 103, 5))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')

def show_result_end():
    print('~~~end~~~')
    plt.legend(loc='lower right')
    plt.show()

def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label

class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()  #???

        self.actFunct = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['leaky_relu', nn.LeakyReLU()],
            ['elu', nn.ELU(alpha=1.0)]
        ])

        self.firstConv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1,51),
                stride=(1,1),
                padding=(0,25),
                bias=False
            ),
            nn.BatchNorm2d( #???
                16,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            )
        )

        self.depthwiseConv1 = nn.Sequential(
            nn.Conv2d(
                16, 32,
                kernel_size=(2,1),
                stride=(1,1),
                groups=16,
                bias=False
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
        )

        self.depthwiseConv2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1,4),
                stride=(1,4),
                padding=0
            ),
            nn.Dropout( #???
                p=0.25
            )
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(
                32, 32,
                kernel_size=(1, 15),
                stride=(1, 1),
                padding=(0, 7),
                bias=False
            ),
            nn.BatchNorm2d(
                32,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True
            ),
        )

        self.separableConv2 = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(1, 8),
                stride=(1, 8),
                padding=0
            ),
            nn.Dropout(
                p=0.25
            )
        )

        self.classify = nn.Sequential(
            nn.Linear(
                in_features=736,
                out_features=2,
                bias=True
            )
        )

    def forward(self, x, act):
        x = self.firstConv(x)
        x = self.depthwiseConv1(x)
        x = self.actFunct[act](x)
        x = self.depthwiseConv2(x)
        x = self.separableConv1(x)
        x = self.actFunct[act](x)
        x = self.separableConv2(x)

        x = x.view(x.size(0), -1)
        output = self.classify(x.float())
        return output

C = 2
T = 750
N = 2

class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()

        self.actFunct = nn.ModuleDict([
            ['relu', nn.ReLU()],
            ['leaky_relu', nn.LeakyReLU()],
            ['elu', nn.ELU(alpha=1.0)]
        ])

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 25, (1,5)),
            nn.Conv2d(25, 25, (C,1)),
            nn.BatchNorm2d(25, eps=1e-5, momentum=0.1),
        )

        self.conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(25, 50, (1,5)),
            nn.BatchNorm2d(50, eps=1e-5, momentum=0.1),
        )

        self.conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(50, 100, (1, 5)),
            nn.BatchNorm2d(100, eps=1e-5, momentum=0.1),
        )

        self.conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
            nn.Conv2d(100, 200, (1, 5)),
            nn.BatchNorm2d(200, eps=1e-5, momentum=0.1),
        )

        self.conv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5),
        )

        self.classify = nn.Sequential(
            nn.Linear(
                in_features=8600,
                out_features=2,
                bias=True
            )
        )

    def forward(self, x, act):
        x = self.conv1(x)
        x = self.actFunct[act](x)
        x = self.conv2(x)
        x = self.actFunct[act](x)
        x = self.conv3(x)
        x = self.actFunct[act](x)
        x = self.conv4(x)
        x = self.actFunct[act](x)
        x = self.conv5(x)
        # x.size(0): batch size
        x = x.view(x.size(0), -1)
        output = self.classify(x.float())
        return output

def lr_decay(optim):
    for param_group in optim.param_groups:
        param_group['lr'] = param_group['lr']*0.98
        print(param_group['lr'])


# Hyper Parameters
#BATCH_SIZE = 1080
BATCH_SIZE = 64
LR = 1e-2
#EPOCH = 500
EPOCH = 300
loss_funct = nn.CrossEntropyLoss()

train_data, train_label, test_data, test_label = read_bci_data()

# data loader
train_x = t.from_numpy(train_data)
train_y = t.from_numpy(train_label)
torch_dataset = Data.TensorDataset(train_x, train_y)
train_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
)

nets = ['EEGNet', 'DeepConvNet']
activation_funct = ['relu', 'leaky_relu', 'elu']
highest_accur = [[0,0], [0,0], [0,0], [0,0], [0,0], [0,0]]
j = 0

for net_i in nets:
    show_result_start(net_i)
    for act_f in activation_funct:

        if(net_i=='EEGNet'):
            net = EEGNet().cuda()
        else:
            net = DeepConvNet().cuda()

        optimizer = t.optim.Adam(
            net.parameters(),
            lr=LR
        )

        print('---', act_f, '---')

        # train
        train_accuracy = []
        test_accuracy = []
        for i in range(EPOCH):
            y_origin = []
            train_label_shuffle = []
            for step, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.cuda()#+t.randn_like(batch_x).cuda()/10
                batch_y = batch_y.cuda()
                # add shuffled label
                train_label_shuffle.extend(batch_y.long())

                # clear gradient
                optimizer.zero_grad()

                # forward + backward
                output = net.forward(batch_x.float(),act_f)
                y_o = t.max(output, 1)[1]
                y_origin.extend(y_o)

                loss = loss_funct(output, batch_y.long())
                loss.backward()

                # update parameters
                optimizer.step()

            y_hat = (np.array(train_label_shuffle)==y_origin).astype(np.int)
            accuracy = sum(y_hat)/np.size(y_origin)
            train_accuracy.append(accuracy*100)
            print(act_f, ': epoch ', i+1, ', loss:', loss.item(), ', accuracy:', accuracy)

            # test
            net.eval() # cancel mask
            test_out = net.forward(t.from_numpy(test_data).cuda().float(), act_f)
            test_out = t.max(test_out, 1)[1]
            test_hat = (test_out == t.from_numpy(test_label).cuda().long()).int()
            accuracy = sum(test_hat).item() / np.size(test_label)
            if(accuracy>highest_accur[j][1]):
                highest_accur[j][0] = i+1
                highest_accur[j][1] = accuracy
                hi_name = "{netn}_{actn}".format(netn=net_i, actn=act_f)
                t.save(net.state_dict(), "{netn}_{actn}_{e}.pkl".format(netn=net_i, actn=act_f, e=(i+1)))
            net.train()
            test_accuracy.append(accuracy*100)
            print('Test accuracy:', accuracy)

            # decay lr
            if((i+1)%50==0):
                lr_decay(optimizer)

        plot_line(EPOCH, train_accuracy, "{name}_train".format(name=act_f))
        plot_line(EPOCH, test_accuracy, "{name}_test".format(name=act_f))

        print("Highest accuracy:", highest_accur[j][1], ', epoch', highest_accur[j][0])
        j+=1

    show_result_end()





