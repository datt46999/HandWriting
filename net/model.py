import torch
import torch.nn.functional as F

from torch import nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding =1, bias = False )
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride =1, padding =1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shorcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion *planes:
            self.shorcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size =1, stride  = stride, bias = False),
                nn.BatchNorm2d(self.expansion *planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shorcut(x)
        out = F.relu(out)
        return out 
    
class CNN(nn.Module):
    def __init__(self, cnn_confg, flattening = "maxpool"):
        super().__init__()
        # self.cnn_config = cnn_confg
        self.flattening  = flattening

        self.k = 1

        self.features = nn.ModuleList([nn.Conv2d(1, 32, 7, [2, 2], 3), nn.ReLU()])
        in_channels = 32
        cntm = 0
        cnt = 1
        for m in cnn_confg:
            if m == 'M':
                self.features.add_module('mxp' + str(cntm), nn.MaxPool2d(kernel_size=2, stride=2))
                cntm += 1
            else:
                for i in range(m[0]):
                    x = m[1]
                    self.features.add_module('cnv' + str(cnt), BasicBlock(in_channels, x,))
                    in_channels = x
                    cnt += 1
    def forward(self, x, reduce = "max"):
        y = x

        for i, nn_module in enumerate(self.features):
            y=  nn_module(y)
        if self.flattening == "maxpool":
            # N, C, H, W
            y = F.max_pool2d(y, [y.size(2), self.k], stride = [y.size(2), 1], padding = [0, self.k//2])
            
        return y 

# connectionisr temporal classification
class CTC(nn.Module):
    def __init__(self, input_size, rnn_cfg, nclasses):
        super().__init__()
        hidden, num_layers = rnn_cfg
        self.rec = nn.LSTM(input_size,  hidden, num_layers = num_layers, bidirectional = True, dropout = .2 )
        self.fnl = nn.Sequential(nn.Dropout(.5), nn.Linear(2 * hidden, nclasses))

        self.cnn = nn.Conv2d(input_size, nclasses, kernel_size =(1, 3), stride = 1, padding = (0,1))
    def forward(self, x):
        y = x.permute(2,3,0,1)[0]
        y = self.rec(y)[0]
        y = self.fnl(y)

        return y, self.cnn(x).permute(2,3, 0, 1)[0]
    

class CNN_RNN(nn.Module):
    def __init__(self, cnn_cfg, hidden_cfg, nclasses, head ='cnn', flattening ='maxpool'):
        super().__init__()
        self.cnn = CNN(cnn_cfg, flattening= flattening)
        if flattening == "maxpool":
            hidden = cnn_cfg[-1][-1]
        else:
            print("problem in: ...")
        
        self.top = CTC(hidden, hidden_cfg, nclasses)
    def forward(self, x):
        y = self.cnn(x)
        y = self.top(y)
        return y
    
def set_bn_eval(m):
    if isinstance(m, nn.modules.batchnorm._BatchNorm):
        m.eval()

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)