import torch.nn as nn
import torch.nn.functional as F

class MyCNN5(nn.Module):

      def __init__(self, numclasses=15):
          super(MyCNN5, self).__init__()
          self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
          self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
          self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
          self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8)
          self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=8, dilation=8)
          self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
          self.conv7 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
          self.conv8 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
          self.last = nn.Conv2d(64, numclasses, kernel_size=1)

      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = F.relu(self.conv4(x))
          x = F.relu(self.conv5(x))
          x = F.relu(self.conv6(x))
          x = F.relu(self.conv7(x))
          x = F.relu(self.conv8(x))
          x = self.last(x)

          return(x)

class PredictPos7(nn.Module):

      def __init__(self, numclasses=15, inputdim=30):
          super(PredictPos7, self).__init__()
          self.depthconv1 = nn.Conv2d(inputdim, inputdim, kernel_size=3, padding=1, groups=inputdim)
          self.pointwise = nn.Conv2d(inputdim, 32, kernel_size=1)
          self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
          self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4)
          self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=8, dilation=8)
          self.conv8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
          self.linear = nn.Conv2d(32, numclasses, kernel_size=1)
          self.adapt = nn.AdaptiveMaxPool2d((1,1))
          #self.lin = nn.Linear(

      def forward(self, x):
          x = self.depthconv1(x)
          x = F.relu(self.pointwise(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = F.relu(self.conv4(x))
          x = F.relu(self.conv8(x))
          x = self.adapt(x)
          x = self.linear(x)


          return(x)

class PredictBond3(nn.Module):

      def __init__(self, numclasses=8, inputdim=31):
          super(PredictBond3, self).__init__()
         # self.conv1 = nn.Conv2d(24, 32, kernel_size=3, padding=1)
          self.depthconv1 = nn.Conv2d(inputdim, inputdim, kernel_size=3, padding=1, groups=inputdim)
          self.pointwise = nn.Conv2d(inputdim, 32, kernel_size=1)
          self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
          self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=4, dilation=4)
          self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=8, dilation=8)
          self.conv8 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
          self.linear = nn.Conv2d(32, numclasses, kernel_size=1)
          self.adapt = nn.AdaptiveMaxPool2d((1,1))
          #self.lin = nn.Linear(

      def forward(self, x):
          x = self.depthconv1(x)
          x = F.relu(self.pointwise(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = F.relu(self.conv4(x))
          x = F.relu(self.conv8(x))
          x = self.adapt(x)
          x = self.linear(x)


          return(x)

class depth_cnn(nn.Module):
  #check defaults
     def __init__(self, nin, nout, kernel_size, padding=0, dilation=1):
         super().__init__()
         self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=nin)
         self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

     def forward(self, x):
         x = self.depthwise(x)
         x = self.pointwise(x)
         return x

class my_depth_cnn(nn.Module):

      def __init__(self, numclasses=10):
          super().__init__()
          self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
          self.conv2 = depth_cnn(32, 32, kernel_size=3, padding=2, dilation=2)
          self.conv3 = depth_cnn(32, 32, kernel_size=3, padding=4, dilation=4)
          self.conv4 = depth_cnn(32, 32, kernel_size=3, padding=8, dilation=8)
          self.last = depth_cnn(32, numclasses, kernel_size=1)

      def forward(self, x):
          x = F.relu(self.conv1(x))
          x = F.relu(self.conv2(x))
          x = F.relu(self.conv3(x))
          x = F.relu(self.conv4(x))
          x = self.last(x)

          return(x)

