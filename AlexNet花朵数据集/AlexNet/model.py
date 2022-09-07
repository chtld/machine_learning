import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_class, init_weights=False):
        super().__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3,48,kernel_size=11, stride=4, padding=2), #in[3,224,224] ==> out[48,55,55]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2),                # out[48,27,27]

            nn.Conv2d(48,128,kernel_size=5,padding=2),           # out[128,27,27]   (K=5,s=1,p=2 conv size no change)
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2),                # out [128,13,13]

            nn.Conv2d(128, 192, kernel_size=3, padding=1),       # out [192,13,13]  (K=3,s=1,p=1 conv size no change)
            nn.ReLU(True),

            nn.Conv2d(192,192, kernel_size=3,padding=1),         # out [192,13,13]
            nn.ReLU(True),

            nn.Conv2d(192, 128, kernel_size=3, padding=1),       # out [128,13,13]
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2)                # out [128,6,6]
        )

        self.classifer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128*6*6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Linear(2048, num_class)
        )

        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) 
        x = self.classifer(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)