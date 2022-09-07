import torch
import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, num_classes, input_channel=3, init_weight=True):
        super().__init__()
        self.features = nn.Sequential(
            # [B, 3, 224, 224] ==> [B, 48, 55, 55] ==> [B, 48, 27, 27]
            nn.Conv2d(in_channels=input_channel, out_channels=48,
                      kernel_size=11, stride=4, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # [B, 48, 27, 27] ==> [B, 128, 27, 27] ==> [B, 128, 13, 13]
            nn.Conv2d(in_channel=48, out_channels=128,
                      kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kenel_size=3, stride=2),
            # [B, 128, 13, 13] ==> [B, 192, 13, 13]
            nn.Conv2d(in_channels=128, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            # [B, 192, 13, 13] ==> [B, 192, 13, 13]
            nn.Conv2d(in_channels=192, out_channels=192,
                      kernel_size=3, stride=1, padding=1),
            # [B, 192, 13, 13] ==> [B, 128, 13, 13] ==> [B, 128, 6, 6]
            nn.Conv2d(in_channels=192, out_channels=128,
                      kernel_size=3, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifer = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(True),

            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),

            nn.Linear(2048, num_classes)
        )

    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_
