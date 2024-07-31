import torch
import torch.nn as nn
import torch.nn.functional as func
from collections import OrderedDict


class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)
    
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = func.relu(x)

        x = self.fc3(x)
        return x
class DigitModelDis(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModelDis, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, 5, 1, 2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv2', nn.Conv2d(64, 64, 5, 1, 2)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=2, stride=2)),

                ('conv3', nn.Conv2d(64, 128, 5, 1, 2)),
                ('bn3', nn.BatchNorm2d(128)),
                ('relu3', nn.ReLU(inplace=True))
            ])
        )
        self.classifier_teacher_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(6272, 2048)),
                ('bn4', nn.BatchNorm1d(2048)),
                ('relu4', nn.ReLU(inplace=True)),

            ])
        )
        self.classifier_teacher_out = nn.Sequential(
            OrderedDict([
                ('fc2', nn.Linear(2048, 512)),
                ('bn5', nn.BatchNorm1d(512)),
                ('relu5', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(512, num_classes)),
            ])
        )



        
        self.classifier_student_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(6272, 2048)),
                ('bn4', nn.BatchNorm1d(2048)),
                ('relu4', nn.ReLU(inplace=True)),
            ])
        )
        

        self.classifier_student_out = nn.Sequential(
            OrderedDict([
                ('fc2', nn.Linear(2048, 512)),
                ('bn5', nn.BatchNorm1d(512)),
                ('relu5', nn.ReLU(inplace=True)),

                ('fc3', nn.Linear(512, num_classes)),
            ])
        )



    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x_t_h = self.classifier_teacher_h(x)
        x_t = self.classifier_teacher_out(x_t_h)
        x_s_h = self.classifier_student_h(x)
        x_s = self.classifier_student_out(x_s_h)
        return x_t,x_s, x_t_h, x_s_h


# class AlexNet(nn.Module):
#     """
#     used for DomainNet and Office-Caltech10
#     """
#     def __init__(self, num_classes=10):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             OrderedDict([
#                 ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
#                 ('bn1', nn.BatchNorm2d(64)),
#                 ('relu1', nn.ReLU(inplace=True)),
#                 ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),
#
#                 ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
#                 ('bn2', nn.BatchNorm2d(192)),
#                 ('relu2', nn.ReLU(inplace=True)),
#                 ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),
#
#                 ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
#                 ('bn3', nn.BatchNorm2d(384)),
#                 ('relu3', nn.ReLU(inplace=True)),
#
#                 ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
#                 ('bn4', nn.BatchNorm2d(256)),
#                 ('relu4', nn.ReLU(inplace=True)),
#
#                 ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
#                 ('bn5', nn.BatchNorm2d(256)),
#                 ('relu5', nn.ReLU(inplace=True)),
#                 ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
#             ])
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#
#         self.classifier = nn.Sequential(
#             OrderedDict([
#                 ('fc1', nn.Linear(256 * 6 * 6, 4096)),
#                 ('bn6', nn.BatchNorm1d(4096)),
#                 ('relu6', nn.ReLU(inplace=True)),
#                 ('fc2', nn.Linear(4096, 4096)),
#                 ('bn7', nn.BatchNorm1d(4096)),
#                 ('relu7', nn.ReLU(inplace=True)),
#                 ('fc3', nn.Linear(4096, num_classes)),
#             ])
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


class AlexNetDis(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    print("========= fc2 =========")
    def __init__(self, num_classes=10):
        super(AlexNetDis, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier_global_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(4096, 4096)),
            ])
        )
        self.classifier_global_out = nn.Sequential(
            OrderedDict([
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes))
            ])
        )

        self.classifier_local_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(4096, 4096)),
            ])
        )
        self.classifier_local_out = nn.Sequential(
            OrderedDict([
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_t_h = self.classifier_global_h(x)
        x_t = self.classifier_global_out(x_t_h)

        x_s_h = self.classifier_local_h(x)
        x_s = self.classifier_local_out(x_s_h)
        return x_t,x_s,x_t_h,x_s_h


class AlexNetDis_GL(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNetDis_GL, self).__init__()
        print("========= AlexNetDis_GL fc1 =========")
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier_global_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),

            ])
        )
        self.classifier_global_out = nn.Sequential(
            OrderedDict([
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes))
            ])
        )

        self.classifier_local_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
            ])
        )
        self.classifier_local_out = nn.Sequential(
            OrderedDict([
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )



    def forward_global(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_t_h = self.classifier_global_h(x)
        x_t = self.classifier_global_out(x_t_h)

        return x_t,x_t_h


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_t_h = self.classifier_global_h(x)
        x_t = self.classifier_global_out(x_t_h)

        x_s_h = self.classifier_local_h(x)
        x_s = self.classifier_local_out(x_s_h)
        return x_t,x_s,x_t_h,x_s_h
class AlexNetDis_GLAll(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNetDis_GLAll, self).__init__()
        print("========= AlexNetDis_GLAll fc1 =========")
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier_global_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),

            ])
        )
        self.classifier_global_out = nn.Sequential(
            OrderedDict([
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes))
            ])
        )

        self.classifier_local_h = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(256 * 6 * 6, 4096)),
            ])
        )
        self.classifier_local_out = nn.Sequential(
            OrderedDict([
                ('bn6', nn.BatchNorm1d(4096)),
                ('relu6', nn.ReLU(inplace=True)),
                ('fc2', nn.Linear(4096, 4096)),
                ('bn7', nn.BatchNorm1d(4096)),
                ('relu7', nn.ReLU(inplace=True)),
                ('fc3', nn.Linear(4096, num_classes)),
            ])
        )



    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x_t_h = self.classifier_global_h(x)
        x_t = self.classifier_global_out(x_t_h)

        x_s_h = self.classifier_local_h(x)
        x_s = self.classifier_local_out(x_s_h)
        return x_t,x_s,x_t_h,x_s_h



