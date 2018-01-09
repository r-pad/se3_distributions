import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

#class LRN(nn.Module):
#    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
#        super(LRN, self).__init__()
#        self.ACROSS_CHANNELS = ACROSS_CHANNELS
#        if ACROSS_CHANNELS:
#            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
#                                      stride=1,
#                                      padding=(int((local_size-1.0)/2), 0, 0))
#        else:
#            self.average=nn.AvgPool2d(kernel_size=local_size,
#                                      stride=1,
#                                      padding=int((local_size-1.0)/2))
#        self.alpha = alpha
#        self.beta = beta
#
#
#    def forward(self, x):
#        if self.ACROSS_CHANNELS:
#            div = x.pow(2).unsqueeze(1)
#            div = self.average(div).squeeze(1)
#            div = div.mul(self.alpha).add(1.0).pow(self.beta)
#        else:
#            div = x.pow(2)
#            div = self.average(div)
#            div = div.mul(self.alpha).add(1.0).pow(self.beta)
#        x = x.div(div)
#        return x


class GenPoseNet(nn.Module):

    def __init__(self, classification_output_dims=(360,360,360)):
        super(GenPoseNet, self).__init__()
        self.features_classification = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.features_regression = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.compare_classification = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.dim0_linear = nn.Linear(4096, classification_output_dims[0])
        self.dim1_linear = nn.Linear(4096, classification_output_dims[1])
        self.dim2_linear = nn.Linear(4096, classification_output_dims[2])
        
        self.compare_regression = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4),
        )

    def forward_regression(self, origin, query):
        origin = self.features_regression(origin)
        query = self.features_regression(query)        
        origin = origin.view(origin.size(0), 256 * 6 * 6)
        query = query.view(query.size(0), 256 * 6 * 6)        
        x = self.compare_regression(torch.cat((origin, query), dim=1))
        return x
        
    def forward_classification(self, origin, query):
        origin = self.features_classification(origin)
        query = self.features_classification(query)
        origin = origin.view(origin.size(0), 256 * 6 * 6)
        query = query.view(query.size(0), 256 * 6 * 6)        
        
        x = self.compare_classification(torch.cat((origin, query), dim=1))

        dim0 = self.dim0_linear(x)
        dim1 = self.dim1_linear(x)
        dim2 = self.dim2_linear(x)
        return dim0, dim1, dim2

    def forward(self, origin, query):
        quat = self.forward_regression(origin, query)
        dim0, dim1, dim2 = self.forward_classification(origin, query)
        return quat, dim0, dim1, dim2

def gen_pose_net(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = GenPoseNet(**kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['alexnet'])
        model_dict = model.state_dict()
        update_dict = {}
        for k, v in pretrained_dict.items():
            k_classification = str(k).replace('features', 'features_classification')
            if k_classification in model_dict:
                update_dict[k_classification] = v
                
            k_regression = str(k).replace('features', 'features_regression')
            if k_regression in model_dict:
                update_dict[k_regression] = v
            
        model_dict.update(update_dict) 
        model.load_state_dict(model_dict)
        
    return model