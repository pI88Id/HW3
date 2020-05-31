import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision.models import alexnet
from torch.utils.model_zoo import load_url as load_state_dict_from_url

''' 
Very easy template to start for developing your AlexNet with DANN 
Has not been tested, might contain incompatibilities with most recent versions of PyTorch (you should address this)
However, the logic is consistent
'''

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class RandomNetworkWithReverseGrad(nn.Module):
    def __init__(self, **kwargs):
        super(RandomNetworkWithReverseGrad, self).__init__()
        self.features = alexnet().features
        self.classifier = alexnet().classifier
        self.dann_classifier = alexnet().classifier

    def forward(self, x, alpha=None):
        x = self.features(x)
        x = alexnet().avgpool(x)
        x = torch.flatten(x, 1)

        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(x, alpha)
            discriminator_output = self.dann_classifier(reverse_feature)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # do something else
            class_outputs = self.dann_classifier(x)
            return class_outputs

def randomNetworkWithReverseGrad(pretrained=True, **kwargs):
    model = RandomNetworkWithReverseGrad(**kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'], progress=True)
        model.load_state_dict(state_dict, strict=False)

        model.dann_classifier[1].weight.data = model.classifier[1].weight.data
        model.dann_classifier[1].bias.data = model.classifier[1].bias.data

        model.dann_classifier[4].weight.data = model.classifier[4].weight.data
        model.dann_classifier[4].bias.data = model.classifier[4].bias.data

        model.dann_classifier[6].weight.data = model.classifier[6].weight.data
        model.dann_classifier[6].bias.data = model.classifier[6].bias.data

    return model
