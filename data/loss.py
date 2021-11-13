import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """
    Constructs a content loss function based on the VGG19 network.
    """

    def __init__(self) -> None:
        super(ContentLoss, self).__init__()
        imagenet_normalize = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }

        vgg19 = models.vgg19(pretrained=True, num_classes=1000).eval()
        # Extract the output of the thirty-fifth layer in the VGG19 model as the content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        self.T_normalize = T.Compose([T.Normalize(
            mean = imagenet_normalize["mean"],
            std = imagenet_normalize["std"]
        )])


    def forward(self, sr, hr):
        # Normalize both images
        sr = self.T_normalize(sr)
        hr = self.T_normalize(hr)
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss