import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import timm

AVAILABLE_MODEL = ['vgg16', 'resnet50', 'mobilenet_v2', 'vit']

class ModelFactory(nn.Module):

    """A tool that construct pytorch model
    Parameters
    ----------
    model_name : str
        name of model. Defaults to 'resnet50'.
    pretrained : bool
        Whether to pretrain when loading the model. Defaults to 'True'.
    class_num : int
        number of classes. Defaults to 10.
    """

    def __init__(self, model_name = 'resnet50', pretrained = True, class_num = 10):
        super(ModelFactory, self).__init__()

        if model_name in AVAILABLE_MODEL:
            self.model_name = model_name
        else:    
            raise NotImplementedError('{} has not been implemented, use model in {}'.format(self.model_name,AVAILABLE_MODEL))

        # vgg16: optimal input shape (224 * 224)
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            
            # Change the output layer to output classes instead of 1000 classes
            num_ftrs = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_ftrs, class_num)

        # Resnet50: optimal input shape (224 * 224)
        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)

            # Change the output layer to output classes instead of 1000 classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, class_num)

        # mobilenet_v2: optimal input shape (224 * 224)
        elif model_name == 'mobilenet_v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)

            # Change the output layer to output classes instead of 1000 classes
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(num_ftrs, class_num)

        # Vistion Transformer: optimal input shape (224 * 224)
        elif model_name == 'vit':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=class_num)

        ## efficientNet etc.. need updated torch + torchvision versions.

    def forward(self, x):
        return self.model(x)