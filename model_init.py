import torch
from loaddataset import get_labelspace_size
from torchvision.models.resnet import resnet18, resnet34
from efficientnet_pytorch import EfficientNet


def init_model(m_name, p_trained, epochs, lr, mom, dev):

    if m_name == 'resnet-18':
        if p_trained:
            print('Beginning with pretrained resnet-18 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
                  .format(epochs, lr, mom))
            model = resnet18(pretrained=True)
        else:
            print('Beginning with untrained resnet-18 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
                  .format(epochs, lr, mom))
            model = resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, get_labelspace_size())
        model = torch.nn.DataParallel(model)
        model.to(dev)
    elif m_name == 'resnet-34':
        if p_trained:
            print('Beginning with pretrained resnet-34 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
                  .format(epochs, lr, mom))
            model = resnet34(pretrained=True)
        else:
            print('Beginning with untrained resnet-34 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
                  .format(epochs, lr, mom))
            model = resnet34(pretrained=False)
        model.fc = torch.nn.Linear(512, get_labelspace_size())
        model = torch.nn.DataParallel(model)
        model.to(dev)
    elif m_name == 'efficientnet-b0':
        if p_trained:
            print('Beginning with pretrained efficientnet-b0 architecture. Epochs = {}; Learning rate = {};'
                  ' momentum = {}\n'.format(epochs, lr, mom))
            model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            print('Beginning with untrained efficientnet-b0 architecture. Epochs = {}; Learning rate = {};'
                  ' momentum = {}\n'.format(epochs, lr, mom))
            model = EfficientNet.from_name('efficientnet-b0')
        model._fc = torch.nn.Linear(1280, get_labelspace_size())
        model = torch.nn.DataParallel(model)
        model.to(dev)
    elif m_name == 'efficientnet-b7':
        if p_trained:
            print('Beginning with pretrained efficientnet-b7 architecture. Epochs = {}; Learning rate = {};'
                  ' momentum = {}\n'.format(epochs, lr, mom))
            model = EfficientNet.from_pretrained('efficientnet-b7')
        else:
            print('Beginning with untrained efficientnet-b7 architecture. Epochs = {}; Learning rate = {};'
                  ' momentum = {}\n'.format(epochs, lr, mom))
            model = EfficientNet.from_name('efficientnet-b7')
        model._fc = torch.nn.Linear(2560, get_labelspace_size())
        model = torch.nn.DataParallel(model)
        model.to(dev)
    else:
        #  copied as placeholder - to be implemented
        if p_trained:
            print('Beginning with pretrained resnet-34 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
                  .format(epochs, lr, mom))
            model = resnet34(pretrained=True)
        else:
            print('Beginning with untrained resnet-34 architecture. Epochs = {}; Learning rate = {}; momentum = {}\n'
                  .format(epochs, lr, mom))
            model = resnet34(pretrained=False)
        model.fc = torch.nn.Linear(512, get_labelspace_size())
        model = torch.nn.DataParallel(model)
        model.to(dev)
    return model
