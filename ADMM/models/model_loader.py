import models.alexnet as alexnet
import models.alexnet_Mo as alexnet_mo
import models.vgg16 as vgg16
import models.vgg16_Mo as vgg16_mo


def load_model(arch, code_length):
    """
    Load cnn model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet.load_model(code_length)
    elif arch == 'vgg16':
        model = vgg16.load_model(code_length)
    else:
        raise ValueError('Invalid model name!')

    return model


def load_model_mo(arch):
    """
    Load cnn model.

    Args
        arch(str): CNN model name.
        code_length(int): Hash code length.

    Returns
        model(torch.nn.Module): CNN model.
    """
    if arch == 'alexnet':
        model = alexnet_mo.load_model()
    elif arch == 'vgg16':
        model = vgg16_mo.load_model()
    else:
        raise ValueError('Invalid model name!')

    return model
