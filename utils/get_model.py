
from models.vgglike import *
from models.vggsnn import *
from layers import *
from utils import add_ann_layers, add_snn_layers
from models.ResNet import get_resnet
from regularizer import *


def get_model(args):
    input_size  = InputSize(args.data.lower())
    num_classes  = OuputSize(args.data.lower())
    if args.method !='ann' and args.method !='snn':
        AssertionError('Training method is wrong!')

    if args.method=='ann':
        multistep = args.multistep_ann
        model = ann_models(args.model, input_size, num_classes,multistep)
        model = add_ann_layers(model, args.T, args.L, args.multistep_ann,
                                    ann_layers=get_layers(args.ann_layers.lower(),args.method), 
                                    regularizer=get_regularizer(args.regularizer.lower(),args.method))    
        return model
    elif args.method=='snn':
        model = ann_models(args.model,input_size,num_classes,multistep=True) 
        model = add_snn_layers(model, args.T,
                                snn_layers=get_layers(args.snn_layers.lower(),args.method), 
                                TEBN=args.TEBN,
                                regularizer=get_regularizer(args.regularizer.lower(),args.method),
                                )  
        return model
    else:
        NameError("The dataset name is not support!")
        exit(0)

def get_basemodel(name):
    if name.lower() in ['resnet18','resnet20','resnet34','resnet50','resnet101','resnet152']:
        return 'resnet'
    else:
        pass

def ann_models( model_name, input_size, num_classes,multistep):
    base_model = get_basemodel(model_name)
    if base_model == 'resnet':
        return get_resnet(model_name, input_size=input_size, num_classes=num_classes,multistep=multistep)
    elif model_name == 'vggann':
        return VGGANN(num_classes=num_classes)
    else:
        AssertionError('The network is not suported!')
        exit(0)

def InputSize(name):
    if 'cifar10-dvs' in name.lower() or 'dvs128-gesture' in name.lower():
        return 128 #'2-128-128'
    elif 'cifar10' in name.lower() or 'cifar100' in name.lower():
        return 32 #'3-32-32'
    else:
        NameError('This dataset name is not supported!')

def OuputSize(name):
    if 'cifar10-dvs' == name.lower() or 'cifar10' == name.lower() :
        return 10
    else:
        NameError('This dataset name is not supported!')


ann_regularizer = {
'none': None,
'rcs': RCSANN(),
}
snn_regularizer = {
'none': None,
'rcs': RCSSNN(),
} 

def get_regularizer(name: str, method: str):
    if method == 'ann':
        return ann_regularizer[name]
    elif method == 'snn':
        return snn_regularizer[name]


ann_layers = {
'qcfs': QCFS,
}
snn_layers = {
'baselayer': BaseLayer,
}

def get_layers(name: str, method: str):
    if method == 'ann':
        return ann_layers[name]
    elif method == 'snn':
        return snn_layers[name]

from regularizer import *


