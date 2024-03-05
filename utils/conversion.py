import torch.nn as nn
from neuron import *
from layers import *


def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

def isContainer(name):
    if 'container' in name.lower():
        return True
    return False

def addPreLayer(name):
    if 'conv2d' == name.lower() or 'linear' == name.lower() or 'pool' in name.lower() or 'flatten' in name.lower():
        return True
    return False

def addPostLayer(name):
    if 'pool' in name.lower() or 'flatten' in name.lower():
        return True
    return False

def addSingleStep(name):
    if  'lifspike' in name:
        return True
    if 'constrs' in name or 'baselayer' in name:
        if  'prelayer' in name or 'postlayer' in name:
            return False
        else:
            return True
    return False

def multi_to_single_step(model, multistep_ann, reset_mode):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = multi_to_single_step(module,multistep_ann,reset_mode)
        if addSingleStep(module.__class__.__name__.lower()):
            model._modules[name] = BaseLayer(vthr=model._modules[name].vthr, 
                                             tau=model._modules[name].tau, 
                                             multistep=False, 
                                             reset_mode=reset_mode)
        if  addPreLayer(module.__class__.__name__.lower()) and not multistep_ann:
            model._modules[name] = PreLayer(T=1, multistep=False, module=model._modules[name])
        if  addPostLayer(module.__class__.__name__.lower()) and not multistep_ann:
            model._modules[name] = PostLayer(T=1, multistep=False, module=model._modules[name]) 
        if  'prelayer' in module.__class__.__name__.lower():
            model._modules[name].multistep=False  
        if  'postlayer' in module.__class__.__name__.lower():
            model._modules[name].multistep=False  
    return model

def set_multistep(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = set_multistep(module)
        if hasattr(module, "multistep"):
            model._modules[name].multistep = True
    return model


def preprocess_ann_arch(model):
    model = set_multistep(model)
    model = nn.Sequential(
        *list(model.children())[1:],
        )
    return model


def set_dropout(model,p=0.0,training=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = set_dropout(module,p,training=training)
        if training:
            if  'baselayer' in module.__class__.__name__.lower():
                model._modules[name] = DropoutLayer(module=model._modules[name],p=p)
                model._modules[name].train()
        else:
            if  'dropoutlayer' in module.__class__.__name__.lower():
                model._modules[name] = model._modules[name].module
                model._modules[name].eval()
    return model


def _add_ann_layers(model, T, L, multistep_ann, ann_constrs, regularizer=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_ann_layers(module, T, L, multistep_ann, ann_constrs,regularizer)
        if  'relu' == module.__class__.__name__.lower():
            model._modules[name] = ann_constrs(T=T, L=L, regularizer=regularizer)
        if  addPreLayer(module.__class__.__name__.lower()) and multistep_ann:
            model._modules[name] = PreLayer(T=T, module=model._modules[name])
        if  addPostLayer(module.__class__.__name__.lower()) and multistep_ann:
            model._modules[name] = PostLayer(T=T, module=model._modules[name])    
    return model

def add_ann_layers(model, T, L, multistep_ann, ann_layers, regularizer=None):
    model = _add_ann_layers(model, T, L, multistep_ann, ann_layers, regularizer=regularizer)
    if multistep_ann:
        model = nn.Sequential(
            *list(model.children()),  
            PostLayer(T=T, module=None)    # Add the new layer
            )
    else:
        model = nn.Sequential(
            PreLayer(T=T, module=None),
            *list(model.children()),  
            PostLayer(T=T, module=None)    # Add the new layer
            )
    return model

def addSNNLayers(name):
    if 'relu' == name.lower() or 'lifspike' == name.lower():
        return True
    return False


def _add_snn_layers(model, T, snn_layers, regularizer=None, TEBN=None):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = _add_snn_layers(module, T, snn_layers,regularizer, TEBN=TEBN)
        if  addSNNLayers(module.__class__.__name__.lower()):
            model._modules[name] = snn_layers(T=T, regularizer=regularizer)
        if  addPreLayer(module.__class__.__name__.lower()):
            model._modules[name] = PreLayer(T=T, module=model._modules[name])
        if  addPostLayer(module.__class__.__name__.lower()):
            model._modules[name] = PostLayer(T=T, module=model._modules[name])    
        if TEBN:
            if  'norm2d' in module.__class__.__name__.lower():
                model._modules[name] = TEBNLayer(T=T, num_features=model._modules[name].num_features)  
    return model

def add_snn_layers(model, T, snn_layers, TEBN=False, regularizer=None):
    model = _add_snn_layers(model, T, snn_layers, regularizer=regularizer,TEBN=TEBN)
    model = nn.Sequential(
        *list(model.children()),  
        PostLayer(T=T, module=None)    # Add the new layer
        ) 
    return model



def reset_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = reset_neuron(module)
        if hasattr(module, "neuron"):
            model._modules[name].neuron.reset()
    return model


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model