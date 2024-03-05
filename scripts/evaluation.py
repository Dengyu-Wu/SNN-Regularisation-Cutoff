import os
import torch
import warnings
import torch.optim
import hydra

import utils.data_loaders as data_loaders
import numpy as np
from configs import *
from omegaconf import DictConfig
from utils import multi_to_single_step, preprocess_ann_arch
from utils import get_model
from utils import save_pickle
import torch.backends.cudnn as cudnn
from utils import set_seed

import torch
import torch.nn as nn
from typing import Type
from pydantic import BaseModel
from utils import OutputHook, sethook, set_dropout
from cutoff import TopKCutoff

class Evaluator:
    def __init__(
        self,
        net: nn.Module,
        args: Type[BaseModel]=None,
    ) -> None:
        """A unified, easy-to-use API for evaluating (most) discriminative OOD
        detection methods.

        Args:
            net (nn.Module):
                The base classifier.
            cutoff (Type[BaseCutoff], optional):
                An actual cutoff instance which inherits
                SNNCutoff's BaseCutoff. Defaults to None.
        Raises:
            ValueError:
                If both postprocessor_name and postprocessor are None.
            ValueError:
                If the specified ID dataset {id_name} is not supported.
            TypeError:
                If the passed postprocessor does not inherit BasePostprocessor.
        """
        self.net = net
        self.args = args
        self.net.eval()
        self.cutoff = TopKCutoff(T=args.T, bin_size=100,add_time_dim=args.add_time_dim, multistep=args.multistep)
        self.T = args.T
        self.add_time_dim = args.add_time_dim

    def evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        outputs_list = torch.softmax(outputs_list,dim=-1)
        acc =(outputs_list.max(-1)[1] == new_label).float().sum(1)/label_list.size()[0]
        return acc.cpu().numpy().tolist(), 0.0
    
    def oct_evaluation(self,data_loader):
        outputs_list, label_list = self.cutoff.inference(net=self.net, data_loader=data_loader)
        new_label = label_list.unsqueeze(0)
        index = (outputs_list.max(-1)[1] == new_label).float()
        for t in range(self.T-1,0,-1):
            index[t-1] = index[t]*index[t-1]
        index[-1] = 1.0
        index = torch.argmax(index,dim=0)
        mask = torch.nn.functional.one_hot(index, num_classes=self.T)
        outputs_list = outputs_list*mask.transpose(0,1).unsqueeze(-1)
        outputs_list = outputs_list.sum(0)
        acc = (outputs_list.max(-1)[1]  == new_label[0]).float().sum()/label_list.size()[0]
        return acc.cpu().numpy().item(), (index+1).cpu().numpy()

    def cutoff_evaluation(self,data_loader,train_loader,epsilon=0.0):
        acc, timestep, conf = self.cutoff.cutoff_evaluation(net=self.net, 
                                                            data_loader=data_loader,
                                                            train_loader=train_loader,
                                                            epsilon=epsilon)
        return acc, timestep, conf

    def ANN_OPS(self,input_size):
            net = self.net
            print('ANN MOPS.......')
            output_hook = OutputHook(get_connection=True)
            net = sethook(output_hook)(net)
            inputs = torch.randn(input_size).unsqueeze(0).to(net.device)
            outputs = net(inputs)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)
            tot_fp = 0
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                N_neuron = torch.prod(torch.tensor(output))
                tot_fp += (fin*2+1)*N_neuron
            print(tot_fp)
            return tot_fp
    
    def SNN_Spike_Count(self,input_size):
            net = self.net
            connections = []
            output_hook = OutputHook(get_connection=True)
            net = sethook(output_hook)(net)
            inputs = torch.randn(input_size).unsqueeze(0).to(net.device)
            outputs = net(inputs)
            connections = list(output_hook)
            net = sethook(output_hook)(net,remove=True)

            tot_fp = 0
            tot_bp = 0
            for name,w,output in connections:
                fin = torch.prod(torch.tensor(w))
                N_neuron = torch.prod(torch.tensor(output))
                tot_fp += (fin*2+1)*N_neuron
                tot_bp += 2*fin + (fin*2+1)*N_neuron
            tot_op = self.Nops[0]*tot_fp + self.Nops[1]*tot_bp
            return [tot_op, tot_fp, tot_bp]



@hydra.main(version_base=None, config_path='../configs', config_name='test')
def main(cfg: DictConfig):
    args = TestConfig(**cfg['base'], **cfg['snn-train'], **cfg['snn-test'])
    if args.seed is not None:
        set_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = data_loaders.get_data_loaders(path=args.dataset_path, data=args.data, transform=False,resize=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    models = get_model(args)
    i= 0
    path = args.model_path
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    models.load_state_dict(state_dict, strict=False)
    if not args.multistep:
        if not args.multistep_ann:
            models = preprocess_ann_arch(models)
        models = multi_to_single_step(models, args.multistep_ann, reset_mode=args.reset_mode)
    models.to(device)
    evaluator = Evaluator(models,args=args)
    acc, loss = evaluator.evaluation(test_loader)
    print('accuracy w.r.t timestep:', acc)
    print(np.mean(loss))
    result={'accuracy': acc}
    save_pickle(result,name='timestep_cutoff',path=os.path.dirname(path))

    acc, timesteps = evaluator.oct_evaluation(test_loader)
    print('OCT accuracy:', acc, 'OCT timesteps:', np.mean(timesteps))
    result={'accuracy': acc, 'timesteps': timesteps}
    save_pickle(result,name='oct_timestep',path=os.path.dirname(path))

    acc=[]
    timesteps=[]
    samples_number=[]
    for i in range(10):
        epsilon = args.epsilon*i
        _acc, _timesteps, _samples_number = evaluator.cutoff_evaluation(test_loader,train_loader=train_loader,epsilon=epsilon)
        acc.append(_acc)
        timesteps.append(_timesteps)
        samples_number.append(_samples_number)
    acc = np.array(acc)
    timesteps = np.array(timesteps)
    print('TopK accuracy:', acc, 'TopK timestep:', timesteps.mean())
    result={'accuracy': acc, 'timesteps': timesteps, 'samples_number':samples_number}
    save_pickle(result,name='topk_cutoff',path=os.path.dirname(path))

if __name__ == '__main__':
   main()
