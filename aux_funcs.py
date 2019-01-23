import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import os.path
import copy
import sys

import matplotlib
matplotlib.use('Agg')
import itertools as it

import matplotlib.pyplot as plt

import pickle
import itertools as it

from bisect import bisect_right
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable

import model_funcs as mf
import network_architectures as arcs

from profiler import profile, profile_sdn

from data import CIFAR10, CIFAR100, TinyImagenet


class Logger(object):
    def __init__(self, log_file, mode='out'):
        if mode == 'out':
            self.terminal = sys.stdout
        else:
            self.terminal = sys.stderr

        self.log = open('{}.{}'.format(log_file, mode), "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()

def set_logger(log_file):
    sys.stdout = Logger(log_file, 'out')
    #sys.stderr = Logger(log_file, 'err')

class MultiStepMultiLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gammas = gammas
        super(MultiStepMultiLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            cur_milestone = bisect_right(self.milestones, self.last_epoch)
            new_lr = base_lr * np.prod(self.gammas[:cur_milestone])
            new_lr = round(new_lr,8)
            lrs.append(new_lr)
        return lrs

def get_random_seed():
    return 1221 # 121 and 1221

def get_subsets(input_list, sset_size):
    return list(it.combinations(input_list, sset_size))

def set_random_seeds():
    torch.manual_seed(get_random_seed())
    np.random.seed(get_random_seed())
    random.seed(get_random_seed())

def extend_lists(list1, list2, items):
    list1.append(items[0])
    list2.append(items[1])


def feature_reduction_formula(input_feature_map_size):
    if input_feature_map_size >= 4:
        return int(input_feature_map_size/4)
    else:
        return -1

def get_uniform_dist(size):
    return torch.ones(size)*(1/size)

def get_distance_to_uniform(values, device):
    uniform_dist = get_uniform_dist(len(values)).to(device)
    return torch.dist(values, uniform_dist, 2)

def softmax(inp, T=1):
    max_val = np.max(inp)
    exps = np.exp((inp - max_val)/T)
    softmax = exps/np.sum(exps)
    return softmax

def single_histogram(save_path, save_name, hist_values, label, title):
    plt.hist(hist_values, bins=25, label=label)
    plt.axvline(np.mean(hist_values), color='k', linestyle='-', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    #plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()

def overlay_two_histograms(save_path, save_name, hist_first_values, hist_second_values, first_label, second_label, title):
    plt.hist([hist_first_values, hist_second_values], bins=25, label=[first_label, second_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()

def overlay_three_histograms(save_path, save_name, hist_first_values, hist_second_values, hist_third_values, first_label, second_label, third_label, title):
    plt.hist([hist_first_values, hist_second_values, hist_third_values], bins=25, label=[first_label, second_label, third_label])
    plt.axvline(np.mean(hist_first_values), color='k', linestyle='-', linewidth=3)
    plt.axvline(np.mean(hist_second_values), color='b', linestyle='--', linewidth=3)
    plt.axvline(np.mean(hist_third_values), color='r', linestyle=':', linewidth=3)

    plt.xlabel(title)
    plt.ylabel('Number of Instances')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig('{}/{}'.format(save_path, save_name))
    plt.close()

def get_confusion_scores(outputs, normalize=None, device='cpu'):
    p = 1 #np.log10(outputs[0].size(1))
    confusion_scores = torch.zeros(outputs[0].size(0))
    confusion_scores = confusion_scores.to(device)

    for output in outputs:
        cur_disagreement = nn.functional.pairwise_distance(outputs[-1], output, p=p)
        cur_disagreement = cur_disagreement.to(device)
        for instance_id in range(outputs[0].size(0)):
            confusion_scores[instance_id] +=  cur_disagreement[instance_id]
    
    if normalize is not None:
        for instance_id in range(outputs[0].size(0)):
            cur_confusion_score = confusion_scores[instance_id]
            cur_confusion_score = cur_confusion_score - normalize[0] # subtract mean
            cur_confusion_score = cur_confusion_score / normalize[1] # divide by the standard deviation
            confusion_scores[instance_id] = cur_confusion_score

    return confusion_scores

def get_dataset(dataset, batch_size=128, add_trigger=False):
    if dataset == 'cifar10':
        return load_cifar10(batch_size, add_trigger)
    if dataset == 'cifar100':
        return load_cifar100(batch_size)
    if dataset == 'tinyimagenet':
        return load_tinyimagenet(batch_size)

def load_cifar10(batch_size, add_trigger=False):
    cifar10_data = CIFAR10(batch_size=batch_size, add_trigger=add_trigger)
    return cifar10_data

def load_cifar100(batch_size):
    cifar100_data = CIFAR100(batch_size=batch_size)
    return cifar100_data

def load_tinyimagenet(batch_size):
    tiny_imagenet = TinyImagenet(batch_size=batch_size)
    return tiny_imagenet

def get_gradient_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def get_weight_norm(parameters, norm_type=2):
    total_norm = 0
    for p in parameters:
        param_norm = p.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

def get_random_order(all_list, count, prev_list=[]):
    if len(all_list) - len(prev_list) < count:
        p = random.sample(prev_list, len(all_list) - count)
    else:
        p = prev_list

    others = [idx for idx in all_list if idx not in p]
    return sorted(random.sample(others, count))

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SDNOutput(nn.Module):
    def __init__(self, input_size, output_channels, num_classes, alpha=0.5):
        super(SDNOutput, self).__init__()
        #red_kernel_size = -1 # to test the effects of the feature reduction
        red_kernel_size = feature_reduction_formula(input_size) # get the pooling size
        self.output_channels = output_channels

        if red_kernel_size == -1:
            self.linear = nn.Linear(output_channels*input_size*input_size, num_classes)
            self.forward = self.forward_wo_pooling
        else:
            red_input_size = int(input_size/red_kernel_size)
            self.max_pool = nn.MaxPool2d(kernel_size=red_kernel_size)
            self.avg_pool = nn.AvgPool2d(kernel_size=red_kernel_size)
            self.alpha = nn.Parameter(torch.rand(1)) # the mixing proportion
            self.linear = nn.Linear(output_channels*red_input_size*red_input_size, num_classes)
            self.forward = self.forward_w_pooling

    def forward_w_pooling(self, x):
        avgp = self.alpha*self.max_pool(x)
        maxp = (1 - self.alpha)*self.avg_pool(x)
        mixed = avgp + maxp
        return self.linear(mixed.view(mixed.size(0), -1))

    def forward_wo_pooling(self, x):
        return self.linear(x.view(x.size(0), -1))


def get_current_coeffs(init_coeffs, end_coeffs, cur_epoch, ramp_up_epochs):
    cur_coeffs = []
    for coeff_id, init_coeff in enumerate(init_coeffs):
        end_coeff = end_coeffs[coeff_id]
        increase = (end_coeff - init_coeff)/ramp_up_epochs
        cur_coeff = increase*(cur_epoch - 1) + init_coeff
        cur_coeffs.append(cur_coeff)
    
    return cur_coeffs

def get_ics_relative_depths(model):
    total_depth = model.init_depth
    output_depths = []

    for layer in model.layers:
        total_depth += layer.depth

        if layer.no_output == False:
            output_depths.append(total_depth)

    total_depth += model.end_depth
 
    return np.array(output_depths)/total_depth, total_depth

def generate_random_target_labels(true_labels, num_classes):
    target_labels = []
    for label in true_labels:
        cur_label = np.argmax(label)
        target_label = cur_label
        while target_label == cur_label:
            target_label = np.random.randint(0, num_classes)
        
        target_labels.append(target_label)

    return np.array(target_labels)

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_pred_single(model, img):
    prep_img = Variable(torch.from_numpy(img.reshape(1,1,28,28)).float(), requires_grad=True)
    output = model(prep_img)
    return output.max(1, keepdim=True)[1].numpy()[0][0]

def model_exists(models_path, model_name):
    return os.path.isdir(models_path+'/'+model_name)

def get_nth_occurance_index(input_list, n):
    if n == -1:
        return len(input_list) - 1
    else:
        return [i for i, n in enumerate(input_list) if n == 1][n]

def get_lr(optimizers):
    if isinstance(optimizers, dict):
        return optimizers[list(optimizers.keys())[-1]].param_groups[-1]['lr']
    else:
        return optimizers.param_groups[-1]['lr']

def get_optimizer(model, lr_params, stepsize_params):
    lr=lr_params[0]
    weight_decay=lr_params[1]
    momentum=lr_params[2]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler


def get_sdn_training_optimizer(model, lr_params, stepsize_params):
    lr=lr_params[0]
    weight_decay=lr_params[1]
    momentum=lr_params[2]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler

def get_ic_only_training_optimizer(model, lr_params, stepsize_params):
    freeze_except_outputs(model)

    lr=lr_params[0]
    weight_decay=lr_params[1]

    milestones = stepsize_params[0]
    gammas = stepsize_params[1]

    param_list = []
    for layer in model.layers:
        if layer.no_output == False:
            param_list.append({'params': filter(lambda p: p.requires_grad, layer.output.parameters())})
        
    optimizer = Adam(param_list, lr=lr, weight_decay=weight_decay)
    scheduler = MultiStepMultiLR(optimizer, milestones=milestones, gammas=gammas)

    return optimizer, scheduler

def get_pytorch_device():
    device = 'cpu'
    cuda = torch.cuda.is_available()
    print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)
    if cuda:
        device = 'cuda'
    return device


def tensor_getter(shape, device):
    if device == 'cpu':
        return torch.FloatTensor(shape)
    else:
        return torch.cuda.FloatTensor(shape)


def get_loss_criterion():
    return CrossEntropyLoss()

def drop_sdn_outputs(sdn_model, keeps):
    output_id = 0
    new_add_output = copy.deepcopy(sdn_model.add_output)

    for layer_id, layer in enumerate(sdn_model.layers):
        if new_add_output[layer_id] == 1:
            if output_id not in keeps:
                layer.output = None
                new_add_output[layer_id] = 0
                layer.forward = layer.only_forward
            output_id +=1

    sdn_model.add_output = new_add_output
    sdn_model.num_output = sum(new_add_output) + 1

def get_avg_l2_distance(x_orig, x_adv):
    distances = []
    for idx, instance in enumerate(x_orig):
        adv_instance = x_adv[idx]
        distances.append(np.linalg.norm(adv_instance-instance))

    return np.mean(distances)

def get_all_trained_models_info(models_path, use_profiler=False, device='gpu'):
    print('Testing all models in: {}'.format(models_path))

    for model_name in sorted(os.listdir(models_path)):
        try:
            model_params = arcs.load_params(models_path, model_name, -1)
            train_time = model_params['total_time']
            num_epochs = model_params['epochs']
            architecture = model_params['architecture']
            print(model_name)
            task = model_params['task']
            print(task)
            net_type = model_params['network_type']
            print(net_type)
            
            top1_test = model_params['test_top1_acc']
            top1_train = model_params['train_top1_acc']
            top5_test = model_params['test_top5_acc']
            top5_train = model_params['train_top5_acc']


            print('Top1 Test accuracy: {}'.format(top1_test[-1]))
            print('Top5 Test accuracy: {}'.format(top5_test[-1]))
            print('\nTop1 Train accuracy: {}'.format(top1_train[-1]))
            print('Top5 Train accuracy: {}'.format(top5_train[-1]))

            print('Training time: {}, in {} epochs'.format(train_time, num_epochs))

            if use_profiler:
                model, _ = arcs.load_model(models_path, model_name, epoch=-1)
                model.to(device)
                input_size = model_params['input_size']

                if architecture == 'sdn':
                    total_ops, total_params = profile_sdn(model, input_size, device)
                    print("#Ops (GOps): {}".format(total_ops))
                    print("#Params (mil): {}".format(total_params))

                else:
                    total_ops, total_params = profile(model, input_size, device)
                    print("#Ops: %f GOps"%(total_ops/1e9))
                    print("#Parameters: %f M"%(total_params/1e6))
            
            print('------------------------')
        except:
            print('FAIL: {}'.format(model_name))
            continue

def get_single_model(sdn_name, models_path, device):
    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = get_dataset(sdn_params['task'])
    top1_test, top5_test = sdn_model.test_func(sdn_model, dataset.test_loader, device)
    print('Top1 Test accuracies: {}'.format(top1_test))
    print('Top5 Test accuracies: {}'.format(top5_test))

    total_ops, total_params = profile_sdn(model, input_size, device)
    print("#Ops (GOps): {}".format(total_ops))
    print("#Params (mil): {}".format(total_params))



def sdn_prune(sdn_path, sdn_name, prune_after_output, epoch=-1, preloaded=None):
    print('Pruning a SDN...')

    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]
    
    output_layer = get_nth_occurance_index(sdn_model.add_output, prune_after_output)

    pruned_model = copy.deepcopy(sdn_model)
    pruned_params = copy.deepcopy(sdn_params)

    new_layers = nn.ModuleList()
    prune_add_output = []

    for layer_id, layer in enumerate(sdn_model.layers):
        if layer_id == output_layer:
            break
        new_layers.append(layer)
        prune_add_output.append(sdn_model.add_output[layer_id])

    last_conv_layer = sdn_model.layers[output_layer]
    end_layer = copy.deepcopy(last_conv_layer.output)

    last_conv_layer.output = nn.Sequential()
    last_conv_layer.forward = last_conv_layer.only_forward
    last_conv_layer.no_output = True
    new_layers.append(last_conv_layer)

    pruned_model.layers = new_layers
    pruned_model.end_layers = end_layer

    pruned_model.add_output = prune_add_output
    pruned_model.num_output = prune_after_output + 1

    pruned_params['pruned_after'] = prune_after_output
    pruned_params['pruned_from'] = sdn_name

    return pruned_model, pruned_params

# convert a cnn to a sdn by adding output layers to internal layers
def cnn_to_sdn(cnn_path, cnn_name, sdn_params, epoch=-1, preloaded=None):
    print('Converting a CNN to a SDN...')
    if preloaded is None:
        cnn_model, _ = arcs.load_model(cnn_path, cnn_name, epoch=epoch)
    else:
        cnn_model = preloaded

    sdn_params['architecture'] = 'sdn'
    sdn_params['converted_from'] = cnn_name
    sdn_model = arcs.get_sdn(cnn_model)(sdn_params)

    sdn_model.init_conv = cnn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, cnn_layer in enumerate(cnn_model.layers):
        sdn_layer = sdn_model.layers[layer_id]
        sdn_layer.layers = cnn_layer.layers
        layers.append(sdn_layer)
    
    sdn_model.layers = layers

    sdn_model.end_layers = cnn_model.end_layers

    return sdn_model, sdn_params

def sdn_to_cnn(sdn_path, sdn_name, epoch=-1, preloaded=None):
    print('Converting a SDN to a CNN...')
    if preloaded is None:
        sdn_model, sdn_params = arcs.load_model(sdn_path, sdn_name, epoch=epoch)
    else:
        sdn_model = preloaded[0]
        sdn_params = preloaded[1]

    cnn_params = copy.deepcopy(sdn_params)
    cnn_params['architecture'] = 'cnn'
    cnn_params['converted_from'] = sdn_name
    cnn_model = arcs.get_cnn(sdn_model)(cnn_params)

    cnn_model.init_conv = sdn_model.init_conv

    layers = nn.ModuleList()
    for layer_id, sdn_layer in enumerate(sdn_model.layers):
        cnn_layer = cnn_model.layers[layer_id]
        cnn_layer.layers = sdn_layer.layers
        layers.append(cnn_layer)
    
    cnn_model.layers = layers

    cnn_model.end_layers = sdn_model.end_layers

    return cnn_model, cnn_params


def freeze_except_outputs(model):
    model.frozen = True
    for param in model.init_conv.parameters():
        param.requires_grad = False

    for layer in model.layers:
        for param in layer.layers.parameters():
            param.requires_grad = False

    for param in model.end_layers.parameters():
        param.requires_grad = False

def save_tinyimagenet_classname():
    filename = 'tinyimagenet_classes'
    dataset = get_dataset('tinyimagenet')
    tinyimagenet_classes = {}
    
    for index, name in enumerate(dataset.testset_paths.classes):
        tinyimagenet_classes[index] = name

    with open(filename, 'wb') as f:
        pickle.dump(tinyimagenet_classes, f, pickle.HIGHEST_PROTOCOL)

def get_tinyimagenet_classes(prediction=None):
    filename = 'tinyimagenet_classes'
    with open(filename, 'rb') as f:
        tinyimagenet_classes = pickle.load(f)
    
    if prediction is not None:
        return tinyimagenet_classes[prediction]

    return tinyimagenet_classes

def main():
    torch.manual_seed(get_random_seed())    # reproducible
    np.random.seed(get_random_seed())
    device = get_pytorch_device()
    trained_models_path = 'networks_icml/{}'.format(get_random_seed())

    #get_all_trained_models_info(trained_models_path, use_profiler=True, device='cpu')
    get_single_model('tinyimagenet_vgg16bn_sdn_trained', trained_models_path, device)
if __name__ == '__main__':
    main()