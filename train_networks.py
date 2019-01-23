import copy
import torch
import time
import os
import random
import numpy as np

from sklearn.model_selection import ParameterGrid

import aux_funcs  as af
import network_architectures as arcs

from architectures.CNNs.VGG import VGG

def train(models_path, untrained_models, conversion=False, device='cpu'):
    print('Training models...')

    for base_model in untrained_models:
        trained_model, model_params = arcs.load_model(models_path, base_model, 0)
        dataset = af.get_dataset(model_params['task'])

        learning_rate = model_params['learning_rate']
        momentum = model_params['momentum']
        weight_decay = model_params['weight_decay']
        milestones = model_params['milestones']
        gammas = model_params['gammas']
        cur_epochs = model_params['epochs']
        model_params['optimizer'] = 'SGD'

        save_func = lambda model,epoch: arcs.save_model(model, None, models_path, trained_model_name, epoch) if epoch == 75 else None
        trained_model.scratch = True

        if conversion:
            learning_rate = model_params['conversion']['learning_rate']
            cur_epochs = model_params['conversion']['epochs']
            milestones = model_params['conversion']['milestones']
            gammas = model_params['conversion']['gammas']
            model_params['optimizer'] = 'Adam'
        
            save_func =  None
            trained_model.scratch = False


        optimization_params = (learning_rate, weight_decay, momentum)
        lr_schedule_params = (milestones, gammas)

        if conversion:
            optimizer, scheduler = af.get_ic_only_training_optimizer(trained_model, optimization_params, lr_schedule_params)
            trained_model_name = base_model+'_converted'
        else:
            optimizer, scheduler = af.get_sdn_training_optimizer(trained_model, optimization_params, lr_schedule_params)
            trained_model_name = base_model+'_trained'

        print('Training: {}...'.format(trained_model_name))
        trained_model.to(device)
        
        metrics = trained_model.train_func(trained_model, dataset, cur_epochs, optimizer, scheduler, save_func=save_func, device=device)
        model_params['train_top1_acc'] = metrics['train_top1_acc']
        model_params['test_top1_acc'] = metrics['test_top1_acc']
        model_params['train_top5_acc'] = metrics['train_top5_acc']
        model_params['test_top5_acc'] = metrics['test_top5_acc']
        model_params['epoch_times'] = metrics['epoch_times']
        model_params['lrs'] = metrics['lrs']
        total_training_time = sum(model_params['epoch_times'])
        model_params['total_time'] = total_training_time
        print('Training took {} seconds...'.format(total_training_time))
        arcs.save_model(trained_model, model_params, models_path, trained_model_name, epoch=-1)

def convert_models(models_path, networks, device='cpu'):
    conversion = True

    for sdn_name in networks:
        cnn_to_tune = sdn_name.replace('sdn', 'cnn')
        sdn_params = arcs.load_params(models_path, sdn_name)
        sdn_model, _ = af.cnn_to_sdn(models_path, cnn_to_tune, sdn_params, -1) # load the CNN and convert it to a sdn
        arcs.save_model(sdn_model, sdn_params, models_path, sdn_name, epoch=0) # save the resulting sdn
    
    train(models_path, networks, conversion=conversion, device=device)

def train_models(models_path, device='cpu'):
    tasks = ['cifar10', 'cifar100', 'tinyimagenet']

    cnns = []
    sdns = []

    for task in tasks:
        af.extend_lists(cnns, sdns, arcs.create_vgg16bn(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_resnet56(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_wideresnet32_4(models_path, task, save_type='cd'))
        af.extend_lists(cnns, sdns, arcs.create_mobilenet(models_path, task, save_type='cd'))

    train(models_path, cnns, conversion=False, device=device) # to train the CNNs
    train(models_path, sdns, conversion=False, device=device) # to train the sdns
    convert_models(models_path, sdns, device)


def main():
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks_icml/{}'.format(af.get_random_seed())
    af.create_path(models_path)
    #af.set_logger('outputs/sdn_{}_train'.format(af.get_random_seed()))

    train_models(models_path, device)

if __name__ == '__main__':
    main()
