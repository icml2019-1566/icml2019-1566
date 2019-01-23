import torch
import torchvision.utils
import numpy as np
import pprint
import os
import time
from shutil import copyfile

from collections import Counter

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs

# To quantify the wasteful effects of overthinking
def wasteful_overthinking_experiment(models_path, device='cpu'):
    #task = 'cifar10'
    #task = 'cifar100'
    task = 'tinyimagenet'
    
    network = 'vgg16bn'
    #network = 'resnet56'
    #network = 'wideresnet32_4'
    #network = 'mobilenet'
    
    sdn_name = task + '_' + network + '_sdn_converted'
    
    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'])

    top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
    print('Top1 Test accuracy: {}'.format(top1_test))
    print('Top5 Test accuracy: {}'.format(top5_test))

    d_layer_correct, _, _, _ = mf.sdn_get_instances(sdn_model, loader=dataset.test_loader, device=device)
    layers = sorted(list(d_layer_correct.keys()))

    end_correct = d_layer_correct[layers[-1]]
    total = 10000
    c_i = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9]
    total_comp = 0

    cum_correct = set()
    for layer in layers:
        cur_correct = d_layer_correct[layer]
        unique_correct = cur_correct - cum_correct
        cum_correct = cum_correct | cur_correct
        cum_overthinking = cum_correct & end_correct
        unique_overthinking = unique_correct & end_correct 

        print('Output: {}'.format(layer))
        print('Current correct: {}'.format(len(cur_correct)))
        print('Cumulative correct: {}'.format(len(cum_correct)))
        print('Redundant overthinking: {}'.format(len(cum_overthinking)))
        print('Unique red. overthinking: {}'.format(len(unique_overthinking)))
        print('Unique correct: {}\n'.format(len(unique_correct)))

        if layer < layers[-1]:
                total_comp += len(unique_correct) * c_i[layer]
        else:
                total_comp += total - (len(cum_correct) - len(unique_correct))
        

    print('Total Comp: {}'.format(total_comp))


# to explain the wasteful effect
def get_simple_complex(models_path, device='cpu'):
    sdn_name = 'tinyimagenet_vgg16bn_sdn_converted'
    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'])
    output_path = 'simple_complex_images'
    af.create_path(output_path)
    dog_path = output_path+'/'+'dog'
    cat_path = output_path+'/'+'cat'
    af.create_path(dog_path)
    af.create_path(cat_path)

    # n02099601 dog 26
    # n02123394 cat 31
    
    d_layer_correct, d_layer_wrong, _, _ = mf.sdn_get_instances(sdn_model, loader=dataset.test_loader, device=device)
    layers = sorted(list(d_layer_correct.keys()))
    
    wrong_until = d_layer_wrong[layers[0]] | d_layer_correct[layers[0]]
    for layer in layers[:-1]:
        instances = d_layer_correct[layer] & wrong_until            
        wrong_until = wrong_until - d_layer_correct[layer]
        print('IC: {}, Num images: {}'.format(layer, len(instances)))
        for instance_id in instances:
            instance_path = dataset.testset_paths.imgs[instance_id][0]
            filename = '{}_{}'.format(layer, os.path.basename(instance_path))
            if 'n02099601' in instance_path:
                copyfile(instance_path, dog_path+'/'+filename)
            if 'n02123394' in instance_path:
                copyfile(instance_path, cat_path+'/'+filename)

# To quantify the destructive effects of overthinking
def destructive_overthinking_experiment(models_path, device='cpu'):
    #sdn_name = 'cifar10_vgg16bn_bd_sdn_converted'; add_trigger = True  # for the backdoored network
    
    add_trigger = False
    
    #task = 'cifar10'
    #task = 'cifar100'
    task = 'tinyimagenet'
   
    network = 'vgg16bn'
    #network = 'resnet56'
    #network = 'wideresnet32_4'
    #network = 'mobilenet'
    
    sdn_name = task + '_' + network + '_sdn_converted'

    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'], add_trigger=add_trigger)
    
    top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
    print('Top1 Test accuracy: {}'.format(top1_test))
    print('Top5 Test accuracy: {}'.format(top5_test))

    d_layer_correct, d_layer_wrong, _, d_layer_confidence = mf.sdn_get_instances(sdn_model, loader=dataset.test_loader, device=device)
    layers = sorted(list(d_layer_correct.keys()))

    end_wrong = d_layer_wrong[layers[-1]]
    cum_correct = set()

    for layer in layers:
        cur_correct = d_layer_correct[layer]
        unique_correct = cur_correct - cum_correct
        cum_correct = cum_correct | cur_correct
        cum_overthinking = cum_correct & end_wrong
        unique_overthinking = unique_correct & end_wrong
        cur_overthinking = cur_correct & end_wrong
        
        print('Output: {}'.format(layer))
        print('Current correct: {}'.format(len(cur_correct)))
        print('Cumulative correct: {}'.format(len(cum_correct)))
        print('Catastrophic overthinking: {}'.format(len(cum_overthinking)))
        print('Unique cat. overthinking: {}'.format(len(unique_overthinking)))
        print('Cur cat. overthinking: {}\n'.format(len(cur_overthinking)))

        total_confidence = 0.0
        for instance in cur_overthinking:
                total_confidence += d_layer_confidence[layer][instance]
        
        print('Average confidence on cat. overthinking instances:{}'.format(total_confidence/(0.1 + len(cur_overthinking))))
        
        total_confidence = 0.0
        for instance in cur_correct:
                total_confidence += d_layer_confidence[layer][instance]
        
        print('Average confidence on correctly classified :{}'.format(total_confidence/(0.1 + len(cur_correct))))

# to explain the destructive effect
def get_destructive_overthinking_samples(models_path, device='cpu'):
    sdn_name = 'tinyimagenet_vgg16bn_sdn_converted'
    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'])
    output_path = 'only_first'
    af.create_path(output_path)
    

    d_layer_correct, d_layer_wrong, d_layer_predictions, _ = mf.sdn_get_instances(sdn_model, loader=dataset.test_loader, device=device)
    layers = sorted(list(d_layer_correct.keys()))
    
    all_correct = set()

    for layer in layers[1:]:
            all_correct = all_correct | d_layer_correct[layer]
    
    only_first = d_layer_correct[layers[0]] - all_correct
    
    for instance_id in only_second:
        instance_path = dataset.testset_paths.imgs[instance_id][0]
        filename = os.path.basename(instance_path)
        print(instance_path)
        first_predict = d_layer_predictions[0][instance_id][0]
        last_predict = d_layer_predictions[layers[-1]][instance_id][0]
        first_predict = dataset.testset_paths.classes[first_predict]
        last_predict = dataset.testset_paths.classes[last_predict]

        filename = '{}_{}_{}'.format(first_predict, last_predict, filename)
        copyfile(instance_path, output_path'/'+filename)

def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device()
    trained_models_path = 'networks_icml/{}'.format(af.get_random_seed())
    
    wasteful_overthinking_experiment(trained_models_path, device)
    get_simple_complex(trained_models_path, device)

    destructive_overthinking_experiment(trained_models_path, device)
    get_destructive_overthinking_samples(trained_models_path, device)

if __name__ == '__main__':
    main()