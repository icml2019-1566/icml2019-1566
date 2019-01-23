import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs

from profiler import profile_sdn, profile

def early_prediction_experiments(models_path, device='cpu'):
    sdn_training_type = 'converted' # IC-only training
    #sdn_training_type = 'trained' # SDN training
    task = 'tinyimagenet'

    #sdn_names = ['vgg16bn_sdn', 'resnet56_sdn', 'wideresnet32_4_sdn', 'mobilenet_sdn']; add_trigger = False
    sdn_names = ['vgg16bn_bd_sdn']; task = 'cifar10'; sdn_training_type = 'converted'; add_trigger = True
    sdn_names = [task + '_' + sdn_name + '_' + sdn_training_type for sdn_name in sdn_names]

    for sdn_name in sdn_names:
        cnn_name = sdn_name.replace('sdn', 'cnn')
        cnn_name = cnn_name.replace('_converted', '')
        cnn_name  = cnn_name.replace('_trained', '')

        print(sdn_name)
        print(cnn_name)

        sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
        sdn_model.to(device)

        dataset = af.get_dataset(sdn_params['task'], add_trigger=add_trigger)

        cnn_model, _ = arcs.load_model(models_path, cnn_name, epoch=-1)
        cnn_model.to(device)

        print('Get CNN results')
        top1_test, top5_test, total_time = mf.cnn_test_time(cnn_model, dataset.test_loader, device)
        total_ops, total_params = profile(cnn_model, cnn_model.input_size, device)
        print("#Ops: %f GOps"%(total_ops/1e9))
        print("#Parameters: %f M"%(total_params/1e6))
        print('Top1 Test accuracy: {}'.format(top1_test))
        #print('Top5 Test accuracy: {}'.format(top5_test))

        print('25 percent cost: {}'.format((total_ops/1e9)*0.25))
        print('50 percent cost: {}'.format((total_ops/1e9)*0.5))
        print('75 percent cost: {}'.format((total_ops/1e9)*0.75))

        print('CNN forward took {} seconds.'.format(total_time))

        # to test cascading
        one_batch_dataset = af.get_dataset(sdn_params['task'], batch_size=1, add_trigger=add_trigger)
        print('Get SDN early predictions results')
        total_ops, total_params = profile_sdn(sdn_model, sdn_model.input_size, device)
        print("#Ops (GOps): {}".format(total_ops))
        print("#Params (mil): {}".format(total_params))
        top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
        print('Top1 Test accuracy: {}'.format(top1_test))
        print('Top5 Test accuracy: {}'.format(top5_test))

        confidence_params = [0.1, 0.15, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999] # grid search for threshold parameter q
        sdn_model.forward = sdn_model.early_forward

        for conf_threshold in confidence_params:
            print('Current threshold: {}'.format(conf_threshold))
            sdn_model.confidence_threshold = conf_threshold
            # change the forward func for sdn to forward with cascade
            top1_test, top5_test, early_output_counts, non_conf_output_counts, total_time = mf.sdn_test_early_predictions(sdn_model, one_batch_dataset.test_loader, device)
            average_mult_ops = 0
            total_num_instances = 0
            for output_id, output_count in enumerate(early_output_counts):
                average_mult_ops += output_count*total_ops[output_id]
                total_num_instances += output_count

            for output_count in non_conf_output_counts:
                total_num_instances += output_count
                average_mult_ops += output_count*total_ops[output_id]
            
            average_mult_ops /= total_num_instances

            print('Early output Counts:')
            print(early_output_counts)

            print('Non confident output counts:')
            print(non_conf_output_counts)

            print('Top1 Test accuracy: {}'.format(top1_test))
            print('Top5 Test accuracy: {}'.format(top5_test))
            print('SDN cascading took {} seconds.'.format(total_time))
            print('Average Mult-Ops: {}'.format(average_mult_ops))

def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device()
    trained_models_path = 'networks_icml/{}'.format(af.get_random_seed())
    early_prediction_experiments(trained_models_path, device)
    
if __name__ == '__main__':
    main()