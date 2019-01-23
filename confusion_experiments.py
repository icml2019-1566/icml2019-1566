import torch
import numpy as np

import aux_funcs  as af
import model_funcs as mf
import network_architectures as arcs


def get_sdn_stats(layer_correct, layer_wrong, instances_catastrophic, instance_confusion):
    layer_keys = sorted(list(layer_correct.keys()))

    correct_con = []
    wrong_con = []
    wrong_wo_catas_con = []
    catas_con = []

    for inst in layer_correct[layer_keys[-1]]:
        correct_con.append(instance_confusion[inst])
        
    for inst in layer_wrong[layer_keys[-1]]:
        wrong_con.append(instance_confusion[inst])

    for inst in layer_wrong[layer_keys[-1]]:
        if inst in instances_catastrophic:
            catas_con.append(instance_confusion[inst])
        else:
            wrong_wo_catas_con.append(instance_confusion[inst])


    mean_correct_con = np.mean(correct_con)
    mean_wrong_con = np.mean(wrong_con)

    mean_wrong_wo_catas_con = np.mean(wrong_wo_catas_con)
    mean_catas_con = np.mean(catas_con)

    print('Confusion of corrects: {}, Confusion of wrongs: {}'.format(mean_correct_con, mean_wrong_con))
    print('Confusion of catastrophic: {}, Confusion of wrongs wo catastrophic: {}'.format(mean_catas_con, mean_wrong_wo_catas_con))

    return correct_con, wrong_con, catas_con, wrong_wo_catas_con

def get_cnn_stats(correct, wrong, instance_cert):
    print('get cnn stats')
    correct_certs = []
    wrong_certs = []

    for inst in correct:
        correct_certs.append(instance_cert[inst])
    for inst in wrong:
        wrong_certs.append(instance_cert[inst])

    mean_correct_certs = np.mean(correct_certs)
    mean_wrong_certs = np.mean(wrong_certs)

    print('Certainty of corrects: {}, Certainty of wrongs: {}'.format(mean_correct_certs, mean_wrong_certs))
    return correct_certs, wrong_certs


def get_catastrophic(models_path, device='cpu'):
    sdn_name = 'tinyimagenet_vgg16bn_sdn_converted'

    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'], add_trigger=False)
    
    d_layer_correct, d_layer_wrong, _, _ = mf.sdn_get_instances(sdn_model, loader=dataset.test_loader, device=device)
    layers = sorted(list(d_layer_correct.keys()))

    end_wrong = d_layer_wrong[layers[-1]]
    cum_correct = set()

    for layer in layers:
        cur_correct = d_layer_correct[layer]
        cum_correct = cum_correct | cur_correct
        cum_overthinking = cum_correct & end_wrong
        #print('Catastrophic overthinking: {}'.format(len(cum_overthinking)))

    return cum_overthinking


def model_confusion_experiment(models_path, device='cpu'):
    sdn_name = 'tinyimagenet_vgg16bn_sdn_converted'
    cnn_name = 'tinyimagenet_vgg16bn_cnn'

    sdn_model, sdn_params = arcs.load_model(models_path, sdn_name, epoch=-1)
    sdn_model.to(device)
    dataset = af.get_dataset(sdn_params['task'])

    sdn_images = 'confusion_images/{}'.format(sdn_name)
    cnn_images = 'confusion_images/{}'.format(cnn_name)

    cnn_model, _ = arcs.load_model(models_path, cnn_name, epoch=-1)
    cnn_model.to(device)

    top1_test, top5_test = mf.sdn_test(sdn_model, dataset.test_loader, device)
    print('SDN Top1 Test accuracy: {}'.format(top1_test))
    print('SDN Top5 Test accuracy: {}'.format(top5_test))

    top1_test, top5_test = mf.cnn_test(cnn_model, dataset.test_loader, device)
    print('CNN Top1 Test accuracy: {}'.format(top1_test))
    print('CNN Top5 Test accuracy: {}'.format(top5_test))


    # the the normalization stats from the training set
    confusion_stats = mf.sdn_confusion_stats(sdn_model, loader=dataset.train_loader, device=device)
    print(confusion_stats)
    # SETTING 1 - IN DISTRIBUTION
    d_layer_correct, d_layer_wrong, d_instance_confusion = mf.sdn_get_confusion(sdn_model, loader=dataset.test_loader, confusion_stats=confusion_stats, device=device)
    c_correct, c_wrong, c_instance_cert = mf.cnn_get_confidence(cnn_model, loader=dataset.test_loader, device=device)
    catastrophic = get_catastrophic(models_path, device)
    d_correct_con, d_wrong_con, d_catas_con, d_wrong_wo_catas_con = get_sdn_stats(d_layer_correct, d_layer_wrong, catastrophic, d_instance_confusion)
    c_correct_cert, c_wrong_cert = get_cnn_stats(c_correct, c_wrong, c_instance_cert)

    af.create_path(sdn_images)
    af.create_path(cnn_images)

    # corrects and wrongs
    af.overlay_two_histograms(sdn_images, '{}_corrects_wrongs'.format(sdn_name), d_correct_con, d_wrong_con, 'Correct', 'Wrong', 'SDN Confusion')
    
    af.overlay_three_histograms(sdn_images, '{}_corrects_wrongs_cat'.format(sdn_name), d_correct_con, d_catas_con, d_wrong_wo_catas_con, 'Correct', 'Catastrophic', 'Non-Catastrophic Wrong', 'SDN Confusion on Catastrophic Overthinking')

    af.overlay_two_histograms(cnn_images, '{}_corrects_wrongs'.format(cnn_name), c_correct_cert, c_wrong_cert, 'Correct', 'Wrong', 'CNN Confidence')

    mean_first = np.mean(d_correct_con)
    mean_second = np.mean(d_wrong_con)

    in_first_above_second = 0
    in_second_below_first = 0
    in_third_below_first = 0

    for item in d_correct_con:
        if float(item) > float(mean_second):
            in_first_above_second += 1

    for item in d_wrong_con:
        if float(item) < float(mean_first):
            in_second_below_first += 1

    for item in d_catas_con:
        if float(item) < float(mean_first):
            in_third_below_first += 1

    print('SDN more confused correct: {}/{}'.format(in_first_above_second, len(d_correct_con)))
    print('SDN less confused wrong: {}/{}'.format(in_second_below_first, len(d_wrong_con)))
    print('SDN less confused catastrophic overthinking: {}/{}'.format(in_third_below_first, len(d_catas_con)))


    mean_first = np.mean(c_correct_cert)
    mean_second = np.mean(c_wrong_cert)
    in_first_below_second = 0
    in_second_above_first = 0
    for item in c_correct_cert:
        if float(item) < float(mean_second):
            in_first_below_second += 1
    for item in c_wrong_cert:
        if float(item) > float(mean_first):
            in_second_above_first += 1

    print('CNN less confident correct: {}/{}'.format(in_first_below_second, len(c_correct_cert)))
    print('CNN more confident wrong: {}/{}'.format(in_second_above_first, len(c_wrong_cert)))

    # all instances
    d_id_coh = list(d_instance_confusion.values())
    c_id_cert = list(c_instance_cert.values())
    print('Avg CNN ID Confidence: {}'.format(np.mean(c_id_cert)))
    print('Avg SDN ID confusion: {}'.format(np.mean(d_id_coh)))
    print('Std SDN Confusion: {}'.format(np.std(d_id_coh)))
    print('Std CNN Confidence {}'.format(np.std(c_id_cert)))

    
def main():
    torch.manual_seed(af.get_random_seed())    # reproducible
    np.random.seed(af.get_random_seed())
    device = af.get_pytorch_device()
    trained_models_path = 'networks_icml/{}'.format(af.get_random_seed())

    model_confusion_experiment(trained_models_path, device)

if __name__ == '__main__':
    main()