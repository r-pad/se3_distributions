import os
import random
import model_renderer.pose_renderer
import generic_pose.eval.evaluate_render_distance as erd

def networkArgs(weight_file):
    network_name = weight_file.split('/')[-4]
    if(network_name == 'shapenet_bcewl_45deg'):
        compare_type = 'sigmoid'
        inverse_distance = True
        prefix = '/home/bokorn/results/shapenet/distance/shapenet_bcewl_45deg'
    elif(network_name == 'shapenet_exp_fo20_th25'):
        compare_type = 'sigmoid'
        inverse_distance = True
        prefix = '/'.join(weight_file.split('/')[:-3])
    elif(network_name == 'shapenet_exp_reg_fo20_th25'):
        compare_type = 'basic'
        inverse_distance = True
        prefix = '/'.join(weight_file.split('/')[:-3])
    elif(network_name == 'shapenet_negexp_fo20_th25'):
        compare_type = 'tanh'
        inverse_distance = False
        prefix = '/'.join(weight_file.split('/')[:-3])
    elif(network_name == 'shapenet_log_fo45_th25'):
        compare_type = 'sigmoid'
        inverse_distance = False
        prefix = '/'.join(weight_file.split('/')[:-3])

    return compare_type, inverse_distance, prefix

def getModelFilename(data_folder):
    if(data_folder[-1] == '/'):
        data_folder = data_folder[:-1]

    if('renders' in data_folder):
        dataset_type = 'numpy'
    else:
        dataset_type = 'linemod_masked'

    model_class, model_name = data_folder.split('/')[-2:]
    if(model_class == 'linemod'):
        model_filename = '/scratch/bokorn/data/benchmarks/linemod/' + \
                         '{}/mesh.ply'.format(model_name)
    else:
        model_filename = '/scratch/bokorn/data/models/shapenetcore/' + \
                         '{}/{}/model.obj'.format(model_class, model_name)

    return model_filename, dataset_type

results_prefix = '/home/bokorn/results/shapenet/distance/' 

linemod_data_folder_prefixes = ['/ssd0/bokorn/data/renders/linemod/',
                                '/scratch/bokorn/data/benchmarks/linemod/']

linemod_data_types = ['linemod_rendered',
                      'linemod_masked']

weight_filenames = ['/home/bokorn/results/shapenet/binary_angle/shapenet_bcewl_45deg/2018-05-17_17-42-17/weights/checkpoint_262000.pth',
                    '/home/bokorn/results/shapenet/distance/shapenet_exp_fo20_th25/2018-08-03_02-29-12/weights/checkpoint_86000.pth',
                    '/home/bokorn/results/shapenet/distance/shapenet_exp_reg_fo20_th25/2018-08-13_16-32-31/weights/checkpoint_10000.pth',
                    '/home/bokorn/results/shapenet/distance/shapenet_negexp_fo20_th25/2018-07-31_16-53-39/weights/checkpoint_33000.pth',
                    '/home/bokorn/results/shapenet/distance/shapenet_log_fo45_th25/2018-08-21_20-33-51/weights/checkpoint_90000.pth']

weight_filenames = weight_filenames[-1:]

linemod_models = ['ape',
                  'benchviseblue',
                  'cam',
                  'can',
                  'cat',
                  'driller',
                  'duck',
                  'eggbox',
                  'glue',
                  'holepuncher',
                  'iron',
                  'lamp',
                  'phone']

# Linemod Tests
for weight_file in weight_filenames:
    print(weight_file)
    compare_type, inverse_distance, weight_prefix = networkArgs(weight_file)
    for data_prefix, data_type in zip(linemod_data_folder_prefixes, linemod_data_types):
        for model in linemod_models:
            data_folder = data_prefix + model
            model_filename, dataset_type = getModelFilename(data_folder)
            results_prefix = '/'.join([weight_prefix, data_type + '_l2', model,''])
            os.makedirs(results_prefix, exist_ok=True)
            erd.main(weight_file,
                     model_filename,
                     data_folder,
                     dataset_type = dataset_type,
                     results_prefix = results_prefix,
                     compare_type = compare_type,
                     inverse_distance = inverse_distance,
                     base_level = 2)

shapenet_data_folders = ['/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/all_class_train_0.txt',
                         '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/all_class_model_valid_0.txt',
                         '/home/bokorn/src/generic_pose/generic_pose/training_sets/folder_sets/all_class_class_valid_0.txt']

shapenet_data_types = ['train',
                       'valid_model',
                       'valid_class']

shapenet_render_list = []
for data_file_list in shapenet_data_folders:
    with open(data_file_list, 'r') as f:
        model_list = f.read().split()
    shapenet_render_list.append(random.sample(model_list,5))

# Shapenet Tests
for weight_file in weight_filenames:
    compare_type, inverse_distance, weight_prefix = networkArgs(weight_file)
    for shapenet_models, data_type in zip(shapenet_render_list, shapenet_data_types):
        for data_folder in shapenet_models:
            model = data_folder.split('/')[-2]
            model_filename, dataset_type = getModelFilename(data_folder)
            results_prefix = '/'.join([weight_prefix, data_type + '_l2', model,''])
            os.makedirs(results_prefix, exist_ok=True)
            erd.main(weight_file,
                     model_filename,
                     data_folder,
                     dataset_type = dataset_type,
                     results_prefix = results_prefix,
                     compare_type = compare_type,
                     inverse_distance = inverse_distance,
                     base_level = 2)


