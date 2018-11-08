from generic_pose.datasets.ycb_dataset import YCBDataset

def generateObjectImageSet(dataset, output_folder):
    obj_image_sets = {}
    for cls in dataset.classes[1:]:
        obj_image_sets[cls] = []

    image_set_file = os.path.join(dataset.data_dir, 'image_sets', dataset.image_set+'.txt')
    assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)

    with open(image_set_file) as f:
        filenames = [x.rstrip('\n') for x in f.readlines()]

    for fn in filenames:
        with open(os.path.join(dataset.data_dir, 'data', fn  + '-box.txt')) as f:
            video_id = fn.split('/')[-1]
            bboxes = [x.rstrip('\n').split(' ') for x in f.readlines()]
            for bb in bboxes:
                obj_image_sets[bb[0]].append(fn)
    
    for k,v in obj_image_sets.items():
        with open(os.path.join(output_folder, k+'_'+dataset.image_set+'.txt'), 'w') as f:
            f.write('\n'.join(v))


train_dataset = YCBDataset('/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset',
                           'train_split', img_size = (224,224))
train_dataset.loop_truth = [1]
train_loader = DataLoader(train_dataset, num_workers=0, batch_size=1, shuffle=False)
valid_dataset = YCBDataset('/scratch/bokorn/data/benchmarks/ycb/YCB_Video_Dataset',
                           'valid_split', img_size = (224,224))
valid_dataset.loop_truth = [1]
valid_loader = DataLoader(valid_dataset, num_workers=0, batch_size=1, shuffle=False)

results_folder = '/media/bokorn/Archive2018_2/seuss/results/'
finetuned_weight_files = {
    '002_master_chef_can':'ycb_finetune/002_master_chef_can/shapenet_exp_fo20_th25/2018-10-24_17-18-41/weights/checkpoint_17000.pth',
    '003_cracker_box':'ycb_finetune/003_cracker_box/shapenet_exp_fo20_th25/2018-10-24_17-20-17/weights/checkpoint_15000.pth',
    '006_mustard_bottle':'ycb_finetune/006_mustard_bottle/shapenet_exp_fo20_th25/2018-10-23_20-54-07/weights/checkpoint_10000.pth',
    '010_potted_meat_can':'ycb_finetune/010_potted_meat_can/shapenet_exp_fo20_th25/2018-10-24_16-35-46/weights/checkpoint_18000.pth',
    '011_banana':'ycb_finetune/011_banana/shapenet_exp_fo20_th25/2018-10-23_21-05-14/weights/checkpoint_11000.pth',
    '035_power_drill':'ycb_finetune/035_power_drill/shapenet_exp_fo20_th25/2018-10-23_20-53-36/weights/checkpoint_10000.pth',
    '037_scissors':'ycb_finetune/037_scissors/shapenet_exp_fo20_th25/2018-10-23_21-05-07/weights/checkpoint_11000.pth',
    '061_foam_brick':'ycb_finetune/061_foam_brick/shapenet_exp_fo20_th25/2018-10-24_16-37-08/weights/checkpoint_16000.pth'}

grid = S3Grid(2)

for obj, cls in enumerate(ycb_dataset.classes):
    if cls in finetuned_weight_files:
	model = gen_pose_net('alexnet', 
                             'sigmoid', 
                             output_dim = 1,
                             pretrained = False,
                             siamese_features = False)
        load_state_dict(model, results_folder+finetuned_weight_files[cls])
        model.eval()
        model.cuda()
        ycb_dataset.setObject(obj)
        idx, (query_imgs, _1, query_quats, _2, _3) in enumerate(train_loader)
            video_id = train_dataset.data_filenames[index].split('/')[-2]
            results = evaluateRenderedDistance(model, self.grid, self.renderer,
                                               query_imgs[0], query_quats[0],
                                               self.base_renders, self.base_vertices,
                                               loss_type = loss_type,
                                               falloff_angle = self.falloff_angle,
                                               optimizer = self.optimizer, 
                                                         disp_metrics = log_data,
                                                         num_indices = num_indices,
                                                         uniform_prop = uniform_prop,
                                                         loss_temperature = loss_temperature)
        idx, (query_imgs, _1, query_quats, _2, _3) in enumerate(valid_loader)
            video_id = valid_dataset.data_filenames[index].split('/')[-2]


