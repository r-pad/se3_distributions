# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:15:38 2018

@author: bokorn
"""
import time
def main():
    import os
    from torch.utils.data import DataLoader
    from generic_pose.datasets.sixdc_dataset import SixDCDataset
    from tqdm import tqdm
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()


    t = time.time()
    data_loader = DataLoader(SixDCDataset(data_dir=args.data_dir,
                                          img_size = (224, 224),
                                          use_mask = True),
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size, 
                             shuffle=False)
    data_loader.dataset.setRenderMasks()
    print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    '''
    for seq in data_loader.dataset.sequence_names:
        print('Sequence {}'.format(seq))
        mask_dir = os.path.join(args.data_dir, 'test/{}/mask'.format(seq))
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        data_loader.dataset.setSequence(seq)
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, models in pbar:
            pass
            #pbar.set_description(models[0][0])
    '''
    for obj in data_loader.dataset.occluded_object_ids:
        #if(obj == 2):
        #    continue
        print('Occluded Object {}'.format(obj))
        data_loader.dataset.setSequence('02', obj=obj)
        pbar = tqdm(enumerate(data_loader), total=len(data_loader))
        for batch_idx, models in pbar:
            pass
            #pbar.set_description(models[0][0])

if __name__=='__main__':
    main()
