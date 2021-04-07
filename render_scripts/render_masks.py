# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:15:38 2018

@author: bokorn
"""
import time
def main():
    import os
    from torch.utils.data import DataLoader
    from se3_distributions.datasets.benchmark_dataset import LinemodDataset
    from tqdm import tqdm
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()


    t = time.time()
    data_loader = DataLoader(LinemodDataset(data_folders=args.data_folder,
                                            img_size = (224, 224),
                                            use_mask = True),
                             num_workers=args.num_workers, 
                             batch_size=args.batch_size, 
                             shuffle=False)
    #data_loader.dataset.loop_truth = [1]
    data_loader.dataset.setRenderMasks()
    print('Dataset initialization time: {}s'.format(round(time.time()-t, 2)))

    t = time.time()
    pbar = tqdm(enumerate(data_loader))

    for batch_idx, models in pbar:
        pbar.set_description(models[0][0])

if __name__=='__main__':
    main()
