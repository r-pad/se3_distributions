# -*- coding: utf-8 -*-
"""
Created on Someday Sometime

@author: bokorn
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import scipy as sp
import scipy.stats

import glob
import shutil

import logging

def plotOutputBoxPlot(img_prefix, data, title_name=None):
    binned_output = []
    angles = data[0]*180/np.pi
    edges = np.linspace(0.0,180.0,100)
    bin_width = edges[1]-edges[0]
    
    for j in range(len(edges)-1):
        binned_output.append(data[1,np.bitwise_and(angles >= edges[j], angles < edges[j+1])])
    bp = plt.boxplot(binned_output, 0, '.')#flierprops={'size':1,'marker':'.'})
    for flier in bp['fliers']:
        flier.set(marker='.', markersize=1)

    plt.xticks(np.arange(0,180,25)/bin_width, np.arange(0,180,25))
    plt.xlabel('Angle Difference (degrees)')
    plt.ylabel('Mean Output')
    if(title_name is not None):
        plt.title('{} Set'.format(title_name))

    plt.savefig(img_prefix + 'output_bp.png')
    plt.gcf().clear()

def plotOutput(img_prefix, data, title_name=None, histogram_bins = 100):
    binned_output = []
    angles = data[0]*180/np.pi
    
    counts, edges = np.histogram(angles, bins = histogram_bins, range=(0.0,180.0))
    output_hist, _ = np.histogram(angles, weights=data[1], 
                                  bins = histogram_bins, range=(0.0,180.0))
    output_hist /= counts
    
    plt.plot(edges[:-1], output_hist) 
    if(title_name is not None):
        plt.title('{} Set'.format(title_name))

    plt.xlabel('Angle Difference (degrees)')
    plt.ylabel('Mean Output')
    if(title_name is not None):
        plt.title('{} Set'.format(title_name))

    plt.savefig(img_prefix + 'output.png')
    plt.gcf().clear()
    return output_hist, counts

def plotGroundTruthRank(img_prefix, gt_ranks, title_name = None,
                        histogram_bins = 100, num_verts = 3885):
    max_bin = num_verts#400
    #gt_ranks = np.array(gt_ranks)
    #gt_ranks[gt_ranks > max_bin] = max_bin 
    counts, edges = np.histogram(gt_ranks, bins = histogram_bins, range=(0,max_bin))
    plt.bar(edges[:-1], counts, width=edges[1]-edges[0])
    
    #if(title_name is not None):
    #    pass
    #    plt.title('{} Set'.format(title_name))

    plt.xlabel('Ground Truth Rank')
    plt.ylabel('Counts')

    plt.savefig(img_prefix + 'gt_rank.png')
    plt.gcf().clear()
    return counts

def plotTopRank(img_prefix, top_ranks, title_name = None, 
                histogram_bins = 100, num_verts = 3885):
    counts, edges = np.histogram(top_ranks, bins = histogram_bins, range=(0,num_verts))
    plt.bar(edges[:-1], counts, width=edges[1]-edges[0])
   
    if(title_name is not None):
        plt.title('{} Set'.format(title_name))

    plt.xlabel('Top Output Rank')
    plt.ylabel('Counts')

    plt.savefig(img_prefix + 'top_rank.png')
    plt.gcf().clear()
    return counts

def plotTopDistance(img_prefix, top_dists, title_name = None, histogram_bins = 100):
    counts, edges = np.histogram(top_dists, bins = histogram_bins, range=(0,180))
    plt.bar(edges[:-1], counts, width=edges[1]-edges[0])
   
    if(title_name is not None):
        plt.title('{} Set'.format(title_name))

    plt.xlabel('Top Ranked Angle')
    plt.ylabel('Counts')

    plt.savefig(img_prefix + 'top_dist.png')
    plt.gcf().clear()
    return counts

def plotGroundTruthOutput(img_prefix, gt_outputs, title_name = None, histogram_bins = 100):
    counts, edges = np.histogram(gt_outputs, bins = histogram_bins, range=(0,1))
    plt.bar(edges[:-1], counts, width=edges[1]-edges[0])
   
    if(title_name is not None):
        plt.title('{} Set'.format(title_name))

    plt.xlabel('Ground Truth Output')
    plt.ylabel('Counts')

    plt.savefig(img_prefix + 'gt_output.png')
    plt.gcf().clear()
    return counts 

def plotResults(img_prefix, data, title_name = None, box_plot = True):
    if(box_plot):
        plotOutputBoxPlot(img_prefix, data['data'], title_name=title_name)
    output_hist, counts = plotOutput(img_prefix, data['data'], title_name=title_name)
    gt_rank_counts = plotGroundTruthRank(img_prefix, data['gt_ranks'], title_name=title_name)
    top_rank_counts = plotTopRank(img_prefix, data['top_ranks'], title_name=title_name)
    gt_output_counts = plotGroundTruthOutput(img_prefix, data['gt_outputs'], title_name=title_name)
    top_dist_counts = plotTopDistance(img_prefix, data['top_dists'], title_name=title_name)
    return output_hist, counts, gt_rank_counts, top_rank_counts, gt_output_counts, top_dist_counts

def translateFilename(filename):
    dist_type, data_type, model = filename.split('/')[-4:-1]
    #data_type, dist_type, model = filename.split('/')[-4:-1]
    if(True):
        title_name = 'Sigmoid Exp Distance (20 degrees)\n'
    #if('_l2' in data_type):
    #    data_type = data_type.replace('_l2', '')
    #if(dist_type in ['shapenet_bcewl_45deg', 'bce2']):
    #    title_name = 'Binary BCE (45 degrees)\n'
    elif(dist_type == 'shapenet_exp_fo20_th25'):
        title_name = 'Sigmoid Exp Distance (20 degrees)\n'
    elif(dist_type == 'shapenet_exp_reg_fo20_th25'):
        title_name = 'Raw Exp Distance (20 degrees)\n'
    elif(dist_type == 'shapenet_negexp_fo20_th25'):
        title_name = 'Tanh Exp Distance (20 degrees)\n'
    elif(dist_type == 'shapenet_log_fo45_th25'):
        title_name = 'Sigmoid Log Distance (45 degrees)\n'
    else:
        raise ValueError('Distance Type {} not Implemented'.format(dist_type))

    
    if(model == 'eval'):
        title_name += 'YCB Power Drill Shapenet: '
    elif(model == 'base'):
        title_name += 'YCB Power Drill Shapenet: '
    #if(data_type == 'linemod_masked'):
    #    title_name += 'Linemod Masked: '
    elif(data_type == 'linemod_rendered'):
        title_name += 'Linemod Rendered: '
    elif(data_type == 'linemod6dc'):
        title_name += 'Linemod 6D Challenge: '
    elif(data_type == 'train'):
        title_name += 'Shapenet Training Set: '
    elif(data_type == 'valid_model'):
        title_name += 'Shapenet Model Validation Set: '
    elif(data_type == 'valid_class'):
        title_name += 'Shapenet Class Validation Set: '
    else:
        raise ValueError('Data Type {} not Implemented'.format(data_type))
    
    title_name += model.capitalize()
    return title_name, dist_type, data_type, model

#results_folder = '/home/bokorn/results/linemod6dc/bce2'
#results_folder = '/home/bokorn/results/shapenet/distance/shapenet_bcewl_45deg/linemod_masked_l2'

results_folder = '/home/bokorn/results/ycb_finetune/035_power_drill'
#results_folder = '/home/bokorn/results/ycb_finetune/035_power_drill/eval'
#results_folder = '/home/bokorn/results/ycb_finetune/035_power_drill/base'
result_files = glob.glob(results_folder+'/**/distance.npz', recursive=True)

last_data_type = None
last_filename = None
data_type_data = {'data':[],
                  'gt_ranks':[],
                  'top_ranks':[],
                  'gt_outputs':[],
                  'top_dists':[]}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Viz')

mean_results = {}
plot_results = True

for filename in sorted(result_files):
    #logger.info(filename)
    data = np.load(filename)
    #import IPython; IPython.embed()
    #if('gt_outputs' not in data.keys()):
    #    #logger.debug('{} does not contain gt_outputs'.format(filename))
    #    continue
    #if('/old/' in filename):
    #    continue
    #if('_l2' not in filename):
    #    continue

    img_prefix = '/'.join(filename.split('/')[:-1])+'/'
    '''
    img_filenames = glob.glob(img_prefix + '*img*.png')
    txt_filenames = glob.glob(img_prefix + '*.txt')
    for fn in img_filenames:
        shutil.move(fn, img_prefix+'images')
    for fn in txt_filenames:
        shutil.move(fn, img_prefix+'images')
    continue
    '''
    title_name, dist_type, data_type, model = translateFilename(filename)
    if(plot_results):
        plotResults(img_prefix, data, title_name)
    data_type = filename.split('/')[-3]

    if(last_data_type is not None and data_type != last_data_type):
        results = mean_results.get(dist_type, {}).get(data_type, {})
        #import IPython; IPython.embed()
        data_type_title = translateFilename(last_filename)[0].split(':')[0]
        data_type_data['data'] = np.hstack(data_type_data['data'])
        logger.info('{} Mean GT Rank: {:0.2f}'.format(data_type_title.replace('\n',': '), 
                                                   np.mean(data_type_data['gt_ranks'])))
        results['gt_rank'] = np.mean(data_type_data['gt_ranks'])
        results['gt_output'] = np.mean(data_type_data['gt_outputs'])
        results['output'] = np.mean(data_type_data['data'][1])
        results['top_rank'] = np.mean(data_type_data['top_ranks'])
        results['top_dist'] = np.mean(data_type_data['top_dists'])

        #logger.info(data_type_title.replace('\n',': '))
        #logger.info('Mean GT Rank:   {}'.format(np.mean(data_type_data['gt_ranks'])))
        #logger.info('Mean GT Output: {}'.format(np.mean(data_type_data['gt_outputs'])))
        #logger.info('Mean Output:    {}'.format(np.mean(data_type_data['data'][1])))
        #logger.info('Mean Top Rank:  {}'.format(np.mean(data_type_data['top_ranks'])))
        logger.info('Mean Top Angle: {}'.format(np.mean(data_type_data['top_dists'])))
        if(plot_results):
            plotResults('/'.join(last_filename.split('/')[:-2])+'/',
                        data_type_data, data_type_title,
                        box_plot = True)#False)'''
        data_type_data = {'data':[],
                          'gt_ranks':[],
                          'top_ranks':[],
                          'gt_outputs':[],
                          'top_dists':[]}
    
    last_data_type = data_type
    last_filename = filename
    data_type_data['data'].append(data['data'])
    data_type_data['gt_ranks'].extend(data['gt_ranks'])
    data_type_data['top_ranks'].extend(data['top_ranks'])
    data_type_data['gt_outputs'].extend(data['gt_outputs'])
    data_type_data['top_dists'].extend(data['top_dists'])

data_type_title = translateFilename(last_filename)[0].split(':')[0]
data_type_data['data'] = np.hstack(data_type_data['data'])
logger.info('{} Mean GT Rank: {:0.2f}'.format(data_type_title.replace('\n',': '), 
np.mean(data_type_data['gt_ranks'])))
#logger.info(data_type_title.replace('\n',': '))
logger.info('Mean GT Rank:   {}'.format(np.mean(data_type_data['gt_ranks'])))
logger.info('Mean GT Output: {}'.format(np.mean(data_type_data['gt_outputs'])))
logger.info('Mean Output:    {}'.format(np.mean(data_type_data['data'][1])))
logger.info('Mean Top Rank:  {}'.format(np.mean(data_type_data['top_ranks'])))
logger.info('Mean Top Angle: {}'.format(np.mean(data_type_data['top_dists'])))

plotResults('/'.join(last_filename.split('/')[:-2])+'/',
               data_type_data, data_type_title,
               box_plot = False)
#import IPython; IPython.embed()
