import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import matplotlib.tri as mtri
from generic_pose.utils.pose_processing import quatAngularDiffBatch

from spherical_kde import SphericalKDE, decra_from_polar
import cartopy

def cart2Sph(pts):
    r = np.linalg.norm(pts, axis=1)   
    theta = np.arccos(pts[:,2]/r)
    phi = np.arctan2(pts[:,1], pts[:,0])
    return phi, theta, r

def plotQuatKDE(quats, dists = None, gt_quat=None, img_prefix = '/home/bokorn/results/test/', num_layers=0, 
                bandwidth = None):
    if(dists is None):
        dists = np.ones(np.size(quats,0))
    angle = 2*np.arccos(np.abs(quats[:,3]))
    num_rows = 2
    if(gt_quat is not None):
        num_rows = 3
        gt_p = gt_quat[:3].copy()
        if(gt_quat[3] < 0):
            gt_p *= -1.0
        gt_angle = 2*np.arccos(np.abs(gt_quat[3]))
        gt_r = np.linalg.norm(gt_p)
        gt_phi = np.arctan2(gt_p[1], gt_p[0])
        gt_theta = np.arccos(gt_p[2]/gt_r)
        gt_ra, gt_dec = decra_from_polar(gt_phi, gt_theta)
        view_phi = gt_phi*180/np.pi
        view_theta = 90 - gt_theta*180/np.pi

    q_pts = quats[:,:3].copy() * (1.0-2.0*(quats[:,3:] < 0))
    r = np.expand_dims(np.linalg.norm(q_pts,axis=1), 1)
    r[r==0]=1.0
    
    fig = plt.figure()
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    angle_range = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]

    projections = [cartopy.crs.Orthographic(0,0), 
                   cartopy.crs.Orthographic(180, 0)]

    if(gt_quat is not None):
        projections.append(cartopy.crs.Orthographic(view_phi, view_theta))
    
    #for k in range(num_rows):
    #    ax = fig.add_subplot(num_rows,5, num_rows*k+1)
    #    ax.grid(False)
    #    plt.axis('off')

    for k in range(4):
        mask = np.bitwise_and(angle >= angle_range[k], angle < angle_range[k+1]).flatten()
        phi, theta, r = cart2Sph(q_pts[mask])
        weights = dists[mask]

        for j, proj in enumerate(projections):
            ax = fig.add_subplot(num_rows,4,4*j+k+1, projection=proj)
            kde = SphericalKDE(phi, theta, weights=weights, bandwidth=bandwidth)
            kde.plot(ax)
            #kde.plot_samples(ax)
            if(gt_quat is not None and gt_angle >= angle_range[k] and gt_angle < angle_range[k+1]):
                ax.plot([gt_ra, gt_ra], [gt_dec, gt_dec], 'kx', transform=cartopy.crs.PlateCarree())
    
    #angle_names = ['0', '$\frac{\pi}{4}$',  '$\frac{\pi}{2}$', '$\frac{3\pi}{4}$', '$\pi$']
    #cols = [r'{} - {}'.format(angle_names[j], angle_names[j+1]) for j in range(4)]
    cols = ['{:.0f} - {:.0f}'.format(angle_range[j]*180/np.pi, angle_range[j+1]*180/np.pi) for j in range(4)]
    rows = ['view(0,0)', 'view(0, 180)']
    if(gt_quat is not None):
        rows.append('view({:.0f},{:.0f})'.format(view_phi, view_theta))
    for ax, col in zip(fig.axes[::num_rows], cols):
        ax.set_title(col)
    for ax, row in zip(fig.axes[:num_rows], rows):
        ax.set_title(row, fontdict = {'verticalalignment': 'center'}, loc='left')

    plt.tight_layout()
    plt.savefig(img_prefix + 'kde.png', dpi=500)
    
    plt.gcf().clear()


     
def plotQuatBall(quats, dists=None, gt_quat=None, img_prefix = '/home/bokorn/results/test/', num_layers=0, 
                 point_size = 1, view_theta = 0):
    if(dists is None):
        dists = np.ones(np.size(quats,0))
    th = 2*np.arccos(np.abs(quats[:,3]))
    num_rows = 2
    if(gt_quat is not None):
        num_rows = 3
        gt_p = gt_quat[:3].copy()
        if(gt_quat[3] < 0):
            gt_p *= -1.0
        gt_th = 2*np.arccos(np.abs(gt_quat[3]))
        view_r = np.linalg.norm(gt_p)
        view_phi = np.arctan2(gt_p[1], gt_p[0])*180/np.pi
        view_theta = 90 - np.arccos(gt_p[2]/view_r)*180/np.pi

    q_pts = quats[:,:3].copy() * (1.0-2.0*(quats[:,3:] < 0))
    r = np.expand_dims(np.linalg.norm(q_pts,axis=1), 1)
    r[r==0]=1.0
    #q_pts *= th/r
    
    #fig, axes = plt.subplots(num_rows+1, 4,gridspec_kw = {'height_ratios':[1, 10, 10]})

    fig = plt.figure()
    fig.subplots_adjust(top=0.99, bottom=0.01, left=0.01, right=0.99)
    
    th_range = [0, 0.25*np.pi, 0.5*np.pi, 0.75*np.pi, np.pi]

    plt.set_cmap('jet')
    cmap = matplotlib.cm.get_cmap()
    c = cmap(dists)
    
    for k in range(4):
        mask = np.bitwise_and(th >= th_range[k], th < th_range[k+1]).flatten()
        
        mask_pos = np.bitwise_and(mask, q_pts[:,0] >= 0).flatten()
        mask_neg = np.bitwise_and(mask, q_pts[:,0] < 0).flatten()
        ax = fig.add_subplot(num_rows,4,k+1, projection='3d')
        ax.view_init(0, 0)

        #cmap = matplotlib.cm.get_cmap('Spectral')
        #c = cmap(dists)
        #c[:,3] = dists/np.max(dists)
        #sc = ax.scatter(q_pts[:,0],q_pts[:,1], q_pts[:,2], s=1, c=c, marker='.')
        #sc = ax.scatter(q_pts[:,0],q_pts[:,1], q_pts[:,2], s=1, c=dists, marker='.', alpha=dists/np.max(dists))
        #import IPython; IPython.embed()
        sc = ax.scatter(q_pts[mask_pos,0], q_pts[mask_pos,1], q_pts[mask_pos,2], 
                        s=point_size, c=c[mask_pos], marker='.', alpha=0.25)
        if(gt_quat is not None and gt_th >= th_range[k] and gt_th < th_range[k+1] and gt_p[2] >= 0):
            ax.scatter(gt_p[0], gt_p[1], gt_p[2], s=10, c='k', marker='x')    
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('tight')
        plt.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_aspect('equal')

        ax = fig.add_subplot(num_rows,4,4+k+1, projection='3d')
        
        ax.view_init(0, 180)
        sc = ax.scatter(q_pts[mask_neg,0], q_pts[mask_neg,1], q_pts[mask_neg,2], 
                        s=point_size, c=c[mask_neg], marker='.', alpha=0.25)
        if(gt_quat is not None and gt_th >= th_range[k] and gt_th < th_range[k+1] and gt_p[2] < 0):
            ax.scatter(gt_p[0], gt_p[1], gt_p[2], s=10, c='k', marker='x')    
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.axis('tight')
        plt.axis('off')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_aspect('equal')
       
        if(gt_quat is not None):
            gt_dists = quatAngularDiffBatch(np.tile(gt_quat, (np.size(quats,0),1)), quats)
            mask_gt = mask #np.bitwise_and(mask, gt_dists < np.pi/2).flatten()
            ax = fig.add_subplot(3,4,8+k+1, projection='3d')
            
            sc = ax.scatter(q_pts[mask_gt,0], q_pts[mask_gt,1], q_pts[mask_gt,2], 
                            s=point_size, c=c[mask_gt], marker='.', alpha=0.25)
            if(gt_th >= th_range[k] and gt_th < th_range[k+1]):
                ax.scatter(gt_p[0], gt_p[1], gt_p[2], s=10, c='k', marker='x')    

            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            plt.axis('tight')
            plt.axis('off')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_aspect('equal')
            ax.view_init(view_theta, view_phi)
    #angle_names = ['0', '$\frac{\pi}{4}$',  '$\frac{\pi}{2}$', '$\frac{3\pi}{4}$', '$\pi$']
    #cols = [r'{} - {}'.format(angle_names[j], angle_names[j+1]) for j in range(4)]
    cols = ['{:.0f} - {:.0f}'.format(th_range[j]*180/np.pi, th_range[j+1]*180/np.pi) for j in range(4)]
    rows = ['view(0,0)', 'view(0, 180)']
    if(gt_quat is not None):
        rows.append('view({:.0f},{:.0f})'.format(view_phi, view_theta))
    for ax, col in zip(fig.axes[::num_rows], cols):
        ax.set_title(col)
    for ax, row in zip(fig.axes[:num_rows], rows):
        ax.set_title(row, rotation=0, size='large', loc='left')
        #ax.set_ylabel(row, rotation=0, size='large')

    plt.tight_layout()
    cax, _ = matplotlib.colorbar.make_axes(fig.axes)
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap)
    #cax, _ = matplotlib.colorbar.make_axes(fig.axes, location='bottom')
    #cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, orientation='horizontal')

    plt.savefig(img_prefix + 'quats.png', dpi=500)
    
    plt.gcf().clear()


        
