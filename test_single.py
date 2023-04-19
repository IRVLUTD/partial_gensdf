#!/usr/bin/env python3

import torch
import torch.utils.data 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np 
import trimesh


import os
import json
import time
from tqdm import tqdm

# remember to add paths in model/__init__.py for new models
from model import *

from renderer.online_object_renderer import OnlineObjectRenderer
from utils import utils


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



def visualization(pc, pc2, color, depth):

    # visualization for your debugging
    import matplotlib.pyplot as plt
    fig = plt.figure()
        
    # show RGB image
    ax = fig.add_subplot(1, 3, 1)
    if color is not None:
        plt.imshow(color[:, :, (2, 1, 0)])
    ax.set_title('RGB image')
        
    # show depth image
    ax = fig.add_subplot(1, 3, 2)
    if depth is not None:
        plt.imshow(depth)
    ax.set_title('depth image')
        
    # up to now, suppose you get the points box as pbox. Its shape should be (5280, 3)
    # then you can use the following code to visualize the points in pbox
    # You shall see the figure in the homework assignment
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], marker='.', color='r')
    # ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], marker='.', color='g')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D ploud cloud')
    set_axes_equal(ax)
                  
    plt.show()



def main():
    
    model = init_model(specs["Model"], specs, 1)
    
    ckpt = "{}.ckpt".format(args.resume) if args.resume=='last' else "epoch={}.ckpt".format(args.resume)
    resume = os.path.join(args.exp_dir, ckpt)
     
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)

    file_ext = args.file[-4:]
    pc_size = 5000
    if file_ext == ".csv":
        f = pd.read_csv(args.file, sep=',',header=None).values
        f = f[f[:,-1]==0][:,:3]
        
        # set up renderer
        renderer = OnlineObjectRenderer(caching=True)
        all_poses = utils.uniform_quaternions()
    
        cad_path = args.file.replace('sdf_data.csv', 'model.obj')
        cad_scale = 1.0
    
        viewing_index = np.random.randint(0, high=len(all_poses))
        camera_pose = all_poses[viewing_index]
    
        # read centroid
        filename = args.file.replace('sdf_data', 'centroid')
        centroid = pd.read_csv(filename, sep=',',header=None).values
    
        color, depth, pc, transferred_pose = renderer.change_and_render(cad_path, cad_scale, centroid, camera_pose)
        pc = pc.dot(utils.inverse_transform(transferred_pose).T)[:, :3]
        
        if pc.shape[0] < pc_size:
            pc_idx = np.random.choice(pc.shape[0], pc_size)
        else:
            pc_idx = np.random.choice(pc.shape[0], pc_size, replace=False)
        pc = pc[pc_idx]        
        
    elif file_ext == ".ply":
        f = trimesh.load(args.file).vertices
    else:
        print("add your extension type here! currently not supported...")
        exit()
        
    # visualization
    print(pc.shape, f.shape)
    visualization(pc[::5], f[::20], color, depth)
    
    # assign poinsts
    f = pc.copy()

    sampled_points = 15000 # load more points for more accurate reconstruction 
    
    # recenter and normalize
    f -= np.mean(f, axis=0)
    bbox_length = np.sqrt( np.sum((np.max(f, axis=0) - np.min(f, axis=0))**2) )
    f /= bbox_length

    f = torch.from_numpy(f)[torch.randperm(f.shape[0])[0:sampled_points]].float().unsqueeze(0)
    model.load_state_dict(checkpoint['state_dict'])
    model.reconstruct(model, {'point_cloud':f, 'mesh_name':"loaded_file"}, eval_dir="single_recon", testopt=True, sampled_points=sampled_points) 


def init_model(model, specs, num_objects):
    if model == "DeepSDF":
        return DeepSDF(specs, num_objects).cuda()
    elif model == "NeuralPull":
        return NeuralPull(specs, num_objects).cuda()
    elif model == "ConvOccNet":
        return ConvOccNet(specs).cuda()
    elif model == "GenSDF":
        return GenSDF(specs, None).cuda()
    else:
        print("model not loaded...")

    
if __name__ == "__main__":

    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--exp_dir", "-e",
        default="config/gensdf/semi",
        help="This directory should include experiment specifications in 'specs.json,' and logging will be done in this directory as well.",
    )
    arg_parser.add_argument(
        "--resume", "-r",
        default="last",
        help="continue from previous saved logs, integer value or 'last'",
    )

    arg_parser.add_argument(
        "--outdir", "-o",
        required=True,
        help="output directory of reconstruction",
    )

    arg_parser.add_argument(
        "--file", "-f",
        required=True,
        help="input point cloud filepath, in csv or ply format",
    )


    args = arg_parser.parse_args()
    specs = json.load(open(os.path.join(args.exp_dir, "specs.json")))
    print(specs["Description"][0])

    main()
