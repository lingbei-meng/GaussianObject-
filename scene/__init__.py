#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors
from typing import NamedTuple
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
    
def fps(points, n_centers):
    """
    Farthest Point Sampling (FPS) algorithm to select n center points from a point cloud.
    """
    # Initialize containers
    centers = np.zeros((n_centers, points.shape[1]))
    distances = np.full(points.shape[0], np.inf)
    
    # Randomly select the first center
    first_center = np.random.randint(0, points.shape[0])
    centers[0] = points[first_center]
    
    for i in range(1, n_centers):
        # Calculate distances from the latest added center and update the minimum distances
        new_distances = np.linalg.norm(points - centers[i - 1], axis=1)
        distances = np.minimum(distances, new_distances)
        
        # Select the point with the maximum distance as the next center
        next_center_index = np.argmax(distances)
        centers[i] = points[next_center_index]
        
    return centers

def patch_mask(point_cloud, n_centers, n_delete_rate):
    """
    Function to create patch masks, delete a number of patches, and return a new point cloud.
    """
    # Step 1: Select n center points using FPS
    centers = fps(point_cloud.points, n_centers)
    
    # Step 2: Use KNN to divide the point cloud into patches based on the n centers
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(centers)
    _, indices = nbrs.kneighbors(point_cloud.points)
    
    # Step 3: Delete a certain number of patches randomly
    patches_to_delete = np.random.choice(n_centers, int(n_delete_rate*n_centers), replace=False)
    mask = np.isin(indices.flatten(), patches_to_delete, invert=True)
    
    # Create a new point cloud without the deleted patches
    new_points = point_cloud.points[mask]
    new_colors = point_cloud.colors[mask]
    new_normals = point_cloud.normals[mask]
    
    #return new_points, new_colors, new_normals
    return BasicPointCloud(new_points, new_colors, new_normals)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], extra_opts=None, load_ply=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path # type: ignore
        # print(self.model_path)
        # exit()
        self.loaded_iter = None
        self.gaussians = gaussians
        # print(gaussians)
        # exit()
        
        if load_iteration and load_ply is None:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        self.render_cameras = {}

        # print(args.source_path, args.images, args.eval, extra_opts)  # /home/pkudba/MaskGaussian/data/mip360/kitchen images False
        # exit()
        if os.path.exists(os.path.join(args.source_path, "sparse")): # type: ignore
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "transforms_alignz_train.json")): # type: ignore
            print("Found transforms_alignz_train.json file, assuming OpenIllumination data set!")
            scene_info = sceneLoadTypeCallbacks["OpenIllumination"](args.source_path, args.white_background, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")): # type: ignore
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, extra_opts=extra_opts) # type: ignore
        elif os.path.exists(os.path.join(args.source_path, "hydrant", "frame_annotations.jgz")): # type: ignore
            scene_info = sceneLoadTypeCallbacks["CO3D"](args.source_path, extra_opts=extra_opts) # type: ignore
        else:
            assert False, "Could not recognize scene type!"

        # print("stop")
        # exit()
        if not self.loaded_iter and load_ply is None:
            # NOTE :this dump use the file name, we dump the SceneInfo.pcd as the input.ply
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            if scene_info.render_cameras:
                camlist.extend(scene_info.render_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            init_time = time.time()
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, mode="train")
            init_time2 = time.time()
            print("Loading training cameras with {}s".format(init_time2 - init_time))
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
            init_time3 = time.time()
            print("Loading test cameras with {}s".format(time.time() - init_time2))
            self.render_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.render_cameras, resolution_scale, args)
            print("Loading render cameras with {}s".format(time.time() - init_time3))

        # print("*******************")
        # print(self.model_path)
        # exit()
        # print(self.loaded_iter, load_ply)
        # exit()
        # exit()
        if self.loaded_iter:
            load_name = "point_cloud.ply"
            # print(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), load_name))
            # exit()
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), load_name))
        elif load_ply:
            self.gaussians.load_ply(load_ply)
            # in this case, we need it to be trainable, so we need to make sure the spatial_lr_scale is not 0
            self.gaussians.spatial_lr_scale = self.cameras_extent
        else:
            # print(scene_info.point_cloud)
            # exit()
            # 1 .# fine guassian 输入时对点云进行patchy mask
            # if extra_opts.coarse_pcd_dir:
            #     print(extra_opts.coarse_pcd_dir)
            #     print(scene_info.point_cloud.points.shape)
            #     #scene_info.point_cloud = patch_mask(scene_info.point_cloud, 100, 0.2)
            #     new_point_cloud= patch_mask(scene_info.point_cloud, 100, 0.2)  
            #     print(new_point_cloud.point_cloud.points.shape)          
            # self.gaussians.create_from_pcd(new_point_cloud, self.cameras_extent)
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.gaussians.save_ply(os.path.join(self.model_path, "input.ply"), color=1)

    def save(self, iteration):
        print("save func")
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), color=1)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getAllCameras(self, scale=1.0):
        return self.train_cameras[scale] + self.test_cameras[scale]

    def getRenderCameras(self, scale=1.0):
        return self.render_cameras[scale]
