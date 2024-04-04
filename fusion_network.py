import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from PIL import Image
import os.path as osp
import numpy as np
from scene.dataset_readers import fetchPly

from tqdm import tqdm
from torchmetrics.functional.regression import pearson_corrcoef
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, monodisp
from torch.utils.tensorboard.writer import SummaryWriter
from utils.system_utils import mkdir_p
from utils.sh_utils import RGB2SH, SH2RGB
from plyfile import PlyData, PlyElement
from additional_network import PointNet
from additional_network import ComplexMLP


import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
from typing import Optional
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1.获取图像
# 2.使用clip 将图像转换成embedding 1
def image_embedding(scene_info): 
    model, preprocess = clip.load("ViT-B/32", device=device)   
        # 提取路径
    image_features = torch.zeros(len(scene_info.train_cameras), 512)
    for i in range(len(scene_info.train_cameras)):
        image = Image.open(scene_info.train_cameras[i].image_path)
            # print(image.shape)#(779,520)
        image = preprocess(image).unsqueeze(0).to(device) 
        with torch.no_grad():
            image_features[i] = model.encode_image(image)                      
        
    return image_features.to(device) #(n,512)        
# 3.获取 coarse 点云文件的xyz+color信息
# 4.使用pointnet 将点云信息处理成embedding 2
# 移动到additional_network.py
def point_cloud_embedding(pcd):
    pcd_combined = torch.from_numpy(np.concatenate((pcd.points, pcd.colors), axis=1)).unsqueeze(0)#.double()  
    #print(pcd_combined.shape)#torch.Size([1, 43146, 6])
    model = PointNet().to(device)
    features = model(pcd_combined.float().to(device))
    #print(features.shape)#([1, 43146, 512]) 
    return features
# 5.用e2作为q，e1作为k，v进行cross attention
def cross_attention(image_embedding, pc_embedding):
    """
    image_embedding 是一个 (n, 512) 的向量
    pc_embedding 是一个 (1, 512) 的向量
    用 pc_embedding 作为 q,image_embedding 作为 k 和 v 进行交叉注意力操作
    """
    d_k = pc_embedding.size(-1)  # 获取key的维度大小
    # 计算相似度得分
    # q: (1, 512), k: (n, 512) -> (1, n)
    similarity_scores = torch.matmul(pc_embedding, image_embedding.T) / torch.sqrt(torch.tensor(d_k, dtype=torch.float))
    # 应用 softmax 标准化得分
    attention_weights = F.softmax(similarity_scores, dim=-1)
    # 加权聚合值
    # v: (n, 512), weights: (1, n) -> (1, 512)
    aggregated_value = torch.matmul(attention_weights, image_embedding)
    return aggregated_value
# 6.将attention结果通过mlp预测得到fine gauss parameter
# 也放到了additional_network.py里

def new_render(output, viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    print(pc.active_sh_degree)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)    
    
    means3D = output[:, :3]
    #means2D = output[:, 3:6]  
    #means2D = screenspace_points
    shs = output[:, 3:6]
    opacity = output[:, 6:7]
    scales = output[:, 7:10]
    rotations = output[:, 10:14]
    colors_precomp = None
    cov3D_precomp = None
    new_shs = torch.zeros((shs.shape[0], 9, 3))
    new_shs[:, 0, :] = shs
    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass
    means2D = screenspace_points
    
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D = means3D,    #pc.get_xyz
        means2D = means2D,    #同xyz
        shs = new_shs,            
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "rendered_depth": rendered_depth, # depth
            "rendered_alpha": rendered_alpha, # acc
    }
def construct_list_of_attributes(f_dc, f_rest, scaling, rotation):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]*f_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(f_rest.shape[1]*f_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l
def save_point_cloud(iteration, model_path, output):
    print("save func")
    point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
    #self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"), color=1)
    path = os.path.join(point_cloud_path, "point_cloud.ply")
    mkdir_p(os.path.dirname(path))
    means3D = output[:, :3]
    shs = output[:, 3:6]
    opacity = output[:, 6:7]
    scales = output[:, 7:10]
    rotations = output[:, 10:14]
    
    xyz = means3D.detach().cpu().numpy()
    normals = np.zeros_like(xyz)# 用不到，但是点云应该有
    #此处可能有误，暂时先这样
    f_dc = shs.transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = torch.zeros((shs.shape[0], 8, 3)).transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
   
    opacities = opacity.detach().cpu().numpy()
    scale = scales.detach().cpu().numpy()
    rotation = rotations.detach().cpu().numpy() 

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(f_dc, f_rest, scale, rotation)]
    dtype_full_withcolor = dtype_full[0:6]+[('red', 'u1'), ('green','u1'), ('blue', 'u1')]+dtype_full[6:]
    elements_withcolor = np.empty(xyz.shape[0], dtype=dtype_full_withcolor)
    attributes_withcolor = np.concatenate((xyz, normals, SH2RGB(f_dc)*255.0, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements_withcolor[:] = list(map(tuple, attributes_withcolor))
    el_withcolor = PlyElement.describe(elements_withcolor, 'vertex')
    # path_withcolor = path.replace("input.ply", "input_withcolor.ply")
    PlyData([el_withcolor]).write(path)

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    #tb_writer = prepare_output_and_logger(dataset)
    #感觉guass这块也要自己写一下
    gaussians = GaussianModel(dataset.sh_degree)
    print("Construct Scene...")
    scene = Scene(dataset, gaussians, extra_opts=args)

    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack, augview_stack = None, None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    # 1. clip
    image_features_2d = image_embedding(scene.scene_info) #(n,512) torch.Size([4, 512])
    # print(image_features_2d.shape)
    
    mlp_model = ComplexMLP()
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        # 2.3d feature #这个位置有问题，单独的point net类能否参与backward过程
        pc_feature_3d = point_cloud_embedding(scene.scene_info.point_cloud).squeeze(0)
        #print(pc_feature_3d.shape)#([43146, 512]) 
        
        # 3. cross attention
        attention_result = cross_attention(image_features_2d, pc_feature_3d)
        #print(attention_result.shape)#([43146, 512])
        
        # 4. mlp
        output = mlp_model(attention_result)
        #iter_start.record() # type: ignore
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 提升球谐函数的阶数
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        #在render之前可以打印
        print(viewpoint_cam, viewpoint_cam.mask)
        #问题出在new_render,不使用output作为属性就不出现问题
        render_pkg = new_render(output, viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        #在render之后无法打印
        print(viewpoint_cam, viewpoint_cam.mask)
        # Loss
        # 报错显示在loss函数里面，但原因在于viewpoint_cam出现故障
        loss, Ll1 = cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, tb_writer=None, iteration=iteration)

        loss.backward()
        iter_end.record()  # type: ignore

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_gauss = len(gaussians._xyz)
            if iteration % 10 == 0:
                progress_bar.set_postfix({'Loss': f"{ema_loss_for_log:.{7}f}",  'n': f"{num_gauss}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                #scene.save(iteration)# 此处保存点云，需要修改，不保存内部点云
                save_point_cloud(iteration, args.model_path, )

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/ckpt" + str(iteration) + ".pth")
# 7. 最后使用render.py 进行渲染 ，都不可避免用到render函数。所以考虑直接更新Gaussian Model

def cal_loss(opt, args, image, render_pkg, viewpoint_cam, bg, silhouette_loss_type="bce", mono_loss_type="mid", tb_writer: Optional[SummaryWriter]=None, iteration=0):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    """
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
    #torch.Size([3, 520, 779]) torch.Size([1, 520, 779])
    if opt.random_background:
        #print()        
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()
    Ll1 = l1_loss(image, gt_image)
    Lssim = (1.0 - ssim(image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim
    # if tb_writer is not None:
    #     tb_writer.add_scalar('loss/l1_loss', Ll1, iteration)
    #     tb_writer.add_scalar('loss/ssim_loss', Lssim, iteration)

    if hasattr(args, "use_mask") and args.use_mask:
        if silhouette_loss_type == "bce":
            silhouette_loss = F.binary_cross_entropy(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        elif silhouette_loss_type == "mse":
            silhouette_loss = F.mse_loss(render_pkg["rendered_alpha"], viewpoint_cam.mask)
        else:
            raise NotImplementedError
        loss = loss + opt.lambda_silhouette * silhouette_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/silhouette_loss', silhouette_loss, iteration)

    if hasattr(viewpoint_cam, "mono_depth") and viewpoint_cam.mono_depth is not None:
        if mono_loss_type == "mid":
            # we apply masked monocular loss
            gt_mask = torch.where(viewpoint_cam.mask > 0.5, True, False)
            render_mask = torch.where(render_pkg["rendered_alpha"] > 0.5, True, False)
            mask = torch.logical_and(gt_mask, render_mask)
            if mask.sum() < 10:
                depth_loss = 0.0
            else:
                disp_mono = 1 / viewpoint_cam.mono_depth[mask].clamp(1e-6) # shape: [N]
                disp_render = 1 / render_pkg["rendered_depth"][mask].clamp(1e-6) # shape: [N]
                depth_loss = monodisp(disp_mono, disp_render, 'l1')[-1]
        elif mono_loss_type == "pearson":
            disp_mono = 1 / viewpoint_cam.mono_depth[viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            disp_render = 1 / render_pkg["rendered_depth"][viewpoint_cam.mask > 0.5].clamp(1e-6) # shape: [N]
            depth_loss = (1 - pearson_corrcoef(disp_render, -disp_mono)).mean()
        else:
            raise NotImplementedError

        loss = loss + args.mono_depth_weight * depth_loss
        if tb_writer is not None:
            tb_writer.add_scalar('loss/depth_loss', depth_loss, iteration)

    return loss, Ll1

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000,10_000, 15_000,20_000,25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    ### some exp args
    parser.add_argument("--sparse_view_num", type=int, default=-1, 
                        help="Use sparse view or dense view, if sparse_view_num > 0, use sparse view, \
                        else use dense view. In sparse setting, sparse views will be used as training data, \
                        others will be used as testing data.")
    parser.add_argument("--use_mask", default=True, help="Use masked image, by default True")
    parser.add_argument("--coarse_pcd_name", default='origin', type=str, 
                        help="the init pcd name. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--coarse_pcd_dir", default='origin', type=str, 
                        help="the init pcd dir. 'random' for random, 'origin' for pcd from the whole scene")
    parser.add_argument("--transform_the_world", action="store_true", help="Transform the world to the origin")
    parser.add_argument('--mono_depth_weight', type=float, default=0.0005, help="The rate of monodepth loss")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, 
             args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")