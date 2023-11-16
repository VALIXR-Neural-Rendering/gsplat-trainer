import os
import sys
import argparse
import yaml
import math
import numpy as np
import torch
import laspy
import cv2
import glm
import timeit as tt
from scipy.spatial.transform import Rotation as R
from netCDF4 import Dataset
import pdb

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from utils.transfer_fn import TransferFunction, colorize_particles
EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description="Gaussian renderer for training data collection")
    parser.add_argument("--conf", type=str, help="Configuration file path")
    parser.add_argument("--output", type=str, default="./output", help="Output folder path")
    args = parser.parse_args()
    # pdb.set_trace()

    if args.conf:
        with open(args.conf, 'r') as f:
            config = yaml.safe_load(f)

    parser.set_defaults(**config)
          
    return parser.parse_args()

def print_args(args):
    from huepy import bold, lightblue, orange, lightred, green, red

    args_v = vars(args)
    
    print(bold(lightblue(' - ARGV: ')), '\n', ' '.join(sys.argv), '\n')
    # Get list of default params and changed ones    
    s_default = ''
    for arg in sorted(args_v.keys()):
        value = args_v[arg]
        s_default += f"{red(arg):>25}  :  {orange(value if value != '' else '<empty>')}\n"

    print(f'{s_default[:-1]}\n\n')

def read_netCDF(args):
    data = Dataset(args.nc_path, 'a', format='NETCDF4')
    tf = TransferFunction(args.rmin, args.rmax, args.control_vals, args.opacity)
    colorize_particles(data, args.viz_var, tf)

    return data    

def rotations_frm_vel(args, vel):
    dir = torch.Tensor([0,0,0])
    dir[args.stretch_dim] = 1
    dir = dir.repeat(vel.shape[0], 1)

    a = torch.cross(dir, vel, dim=1)
    w = torch.sqrt((torch.norm(dir, dim=1) ** 2) * torch.norm(vel, dim=1) ** 2) + torch.sum(dir * vel, dim=1)
    q = torch.cat((a, w.unsqueeze(1)), dim=1)
    q = q / torch.norm(q, dim=1).unsqueeze(1)

    return q

class GSplatRenderer:
    def __init__(self, args, pcdata, campos = torch.Tensor([0.0, 0.0, 0.0])) -> None:
        self.args = args
        self.pcdata = pcdata
        self.campos = campos
        self.tanfovx = math.tan((args.fovx / 180.0) * np.pi * 0.5)
        self.tanfovy = math.tan((args.fovy / 180.0) * np.pi * 0.5)

        self.proj = np.array(glm.perspective(np.pi * (args.fovy / 180.0), 
                            args.img_width / args.img_height, 
                            args.near, 
                            args.far)).T
        self.raster_settings = GaussianRasterizationSettings(
                                    image_height=int(args.img_height),
                                    image_width=int(args.img_width),
                                    tanfovx=self.tanfovx,
                                    tanfovy=self.tanfovy,
                                    bg=torch.Tensor(args.bg_color).cuda(),
                                    scale_modifier=args.scale_factor,
                                    viewmatrix=torch.Tensor(np.eye(4)).cuda(),
                                    projmatrix=torch.Tensor(self.proj).cuda(),
                                    sh_degree=0,
                                    campos=torch.Tensor(campos).cuda(),
                                    prefiltered=args.prefiltered,
                                    debug=args.debug
                                )
        self.rasterizer = GaussianRasterizer(self.raster_settings)
        self.init_scene()

    def init_scene(self):
        self.pos = self.pcdata.variables['Position'][:]
        col = self.pcdata.variables['color'][:]
        self.vis_filt = np.expand_dims((col[0,:,3] > 0), axis=0)
        self.xyz = torch.Tensor(self.pos[self.vis_filt])
        self.color = torch.Tensor(np.array(np.vstack((col[self.vis_filt][:,2], col[self.vis_filt][:,1], 
                                                  col[self.vis_filt][:,0])).T, dtype=np.uint8))
        self.opacity = torch.Tensor(np.array(col[self.vis_filt][:,3], dtype=np.float32))
        if (self.opacity.min() == self.opacity.max()):
            self.opacity = torch.ones_like(self.xyz[:,0]) *(self.opacity.min()/255)
        else:
            self.opacity = (self.opacity - self.opacity.min())  / ((self.opacity.max() - self.opacity.min()) + EPS)
        
        if self.args.viz_flow:
            # pdb.set_trace()
            self.vel = self.pcdata.variables[self.args.viz_var[0]][:]
            stretch = torch.Tensor(np.array([1, 1, 1]))
            stretch[self.args.stretch_dim] = self.args.stretch_factor
            self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3))) * stretch
            t_vel = torch.Tensor(self.vel[self.vis_filt])
            t_vel = t_vel / torch.norm(t_vel, dim=1).unsqueeze(1)
            self.rotations = rotations_frm_vel(self.args, t_vel)
        else:
            self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3)))
            self.rotations = torch.Tensor([1,0,0,0]).repeat(self.xyz.shape[0], 1, 1)

    def set_stretch(self, stretch):
        st = torch.Tensor(np.array([1, 1, 1]))
        st[self.args.stretch_dim] = stretch
        self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3))) * st

    def set_splat_size(self, splat_size):
        self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3))) * splat_size

    def render(self, view):
        transl = view[:3,3]
        transl = torch.Tensor(np.expand_dims(transl,1).repeat(self.xyz.shape[0],1).T)
        xyz = self.xyz @ torch.Tensor(view[:3,:3]).T - transl
        
        means2D = torch.zeros_like(self.xyz, dtype=self.xyz.dtype) + 0

        rendered_image, radii = self.rasterizer(
            means3D = xyz.cuda(),
            means2D = means2D.cuda(),
            shs = None,
            colors_precomp = self.color.cuda(),
            opacities = self.opacity.cuda(),
            scales = self.scales.cuda(),
            rotations = self.rotations.cuda(),
            cov3D_precomp = None)
        rendered_image = rendered_image.permute(1, 2, 0) #.detach()
        
        # print("Radii: ", radii.unique())
        return rendered_image

def gaussian_test(args, tanfovx, tanfovy):
    campos = np.array([-10.0, 0.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, -0.05])
    up = up / np.linalg.norm(up)
    view = np.array(glm.lookAt(campos, center, up))
    # view[:3,:3] = view[:3,:3].T
    proj = np.array(glm.perspective(np.pi * (args.fovy / 180.0), args.img_width / args.img_height, 0.01, 1e3)).T

    raster_settings = GaussianRasterizationSettings(
        image_height=int(args.img_height),
        image_width=int(args.img_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.Tensor(args.bg_color).cuda(),
        scale_modifier=args.scale_factor,
        viewmatrix=torch.Tensor(view).cuda(),
        projmatrix=torch.Tensor(proj).cuda(),
        sh_degree=0,
        campos=torch.Tensor(campos).cuda(),
        prefiltered=args.prefiltered,
        debug=args.debug
    )

    rasterizer = GaussianRasterizer(raster_settings)
    xyz = torch.Tensor(np.array([[0,0,0], [1,0,0]]))
    transl = view[:3,3]
    transl = torch.Tensor(np.expand_dims(transl,1).repeat(xyz.shape[0],1).T)
    xyz = xyz + transl
    means2D = torch.zeros_like(xyz, dtype=xyz.dtype) + 0
    opacity = torch.Tensor(np.array([255, 255]))#/255
    colors = torch.Tensor(np.array([[255,0,0], [0,0,255]]))
    scales = torch.Tensor(np.array([[0.1,0.1,0.1], [0.1,0.1,0.1]]))
    # scales = torch.Tensor(np.array([np.diag([1, 1, 1]), np.diag([1, 1, 1])]))
    rotations = torch.Tensor(np.array([[1,0,0,0], [1,0,0,0]]))
    # rotations = torch.Tensor(np.array([[1,0,0,0]]))
    # rotations = torch.Tensor(np.eye(3)).repeat(xyz.shape[0], 1, 1)
    cov3d = torch.Tensor(np.eye(3)).repeat(xyz.shape[0], 1, 1)

    rendered_image, radii = rasterizer(
        means3D = xyz.cuda(),
        means2D = means2D.cuda(),
        shs = None,
        colors_precomp = colors.cuda(),
        opacities = opacity.cuda(),
        scales = scales.cuda(),
        rotations = rotations.cuda(),
        cov3D_precomp = None)
    rendered_image = rendered_image.permute(1, 2, 0).cpu().numpy()
    rendered_image = cv2.flip(rendered_image, 1)
    
    print("Radii: ", radii.unique())

    return rendered_image

if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    view = np.array([
        # 231
        # [-7.071650000000000436e-01, -7.070490000000000386e-01, 1.410000000000000120e-04, 2.648096092000000226e+03],
        # [-3.534379999999999744e-01, 3.536690000000000111e-01, 8.660250000000000448e-01, 4.215108035000000200e+03],
        # [6.123720000000000274e-01, -6.123720000000000274e-01, 5.000000000000000000e-01, -7.530153011300000071e+04],
        # [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]

        # 332
        # [-7.072760000000000158e-01, -7.069370000000000376e-01, 1.410000000000000120e-04, -4.342690446800000063e+04],
        # [-6.090670000000000250e-01, 6.094610000000000305e-01, 5.075380000000000447e-01, 1.062140321000000085e+03],
        # [3.588839999999999808e-01, -3.588839999999999808e-01, 8.616289999999999782e-01, -4.007865391100000124e+04],
        # [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
        
        # 2044
        # [1.000000000000000000e+00, 0.000000000000000000e+00, -0.000000000000000000e+00, -3.861511111099999835e+04],
        # [0.000000000000000000e+00, -9.998479999999999590e-01, 1.745199999999999876e-02, 3.829724960299999657e+04],
        # [-0.000000000000000000e+00, 1.745199999999999876e-02, 9.998479999999999590e-01, -3.467366013199999725e+04],
        # [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]

        # 2052
        [1.000000000000000000e+00, 0.000000000000000000e+00, -0.000000000000000000e+00, -3.861511111099999835e+04],
        [0.000000000000000000e+00, -8.616289999999999782e-01, 5.075380000000000447e-01, 1.625810216699999910e+04],
        [-0.000000000000000000e+00, 5.075380000000000447e-01, 8.616289999999999782e-01, -4.903688558599999669e+04],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]

        # 2449
        # [9.788350000000000106e-01, -2.046530000000000016e-01, 4.100000000000000047e-05, -2.098760739999999885e+02],
        # [-2.012239999999999862e-01, -9.623979999999999757e-01, 1.824800000000000033e-01, 5.793920990000000302e+02],
        # [3.730599999999999888e-02, 1.786260000000000070e-01, 9.832100000000000284e-01, -1.460944776889999921e+05],
        # [0.000000000000000000e+00, 0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
        
        # Test
        # [1, 0, 0, 0],
        # [0, 1, 0, 0],
        # [0, 0, 1, 0],
        # [0, 0, 0, 1]
    ])
    rot = view[:3,:3]
    t = view[:3,3]
    t[1] *= -1
    r = R.from_rotvec(180 * np.array([0, 1, 0]), degrees=True)
    view[:3,:3] = r.as_matrix() @ rot
    view[:3,3] = t.T

    # bbox_min = np.array([-48764.000000000000, -46231.000000000000, 731.00000000000000])
    # bbox_max = np.array([49975.000000000000, 49975.000000000000, 14000.000000000000])
    # cpos = (bbox_max + bbox_min) / 2.0
    # delta = 30000
    # view[:3,3] = bbox_max[2] + delta
    # view[:3,:3] = view[:3,:3].T
    # pdb.set_trace()

    
    # proj = np.array(glm.perspective(np.pi * (args.fovy / 180.0), args.img_width / args.img_height, 1, 1e5)).T
    pcdata = read_netCDF(args)
    renderer = GSplatRenderer(args, pcdata)
    rendered_image = renderer.render(view)
    # rendered_image = gaussian_test(args, tanfovx, tanfovy)
    rendered_image = rendered_image.detach().cpu().numpy()
    rendered_image = cv2.flip(rendered_image, 1)

    # check if output folder exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    cv2.imwrite(os.path.join(args.output, "out.png"), rendered_image)