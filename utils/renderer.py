import os
import sys
import argparse
import yaml
import math
import numpy as np
import torch
import cv2
import glm
from scipy.spatial.transform import Rotation as R
import xarray as xr

from diff_gaussian_rasterization_depth_acc import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.transfer_fn import TransferFunction, colorize_particles

EPS = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gaussian renderer for training data collection"
    )
    parser.add_argument("--conf", type=str, help="Configuration file path")
    parser.add_argument(
        "--output", type=str, default="./output", help="Output folder path"
    )
    args = parser.parse_args()

    if args.conf:
        with open(args.conf, "r") as f:
            config = yaml.safe_load(f)

    parser.set_defaults(**config)

    return parser.parse_args()


def print_args(args):
    from huepy import bold, lightblue, orange, red

    args_v = vars(args)

    print(bold(lightblue(" - ARGV: ")), "\n", " ".join(sys.argv), "\n")
    # Get list of default params and changed ones
    s_default = ""
    for arg in sorted(args_v.keys()):
        value = args_v[arg]
        s_default += f"{red(arg):>25}  :  {orange(value if value != '' else '<empty>')}\n"

    print(f"{s_default[:-1]}\n\n")


def read_netCDF(args):
    data = xr.open_dataset(args.nc_path)
    if args.colorize:
        colmap = args.colmap if hasattr(args, "colmap") else "bwr"
        tf = TransferFunction(
            args.rmin, args.rmax, args.control_vals, args.opacity, colmap
        )
        data = colorize_particles(data, args.viz_var, tf)

    return data


def rotations_frm_vel(sdim, vel):
    dir = torch.Tensor([0, 0, 0])
    dir[sdim] = 1
    dir = dir.repeat(vel.shape[0], 1)

    a = torch.cross(dir, vel, dim=1)
    w = torch.sqrt(
        (torch.norm(dir, dim=1) ** 2) * (torch.norm(vel, dim=1) ** 2)
    ) + torch.sum(dir * vel, dim=1)
    q = torch.cat((w.unsqueeze(1), a), dim=1)
    q = q / torch.norm(q, dim=1).unsqueeze(1)
    # q = torch.zeros_like(q)
    # q[:,0] = 1

    return q


class GSplatRenderer:
    def __init__(
        self, args, pcdata, campos=torch.Tensor([0.0, 0.0, 0.0])
    ) -> None:
        self.args = args
        self.pcdata = pcdata
        self.campos = campos
        self.tanfovx = math.tan((args.fovx / 180.0) * np.pi * 0.5)
        self.tanfovy = math.tan((args.fovy / 180.0) * np.pi * 0.5)

        self.proj = np.array(
            glm.perspective(
                np.pi * (args.fovy / 180.0),
                args.img_width / args.img_height,
                args.near,
                args.far,
            )
        ).T
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
            debug=args.debug,
        )
        self.rasterizer = GaussianRasterizer(self.raster_settings)
        self.init_scene()

    def init_scene(self):
        self.pos = np.array(self.pcdata.variables["Position"][:])
        col = np.array(self.pcdata.variables["color"][:])
        self.vis_filt = np.expand_dims((col[0, :, 3] > 0), axis=0)
        self.xyz = torch.Tensor(self.pos[self.vis_filt])
        print("XYZ: ", self.xyz.shape)
        self.color = torch.Tensor(
            np.array(
                np.vstack(
                    (
                        col[self.vis_filt][:, 2],
                        col[self.vis_filt][:, 1],
                        col[self.vis_filt][:, 0],
                    )
                ).T,
                dtype=np.uint8,
            )
        )
        self.opacity = torch.Tensor(
            np.array(col[self.vis_filt][:, 3], dtype=np.float32)
        )
        if self.opacity.min() == self.opacity.max():
            self.opacity = torch.ones_like(self.xyz[:, 0]) * (
                self.opacity.min() / 255
            )
        else:
            self.opacity = (self.opacity - self.opacity.min()) / (
                (self.opacity.max() - self.opacity.min()) + EPS
            )
        self.opacity = torch.ones_like(
            self.xyz[:, 0]
        )  # switch off to enable transparency

        if self.args.viz_flow:
            self.vel = np.array(self.pcdata.variables["vel"][:])
            self.vel = self.vel[self.vis_filt]
            stretch = torch.Tensor(np.array([1, 1, 1]))
            stretch[self.args.stretch_dim] = self.args.stretch_factor
            self.scales = (
                torch.Tensor(np.ones((self.xyz.shape[0], 3))) * stretch
            )
        else:
            if args.enable_nn:
                sfactor = torch.Tensor(
                    np.array(self.pcdata.variables["ssize"][:])[self.vis_filt]
                )
                self.scales = sfactor.unsqueeze(1).repeat(1, 3)
            else:
                self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3)))

            self.rotations = torch.Tensor([1, 0, 0, 0]).repeat(
                self.xyz.shape[0], 1, 1
            )

    def set_stretch(self, stretch):
        st = torch.Tensor(np.array([1, 1, 1]))
        st[self.args.stretch_dim] = stretch
        self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3))) * st

    def set_splat_size(self, splat_size):
        st = torch.Tensor(np.array([1, 1, 1])) * splat_size
        self.scales = torch.Tensor(np.ones((self.xyz.shape[0], 3))) * st

    def render(self, view, norm_depth=True):
        transl = view[:3, 3]
        transl = torch.Tensor(
            np.expand_dims(transl, 1).repeat(self.xyz.shape[0], 1).T
        )
        xyz = self.xyz @ torch.Tensor(view[:3, :3]).T - transl

        if self.args.viz_flow:
            t_vel = torch.Tensor(self.vel)
            t_vel = t_vel / torch.norm(t_vel, dim=1).unsqueeze(1)
            t_vel = t_vel @ torch.Tensor(view[:3, :3]).T
            self.rotations = rotations_frm_vel(self.args.stretch_dim, t_vel)
        else:
            self.rotations = torch.Tensor([1, 0, 0, 0]).repeat(
                self.xyz.shape[0], 1, 1
            )

        means2D = torch.zeros_like(self.xyz, dtype=self.xyz.dtype) + 0

        rendered_image, depth_image, _, radii = self.rasterizer(
            means3D=xyz.cuda(),
            means2D=means2D.cuda(),
            shs=None,
            colors_precomp=self.color.cuda(),
            opacities=self.opacity.cuda(),
            scales=self.scales.cuda(),
            rotations=self.rotations.cuda(),
            cov3D_precomp=None,
        )
        rendered_image = rendered_image.permute(1, 2, 0)  # .detach()
        if norm_depth:
            depth_image = (depth_image - depth_image.min()) / (
                depth_image.max() - depth_image.min() + EPS
            )

        # print("Radii: ", radii.unique())
        return rendered_image, depth_image


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    view = np.array(
        [
            # Test Hurricane
            [
                -9.448362718362994794e-01,
                -3.275429973776049497e-01,
                6.550859947552099198e-05,
                1.184776350098319199e03,
            ],
            [
                -3.253671295722339596e-01,
                9.385827025785792310e-01,
                1.148857755086133414e-01,
                -2.406208201779802039e03,
            ],
            [
                3.769151650447973012e-02,
                -1.085269334736066321e-01,
                9.933787063826201580e-01,
                -1.110081439929906919e05,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                1.000000000000000000e00,
            ],
            # Test Storms
            # [
            #     -7.112930227966427488e-01,
            #     -7.028955939243937134e-01,
            #     1.405791187848787400e-04,
            #     7.578819041226486419e00,
            # ],
            # [
            #     -6.864222811510963806e-01,
            #     6.946659228737845915e-01,
            #     2.150899986918001217e-01,
            #     -7.114502766173899317e01,
            # ],
            # [
            #     1.512834679009573691e-01,
            #     -1.528955187034178875e-01,
            #     9.765942211073506130e-01,
            #     -6.527772216711432520e02,
            # ],
            # [
            #     0.000000000000000000e00,
            #     0.000000000000000000e00,
            #     0.000000000000000000e00,
            #     1.000000000000000000e00,
            # ],
            # Test Morro Bay
            # [
            #     3.522526411568872584e-01,
            #     9.332171240870483775e-01,
            #     7.087930663245675666e-02,
            #     -1.760935758250705021e03,
            # ],
            # [
            #     6.274480122978506325e-01,
            #     -2.916714169117141653e-01,
            #     7.219672959491891806e-01,
            #     -4.667233775880207531e02,
            # ],
            # [
            #     -6.944257114058153268e-01,
            #     2.098418067674148735e-01,
            #     6.882872565078574922e-01,
            #     4.135048950057085904e02,
            # ],
            # [
            #     0.000000000000000000e00,
            #     0.000000000000000000e00,
            #     0.000000000000000000e00,
            #     1.000000000000000000e00,
            # ],
        ]
    )  # .T
    rot = view[:3, :3]
    t = view[:3, 3]
    t[1] *= -1
    r = R.from_rotvec(180 * np.array([0, 1, 0]), degrees=True)
    view[:3, :3] = r.as_matrix() @ rot
    view[:3, 3] = t.T

    pcdata = read_netCDF(args)
    renderer = GSplatRenderer(args, pcdata)
    rendered_image, depth_image = renderer.render(view)
    rendered_image = rendered_image.detach().cpu().numpy()
    depth_image = depth_image.detach().cpu().numpy()

    # rendered_image = cv2.flip(rendered_image, 1)
    # check if output folder exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cv2.imwrite(os.path.join(args.output, "out.png"), rendered_image)
    cv2.imwrite(os.path.join(args.output, "depth.png"), depth_image * 255)
