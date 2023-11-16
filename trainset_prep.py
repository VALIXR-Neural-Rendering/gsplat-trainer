import os
import numpy as np
import cv2
import math
import glm
import torch
import tqdm
from scipy.spatial.transform import Rotation as R

import pdb

from utils.renderer import parse_args, print_args, GSplatRenderer, read_netCDF


def load_view_list(fname):
    vlist = np.loadtxt(fname).reshape(-1, 4, 4)
    
    def cr2gsplat_transform(V):
        rot = V[:3,:3]
        t = V[:3,3]
        t[1] *= -1
        r = R.from_rotvec(180 * np.array([0, 1, 0]), degrees=True)
        V[:3,:3] = r.as_matrix() @ rot
        V[:3,3] = t.T
        return V
    
    trans_views = []
    for view in vlist:
        trans_views.append(cr2gsplat_transform(view))

    return np.array(trans_views)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)

    view_lst = load_view_list(args.view_path)
    
    pcdata = read_netCDF(args)
    renderer = GSplatRenderer(args, pcdata)

    for i,view in enumerate(tqdm.tqdm(view_lst)):
        rendered_image = renderer.render(view)
        rendered_image = rendered_image.detach().cpu().numpy()
        rendered_image = cv2.flip(rendered_image, 1)

        if not os.path.exists(args.output):
            os.makedirs(args.output)
        cv2.imwrite(os.path.join(args.output, f"{i}.png"), rendered_image)