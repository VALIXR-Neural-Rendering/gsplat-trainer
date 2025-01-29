import os
import numpy as np
import cv2
import tqdm
from scipy.spatial.transform import Rotation as R

from utils.renderer import parse_args, print_args, GSplatRenderer, read_netCDF


def load_view_list(fname):
    vlist = np.loadtxt(fname).reshape(-1, 4, 4)

    def cr2gsplat_transform(V):
        rot = V[:3, :3]
        t = V[:3, 3]
        t[1] *= -1
        r = R.from_rotvec(180 * np.array([0, 1, 0]), degrees=True)
        V[:3, :3] = r.as_matrix() @ rot
        V[:3, 3] = t.T
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
    tstats = 0

    for i, view in enumerate(tqdm.tqdm(view_lst)):
        rendered_image, depth_image = renderer.render(view)
        rendered_image = rendered_image.detach().cpu().numpy()
        depth_image = depth_image.detach().cpu().numpy()
        # rendered_image = cv2.resize(rendered_image, (512, 512))      # Enable
        # for storms dataset only
        rendered_image = cv2.flip(rendered_image, 1)
        # depth_image = cv2.resize(depth_image, (512, 512))      # Enable for
        # storms dataset only
        depth_image = cv2.flip(depth_image, 1)

        if not os.path.exists(args.output):
            os.makedirs(args.output)
        if not os.path.exists(os.path.join(args.output, "depth")):
            os.makedirs(os.path.join(args.output, "depth"))
        cv2.imwrite(os.path.join(args.output, f"{i}.png"), rendered_image)
        cv2.imwrite(
            os.path.join(
                args.output,
                "depth",
                f"{i}.png"),
            depth_image *
            255)
