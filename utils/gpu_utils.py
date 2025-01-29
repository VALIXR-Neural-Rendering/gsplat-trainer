from contextlib import contextmanager
import torch
from glumpy import gloo

try:
    import pycuda.driver
    from pycuda.gl import graphics_map_flags

    _PYCUDA = True
except ImportError as err:
    print("pycuda import error:", err)
    _PYCUDA = False


@contextmanager
def cuda_activate_array(img):
    """Context manager simplifying use of pycuda.gl.RegisteredImage"""
    mapping = img.map()
    yield mapping.array(0, 0)
    mapping.unmap()


def create_shared_texture(arr, map_flags=None):
    """Create and return a Texture2D with gloo and pycuda views."""

    if map_flags is None:
        map_flags = graphics_map_flags.WRITE_DISCARD

    gl_view = arr.view(gloo.Texture2D)
    gl_view.activate()  # force gloo to create on GPU
    gl_view.deactivate()

    cuda_view = pycuda.gl.RegisteredImage(
        int(gl_view.handle), gl_view.target, map_flags
    )

    return gl_view, cuda_view


def cpy_tensor_to_texture(tensor, texture, size):
    """Copy pytorch tensor to GL texture (cuda view)"""
    with cuda_activate_array(texture) as ary:
        cpy = pycuda.driver.Memcpy2D()

        cpy.set_src_device(tensor.data_ptr())
        cpy.set_dst_array(ary)
        cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = size
        cpy.height = tensor.shape[0]
        cpy(aligned=False)

        torch.cuda.synchronize()

    return tensor
