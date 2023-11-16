import sys
from contextlib import contextmanager
import numpy as np
import torch
import math
import glm
import timeit as tt

from torch import Tensor, ByteTensor
import torch.nn.functional as F
from torch.autograd import Variable

import pycuda.driver
from pycuda.gl import graphics_map_flags

from glumpy import app, gloo, gl

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QApplication, QSlider, QLabel

from utils.renderer import parse_args, print_args, GSplatRenderer, read_netCDF
from utils.gpu_utils import create_shared_texture, cpy_tensor_to_texture
from utils.trackball import Trackball

import pdb


args = parse_args()

# create window with OpenGL context
app.use('qt5')

class GLWindow:
    def __init__(self, args, renderer, tball):
        self.args = args
        self.glumpy_window = app.Window(args.img_width, args.img_height, fullscreen=False)
        self.glumpy_window.set_title("Test")

        self.renderer = renderer
        self.tball = tball

        self.glumpy_window.push_handlers(on_init=self.on_init)
        self.glumpy_window.push_handlers(on_draw=self.on_draw)
        self.glumpy_window.push_handlers(on_mouse_press=self.on_mouse_press)
        self.glumpy_window.push_handlers(on_mouse_drag=self.on_mouse_drag)
        self.glumpy_window.push_handlers(on_mouse_scroll=self.on_mouse_scroll)

    def set_position(self, x, y):
        self.glumpy_window.set_position(x,y)

    def on_init(self):
        pass

    def on_draw(self, dt):
        global state
        tex = screen['tex']
        h,w = tex.shape[:2]
        view = tball.pose
        
        state = self.renderer.render(view)
        tensor = torch.cat((state/255, torch.ones_like(state[:,:,:1])), dim=2)
        tensor = (255*tensor).byte().contiguous() # convert to ByteTensor
        # copy from torch into buffer
        # pdb.set_trace()
        assert tex.nbytes == tensor.numel()*tensor.element_size()
        
        cpy_tensor_to_texture(tensor, cuda_buffer, tex.nbytes//h)
        # draw to screen
        self.glumpy_window.clear()
        screen.draw(gl.GL_TRIANGLE_STRIP)

    def on_mouse_press(self, x, y, button):
        self.tball.down((x, y))

    def on_mouse_drag(self, x, y, dx, dy, button):
        self.tball.drag((x+dx, y+dy))

    def on_mouse_scroll(self, x, y, dx, dy):
        self.tball.scroll(dy*40)

    @property
    def _native_window(self):
        return self.glumpy_window._native_window

class Window(QtWidgets.QMainWindow):
    def __init__(self, args, window):
        super().__init__()
        self.args = args
        self.window = window
        
        self.resize(args.img_width + 200, args.img_height)
        self.move(0,0)

    def create_layout(self):
        self.central_widget = QtWidgets.QWidget(self)
        self.horizontal_layout = QtWidgets.QHBoxLayout(self.central_widget)
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.horizontal_layout.addLayout(self.vertical_layout, stretch=0)
        self.setCentralWidget(self.central_widget)

        self.window.set_position(0,0)
        self.horizontal_layout.addWidget(self.window._native_window, stretch=1)

        # Stretch control
        self.stretch_label = QLabel("Stretch")
        self.stretch_label.setAlignment(QtCore.Qt.AlignCenter)
        self.vertical_layout.addWidget(self.stretch_label, stretch=0)

        self.stretch_sl = QSlider(QtCore.Qt.Horizontal)
        self.stretch_sl.setMinimumWidth(200)
        self.stretch_sl.setMinimum(0)
        self.stretch_sl.setMaximum(1000)
        self.stretch_sl.setValue(args.stretch_factor)
        self.stretch_sl.setTickPosition(QSlider.TicksBelow)
        self.stretch_sl.setTickInterval(100)
        self.stretch_sl.valueChanged.connect(self.stretch_value_change)
        self.vertical_layout.addWidget(self.stretch_sl, stretch=0)

        # Splat size control
        self.ssize_label = QLabel("Splat Size")
        self.ssize_label.setAlignment(QtCore.Qt.AlignCenter)
        self.vertical_layout.addWidget(self.ssize_label, stretch=0)

        self.ssize_sl = QSlider(QtCore.Qt.Horizontal)
        self.ssize_sl.setMinimum(0)
        self.ssize_sl.setMaximum(500)
        self.ssize_sl.setValue(args.scale_factor)
        self.ssize_sl.setTickPosition(QSlider.TicksBelow)
        self.ssize_sl.setTickInterval(50)
        self.ssize_sl.valueChanged.connect(self.ssize_value_change)
        self.vertical_layout.addWidget(self.ssize_sl, stretch=0)

    def stretch_value_change(self):
        stretch = self.stretch_sl.value()
        gaussr.set_stretch(stretch)

    def ssize_value_change(self):
        ssize = self.ssize_sl.value()
        gaussr.set_splat_size(ssize)

def setup():
    global screen, cuda_buffer, state
    # setup pycuda and torch
    import pycuda.gl.autoinit
    import pycuda.gl
    assert torch.cuda.is_available()
    print('using GPU {}'.format(torch.cuda.current_device()))
    # torch.nn layers expect batch_size, channels, height, width
    state = torch.cuda.FloatTensor(1,3,args.img_height,args.img_width)
    # create a buffer with pycuda and gloo views
    # tex, cuda_buffer = create_shared_texture(args.img_width, args.img_height, 4)
    tex, cuda_buffer = create_shared_texture(
                np.zeros((args.img_height, args.img_width, 4), np.uint8)
            )
    # create a shader to program to draw to the screen
    vertex = """
    uniform float scale;
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        v_texcoord = texcoord;
        gl_Position = vec4(scale*position, 0.0, 1.0);
    } """
    fragment = """
    uniform sampler2D tex;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(tex, v_texcoord);
    } """
    # Build the program and corresponding buffers (with 4 vertices)
    screen = gloo.Program(vertex, fragment, count=4)
    # Upload data into GPU
    screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
    screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
    screen['scale'] = 1.0
    screen['tex'] = tex


if __name__=='__main__':
    print_args(args)
    
    pcdata = read_netCDF(args)
    gaussr = GSplatRenderer(args, pcdata)
    init_view = np.loadtxt(args.view_path).reshape(-1, 4, 4)[-1]
    tball = Trackball(init_view,(args.img_width,args.img_height),1)
    window = GLWindow(args, gaussr, tball)

    appl = QApplication([])
    main_window = Window(args, window)
    main_window.create_layout()
    main_window.show()
    
    setup()
    app.run()
    pycuda.gl.autoinit.context.pop()
    sys.exit(app.exec_())