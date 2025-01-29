"""Trackball class for 3D manipulation of viewpoints.
"""

import numpy as np
import quaternion
from math import sqrt

import trimesh.transformations as transformations


def linePlaneCollision(
    planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6
):

    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        raise RuntimeError("no intersection or line is within plane")

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi


def dot_product(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])


def norm(x):
    return sqrt(dot_product(x, x))


def normalize(x):
    return [x[i] / norm(x) for i in range(len(x))]


def project_onto_plane(x, n):
    d = dot_product(x, n) / norm(n)
    p = [d * normalize(n)[i] for i in range(len(n))]
    return [x[i] - p[i] for i in range(len(x))]


def axis_rot(axis, a):
    return quaternion.as_rotation_matrix(
        quaternion.from_rotation_vector(a * axis)
    )


class Trackball(object):
    """A trackball class for creating camera transforms from mouse movements."""

    STATE_ROTATE = 0
    STATE_PAN = 1
    STATE_ROLL = 2
    STATE_ZOOM = 3

    def __init__(
        self,
        pose,
        size,
        scale,
        target=np.array([0.0, 0.0, 0.0]),
        rotation_mode="trackball",
    ):
        """Initialize a trackball with an initial camera-to-world pose
        and the given parameters.
        Parameters
        ----------
        pose : [4,4]
            An initial camera-to-world pose for the trackball.
        size : (float, float)
            The width and height of the camera image in pixels.
        scale : float
            The diagonal of the scene's bounding box --
            used for ensuring translation motions are sufficiently
            fast for differently-sized scenes.
        target : (3,) float
            The center of the scene in world coordinates.
            The trackball will revolve around this point.
        """
        self._size = np.array(size)
        self._scale = float(scale)

        self._pose = pose
        self._n_pose = pose

        self._target = target
        self._n_target = target

        self._state = Trackball.STATE_ROTATE

        self._target = -self._pose[:3, 2]

        self.rotation_mode = rotation_mode
        self.ip = target

    @property
    def pose(self):
        """autolab_core.RigidTransform : The current camera-to-world pose."""
        return self._n_pose

    def set_state(self, state):
        """Set the state of the trackball in order to change the effect of
        dragging motions.
        Parameters
        ----------
        state : int
            One of Trackball.STATE_ROTATE, Trackball.STATE_PAN,
            Trackball.STATE_ROLL, and Trackball.STATE_ZOOM.
        """
        self._state = state

    def resize(self, size):
        """Resize the window.
        Parameters
        ----------
        size : (float, float)
            The new width and height of the camera image in pixels.
        """
        self._size = np.array(size)

    def down(self, point):
        """Record an initial mouse press at a given point.
        Parameters
        ----------
        point : (2,) int
            The x and y pixel coordinates of the mouse press.
        """
        self._pdown = np.array(point, dtype=np.float32)
        self._pose = self._n_pose
        self._target = self._n_target

    def drag(self, point):
        """Update the tracball during a drag.
        Parameters
        ----------
        point : (2,) int
            The current x and y pixel coordinates of the mouse during a drag.
            This will compute a movement for the trackball with the relative
            motion between this point and the one marked by down().
        """
        point = np.array(point, dtype=np.float32)
        dx, dy = point - self._pdown
        # dy = -dy
        mindim = 0.3 * np.min(self._size)
        # mindim = 1000 * np.min(self._size)

        target = np.array(self._n_target)
        x_axis = self._pose[:3, 0].flatten() / np.linalg.norm(self._pose[:3, 0])
        y_axis = self._pose[:3, 1].flatten() / np.linalg.norm(self._pose[:3, 1])
        z_axis = self._pose[:3, 2].flatten() / np.linalg.norm(self._pose[:3, 2])
        eye = self._pose[:3, 3].flatten()
        # direction = -self._pose[:3,2]
        direction = (target - eye) / np.linalg.norm(target - eye)

        # Interpret drag as a rotation
        if self._state == Trackball.STATE_ROTATE:
            if self.rotation_mode == "trackball":
                intersection_point = linePlaneCollision(
                    planeNormal=np.array([0, 0, 1.0]),
                    planePoint=np.array([0, 0, 0]),
                    rayDirection=direction,
                    rayPoint=eye,
                )

                zax = (intersection_point - eye) / np.linalg.norm(
                    eye - intersection_point
                )
                xax = np.cross(zax, np.array([0, 0, 1.0]))
                xax = xax / np.linalg.norm(xax)
                yax = np.cross(zax, xax)
                yax = yax / np.linalg.norm(yax)

                x_angle = -dx / mindim
                x_rot_mat = transformations.rotation_matrix(
                    x_angle, -yax, intersection_point - self.ip
                )

                y_angle = dy / mindim
                y_rot_mat = transformations.rotation_matrix(
                    y_angle, xax, intersection_point - self.ip
                )

                self._n_pose = x_rot_mat.dot(y_rot_mat.dot(self._pose))
                self.ip = intersection_point

            else:
                self._n_pose = self._pose.copy()
                self._n_pose = self.tilt(
                    -dy / 10, self._n_pose, cm_orig=self._n_pose
                )
                self._n_pose = self.yaw(dx / 10, self._n_pose)

        # Interpret drag as a roll about the camera axis
        elif self._state == Trackball.STATE_ROLL:
            center = self._size / 2.0
            v_init = self._pdown - center
            v_curr = point - center
            v_init = v_init / np.linalg.norm(v_init)
            v_curr = v_curr / np.linalg.norm(v_curr)

            theta = -np.arctan2(v_curr[1], v_curr[0]) + np.arctan2(
                v_init[1], v_init[0]
            )

            rot_mat = transformations.rotation_matrix(theta, z_axis, target)

            self._n_pose = rot_mat.dot(self._pose)

        # Interpret drag as a camera pan in view plane
        elif self._state == Trackball.STATE_PAN:
            dx = -dx / (2.0 * mindim) * self._scale
            dy = dy / (2.0 * mindim) * self._scale

            translation = dx * x_axis + dy * y_axis
            self._n_target = self._target + translation
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            self._n_pose = t_tf.dot(self._pose)

        # Interpret drag as a zoom motion
        elif self._state == Trackball.STATE_ZOOM:
            intersection_point = linePlaneCollision(
                planeNormal=np.array([0, 0, 1.0]),
                planePoint=np.array([0, 0, 0]),
                rayDirection=direction,
                rayPoint=eye,
            )
            y_angle = dy / mindim
            y_rot_mat = transformations.rotation_matrix(
                y_angle / 1000, y_axis, intersection_point
            )
            self._n_pose = y_rot_mat.dot(self._pose)

    def scroll(self, clicks):
        """Zoom using a mouse scroll wheel motion.
        Parameters
        ----------
        clicks : int
            The number of clicks. Positive numbers indicate forward wheel
            movement.
        """
        ratio = 0.99995

        mult = 1.0
        if clicks > 0:
            mult = ratio**clicks
        elif clicks < 0:
            mult = (1.0 / ratio) ** abs(clicks)

        def compute_zoom(mat, mult):
            z_axis = mat[:3, 2].flatten() / np.linalg.norm(mat[:3, 2])
            eye = mat[:3, 3].flatten()
            target = self._target
            radius = np.linalg.norm(eye - target)
            translation = (mult * radius - radius) * z_axis
            t_tf = np.eye(4)
            t_tf[:3, 3] = translation
            mat = t_tf.dot(mat)
            return mat

        self._n_pose = compute_zoom(self._n_pose, mult)
        self._pose = compute_zoom(self._pose, mult)

    def rotate(self, azimuth, axis=None):
        """Rotate the trackball about the "Up" axis by azimuth radians.
        Parameters
        ----------
        azimuth : float
            The number of radians to rotate.
        """
        target = self._target

        y_axis = self._n_pose[:3, 1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        self._n_pose = x_rot_mat.dot(self._n_pose)

        y_axis = self._pose[:3, 1].flatten()
        if axis is not None:
            y_axis = axis
        x_rot_mat = transformations.rotation_matrix(azimuth, y_axis, target)
        self._pose = x_rot_mat.dot(self._pose)

    def forward(self, step=0.5):
        cm = self.get_camera_mat()
        cm[:3, 3] += -step * cm[:3, 2]
        self.set_camera_mat(cm)

    def yaw(self, step=1.0, cm=None):
        a = np.deg2rad(step)
        rot = np.array(
            [[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]]
        )
        if cm is None:
            cm = self.get_camera_mat()

        cm1 = cm.copy()
        cm1[:3, :3] = rot.dot(cm1[:3, :3])

        return cm1

    def tilt(self, step=1.0, cm=None, cm_orig=None):
        a = np.deg2rad(step)

        if cm is None:
            cm = self.get_camera_mat()

        cm1 = cm.copy()

        axis = cm_orig[:3, 0]
        rot = axis_rot(axis, a)
        cm1[:3, :3] = rot.dot(cm1[:3, :3])

        return cm1
