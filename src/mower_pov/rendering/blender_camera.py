"""Convert the canonical optical camera to Blender's local camera axes."""

import numpy as np

from ..camera import camera_pose
from ..motion import sample_pose


def blender_camera_matrix(optical_pose):
    return optical_pose @ np.diag([1., -1., -1., 1.])


def mount_camera(scene, config):
    import bpy

    data = bpy.data.cameras.new("Onboard wide-angle lens")
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = 36.
    data.lens = data.sensor_width / (2 * np.tan(np.deg2rad(config.camera.hfov_deg) / 2))
    data.clip_start = .035
    data.clip_end = 160.
    camera = bpy.data.objects.new("Rigid onboard camera", data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def apply_vehicle_pose(bundle, camera, chassis, time):
    from mathutils import Matrix

    body = sample_pose(bundle.trajectory, time)
    chassis.matrix_world = Matrix(body.tolist())
    camera.matrix_world = Matrix(blender_camera_matrix(camera_pose(bundle.resolved_config, body)).tolist())
