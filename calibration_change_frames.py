import numpy as np
import robotdatapy as rdp

"""
This file can be used if you have a tree of transformations (e.g., from tf_static)
and a calibrated transform between two frames, and you want to express that calibration 
instead between two different frames.

For example, you already have in your tf tree a transform between base_link and lidar,
and now you just computed the transformation between the lidar and the camera_optical frames,
but you want to put in your tf tree a transform between the lidar and the camera_link frames.
"""

def calibration_change_frames(T_desired1_calibrated1, T_desired2_calibrated2, 
    T_calibrated1_calibrated2):
    """
    T_desired1_calibrated1: T^desired1_calibrated1. Transform between your desired parent frame_id 
        and the frame id of the parent frame found from your calibration (identity if these are
        the same frames).
    T_desired2_calibrated2: T^desired2_calibrated2. Transform between your desired frame_id and 
        the frame id of the parent frame found from your calibration (identity if these are
        the same frames).
    T_calibrated1_calibrated2: T^calibrated1_calibrated2. Transform from calibration.
    """

    T_desired1_desired2 = T_desired1_calibrated1 @ T_calibrated1_calibrated2 @ \
        np.linalg.inv(T_desired2_calibrated2)
    return T_desired1_desired2

def calibration_change_frames_from_tf_static(
    bag_path, desired_frame_1, desired_frame_2, calibration_frame_1, 
    calibration_frame_2, T_calib1_calib2
):
    T_desired1_calibrated1 = rdp.data.PoseData.any_static_tf_from_bag(
        bag_path, parent_frame=desired_frame_1, child_frame=calibration_frame_1)
    T_desired2_calibrated2 = rdp.data.PoseData.any_static_tf_from_bag(
        bag_path, parent_frame=desired_frame_2, child_frame=calibration_frame_2)
    return calibration_change_frames(T_desired1_calibrated1, T_desired2_calibrated2,
                                     T_calib1_calib2)