import json
import argparse
import robotdatapy as rdp

from calibration_change_frames import calibration_change_frames_from_tf_static

def process_direct_visual_lidar_calibration_output(
        json_output, bag, desired_parent_frame_id, desired_child_frame_id):
    f = open(json_output)
    output_dict = json.load(f)
    f.close()

    image_topic = output_dict['meta']['image_topic']
    points_topic = output_dict['meta']['points_topic']

    img_data = rdp.data.ImgData.from_bag(bag, image_topic)
    pcd_data = rdp.data.PointCloudData.from_bag(bag, points_topic)

    img_frame_id = img_data.img_header(img_data.t0).frame_id
    pcd_frame_id = pcd_data.msg_header(pcd_data.t0).frame_id

    T_lidar_camera_xyzquat = output_dict['results']['T_lidar_camera']
    T_lidar_camera = rdp.transform.xyz_quat_to_transform(
        T_lidar_camera_xyzquat[:3], T_lidar_camera_xyzquat[3:])

    T_parent_child = calibration_change_frames_from_tf_static(
        bag_path=bag,
        desired_frame_1=desired_parent_frame_id,
        desired_frame_2=desired_child_frame_id,
        calibration_frame_1=pcd_frame_id,
        calibration_frame_2=img_frame_id,
        T_calib1_calib2=T_lidar_camera
    )

    return T_parent_child

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--json-calibration', required=True)
    parser.add_argument('-b', '--bag', required=True)
    parser.add_argument('-f', '--frame-ids', nargs=2, required=True)

    args = parser.parse_args()

    T_parent_child = process_direct_visual_lidar_calibration_output(
        args.json_calibration, args.bag, args.frame_ids[0], args.frame_ids[1]
    )

    print(f"T^{args.frame_ids[0]}_{args.frame_ids[1]} = ")
    print(T_parent_child)

    xyzrpy = rdp.transform.transform_to_xyz_quat(T_parent_child)
    print(f'<origin xyz="{xyzrpy[0]} {xyzrpy[1]} {xyzrpy[2]}" rpy="{xyzrpy[3]} {xyzrpy[4]} {xyzrpy[5]}" />')