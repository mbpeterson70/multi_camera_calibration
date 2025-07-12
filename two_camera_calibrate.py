import argparse
import numpy as np
from robotdatapy.data import ImgData
from robotdatapy.data import PoseData
from robotdatapy.transform import transform_to_xyzrpy
from robotdatapy.exceptions import NoDataNearTimeException
from pupil_apriltags import Detector
from scipy.spatial.transform import Rotation as R
import tqdm
import yaml

SYNC_THRESHOLD = 0.02  # seconds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to ROS 2 bag")
    parser.add_argument("-c", "--camera-topics", nargs=2, required=False, help="Two image topics")
    parser.add_argument("-i", "--info-topics", nargs=2, required=False, help="Two camera info topics")
    parser.add_argument("-a", "--april-tag", type=int, required=False, help="AprilTag ID to detect")
    parser.add_argument("-s", "--size", type=float, required=False, help="AprilTag size in meters")
    parser.add_argument("-f", "--frames", type=str, nargs=2, required=False, help="Output transformation between two camera frames")
    parser.add_argument("-y", "--yaml", type=str, required=False, help="Path to YAML file with calibration configuration")
    args = parser.parse_args()
    
    if args.yaml:
        yaml_config = yaml.safe_load(open(args.yaml, 'r'))
        args.camera_topics = yaml_config['camera_topics']
        args.info_topics = yaml_config['info_topics']
        args.april_tag = yaml_config['april_tag']
        args.size = yaml_config['size']
        args.frames = yaml_config['frames']
    return args

def find_synchronized_pairs(imgdata1, imgdata2, threshold=SYNC_THRESHOLD):
    pairs = []
    for t1 in imgdata1.times:
        try:
            t2 = imgdata2.nearest_time(t1)
        except NoDataNearTimeException:
            continue
        if abs(t1 - t2) < threshold:
            pairs.append((t1, t2))
    return pairs

def detect_tag(detector, img, K, tag_id, tag_size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    detections = detector.detect(gray, estimate_tag_pose=True,
                                 camera_params=(fx, fy, cx, cy),
                                 tag_size=tag_size)
    for d in detections:
        if d.tag_id == tag_id:
            return d.pose_t, d.pose_R
    return None, None

def compute_relative_transform(pose_t1, pose_R1, pose_t2, pose_R2):
    T_cam1_tag = np.eye(4)
    T_cam1_tag[:3, :3] = pose_R1
    T_cam1_tag[:3, 3] = pose_t1.flatten()

    T_cam2_tag = np.eye(4)
    T_cam2_tag[:3, :3] = pose_R2
    T_cam2_tag[:3, 3] = pose_t2.flatten()

    T_cam1_cam2 = T_cam1_tag @ np.linalg.inv(T_cam2_tag)
    return T_cam1_cam2

def average_transforms(transforms):
    print(f"Found {len(transforms)} valid transforms.")
    
    # for transform in transforms:
    #     print(transform)
    
    if not transforms:
        return None

    translations = np.array([T[:3, 3] for T in transforms])
    avg_translation = np.mean(translations, axis=0)

    rotations = R.from_matrix([T[:3, :3] for T in transforms])
    avg_rotation = rotations.mean().as_matrix()

    T_avg = np.eye(4)
    T_avg[:3, :3] = avg_rotation
    T_avg[:3, 3] = avg_translation
    return T_avg

def main():
    args = parse_args()

    imgdata1 = ImgData.from_bag(args.input, args.camera_topics[0], camera_info_topic=args.info_topics[0])
    imgdata2 = ImgData.from_bag(args.input, args.camera_topics[1], camera_info_topic=args.info_topics[1])
    
    # PoseData.static_tf_from_bag(
    #     args.input, args.frames[0], imgdata1.img_header(imgdata1.t0).frame_id)
    if args.frames[0] == imgdata1.img_header(imgdata1.t0).frame_id:
        T_frame1_cam1 = np.eye(4)
    else:
        T_frame1_cam1 = PoseData.static_tf_from_bag(
            args.input, args.frames[0], imgdata1.img_header(imgdata1.t0).frame_id)
    if args.frames[1] == imgdata2.img_header(imgdata2.t0).frame_id:
        T_frame2_cam2 = np.eye(4)
    else:
        T_frame2_cam2 = PoseData.static_tf_from_bag(
            args.input, args.frames[1], imgdata2.img_header(imgdata2.t0).frame_id)

    detector = Detector(families="tag36h11")
    transforms = []

    for t1, t2 in tqdm.tqdm(find_synchronized_pairs(imgdata1, imgdata2)):
        img1 = imgdata1.img(t1)
        img2 = imgdata2.img(t2)

        pose_t1, pose_R1 = detect_tag(detector, img1, imgdata1.K, args.april_tag, args.size)
        pose_t2, pose_R2 = detect_tag(detector, img2, imgdata2.K, args.april_tag, args.size)

        if pose_t1 is not None and pose_t2 is not None:
            T = compute_relative_transform(pose_t1, pose_R1, pose_t2, pose_R2)
            transforms.append(T)

    T_avg = average_transforms(transforms)

    if T_avg is not None:
        print("Averaged Transform from Camera 1 to Camera 2:")
        print(T_avg)
        print(f"Average rotation error (deg): {np.rad2deg(np.mean([R.from_matrix(T[:3, :3] @ T_avg[:3, :3].T).magnitude() for T in transforms]))}")
        print(f"Average translation error (m): {np.mean([np.linalg.norm(T[:3, 3] - T_avg[:3, 3]) for T in transforms])}")
    
    else:
        print("No valid tag detections in synchronized image pairs.")
        
    T_cam1_cam2 = T_avg
    T_frame1_frame2 = T_frame1_cam1 @ T_cam1_cam2 @ np.linalg.inv(T_frame2_cam2)
    print(f"Desired tf: {T_frame1_frame2}")
    xyzrpy = transform_to_xyzrpy(T_frame1_frame2)
    print(f'<origin xyz="{xyzrpy[0]} {xyzrpy[1]} {xyzrpy[2]}" rpy="{xyzrpy[3]} {xyzrpy[4]} {xyzrpy[5]}" />')

if __name__ == "__main__":
    import cv2
    main()
