#!/bin/bash
ros2 bag record \
    /zed/zed_node/rgb/camera_info \
    /zed/zed_node/rgb/image_rect_color \
    /hamilton/frontleft/color/camera_info \
    /hamilton/frontleft/color/image_raw \
    /hamilton/frontright/color/camera_info \
    /hamilton/frontright/color/image_raw \
    /tf \
    /tf_static $@