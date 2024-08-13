"""
File: pose_estimation_client.py
Author: xmfang
Description: interface for pose estimation 

History:
    - Version 0.0 (2024-07-31): xmfang, created

Dependencies:
    - in README.md
"""
import os
import rospy
import imageio
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import Bool
from foundation_pose_ros.srv import PoseEstimation, PoseEstimationRequest

def foundation_pose_client(rgb, depth, mask, K, is_vis):
    cb = CvBridge()
    rospy.wait_for_service('/foundation_pose_server')
    try:
        pose_estimation = rospy.ServiceProxy('foundation_pose_server', PoseEstimation)
        rgb = cb.cv2_to_imgmsg(rgb, 'rgb8')
        depth = cb.cv2_to_imgmsg(depth, 'passthrough')
        mask = cb.cv2_to_imgmsg(mask, 'mono8')
        K = K.flatten().tolist()
        is_vis = Bool(is_vis)
        resp = pose_estimation(PoseEstimationRequest(rgb, depth, mask, K, is_vis))
        return resp.pose, resp.bbox
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")

if __name__ == "__main__":
    
    code_dir = os.path.dirname(os.path.realpath(__file__))
    color = imageio.imread(f'{code_dir}/demo_data/tote_transfering/box/rgb/000000.png')
    depth = imageio.imread(f'{code_dir}/demo_data/tote_transfering/box/depth/000000.png')
    mask = imageio.imread(f'{code_dir}/demo_data/tote_transfering/box/masks/000000.png')
    K = np.array([607.1722412109375, 0, 319.3473205566406, 0, 607.257080078125, 253.62425231933594, 0, 0, 1]).reshape(3,3)
        
    pose, bbox = foundation_pose_client(color, depth, mask, K, True)
    print(pose, bbox)

