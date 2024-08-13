"""
File: pose_estimation_server.py
Author: xmfang
Description: interface for pose estimation 

History:
    - Version 0.0 (2024-07-31): xmfang, created

Dependencies:
    - in README.md
"""
import os
import rospy
from cv_bridge import CvBridge
import numpy as np
from estimater import *
from datareader import *
from foundation_pose_ros.srv import PoseEstimation, PoseEstimationResponse

class PoseEstimationServer():
    
    def __init__(self, mesh_file:str):
        set_seed(0)

        self.cb = CvBridge()
        code_dir = os.path.dirname(os.path.realpath(__file__))
        debug_dir = f'{code_dir}/debug'
        debug = 1
        self.mesh = trimesh.load(mesh_file)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        
        self.est = FoundationPose(model_pts=self.mesh.vertices, 
                                  model_normals=self.mesh.vertex_normals, 
                                  mesh=self.mesh, 
                                  scorer=scorer, 
                                  refiner=refiner, 
                                  debug_dir=debug_dir, 
                                  debug=debug, 
                                  glctx=glctx)
        
        self.est_refine_iter = 5
        self.track_refine_iter = 2
        self.cnt = 0
        
    def inference(self, 
                  rgb:np.ndarray, 
                  depth:np.ndarray, 
                  mask:np.ndarray, 
                  K:np.ndarray, 
                  is_vis:bool=False,
                  ):
        if depth.mean() > 10.0:
            depth = depth * 0.001
        if self.cnt >= 0:
            pose = self.est.register(K=K, 
                                     rgb=rgb, 
                                     depth=depth, 
                                     ob_mask=mask, 
                                     iteration=self.est_refine_iter)
        else:
            pose = self.est.track_one(rgb=rgb, 
                                      depth=depth, 
                                      K=K, 
                                      iteration=self.track_refine_iter)
        self.cnt += 1
        
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        center_pose = pose@np.linalg.inv(to_origin)
        if is_vis:
            vis = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(rgb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, 
                                transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(0)
        
        return center_pose, bbox

    def pose_estimation(self, srv):
        rospy.logfatal("Received a pose estimation request.")
        color = self.cb.imgmsg_to_cv2(srv.rgb, desired_encoding='rgb8')
        depth = self.cb.imgmsg_to_cv2(srv.depth, desired_encoding='passthrough')
        mask = self.cb.imgmsg_to_cv2(srv.mask, desired_encoding='mono8')
        K = np.array(srv.K).reshape(3,3)
        is_vis = srv.is_vis.data
        pose, bbox = self.inference(rgb=color, depth=depth, mask=mask, K=K, is_vis=is_vis)
        rospy.loginfo(f'pose:\n{pose}')
        rospy.loginfo(f'bbox:\n{bbox}')
        return PoseEstimationResponse(pose=pose.flatten().tolist(), bbox=bbox.flatten().tolist())

    def foundation_pose_server(self):
        rospy.init_node('foundation_pose_server')
        s = rospy.Service('foundation_pose_server', PoseEstimation, self.pose_estimation)
        rospy.loginfo("Ready to estimate the pose.")
        rospy.spin()

def scale_obj(file_path, x_scale_factor=1.0, y_scale_factor=1.0, z_scale_factor=1.0):
    """scale obj model

    Args:
        file_path (str): file path  
        x_scale_factor (float, optional): length factor. Defaults to 1.0.
        y_scale_factor (float, optional): height factor. Defaults to 1.0.
        z_scale_factor (float, optional): width factor. Defaults to 1.0.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    file_name = os.path.basename(file_path)
    length = float(file_name.split('_')[1])
    width = float(file_name.split('_')[2])
    height = float(file_name.split('_')[3].split('.')[0])
    dir = os.path.dirname(file_path)
    new_file_path = f'{dir}/box_{int(length*x_scale_factor)}_{int(width*z_scale_factor)}_{int(height*y_scale_factor)}.obj'
    with open(new_file_path, 'w') as file:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x *= x_scale_factor  # Adjust the x-coordinate
                y *= y_scale_factor  # Adjust the y-coordinate
                z *= z_scale_factor  # Adjust the z-coordinate
                file.write(f'v {x} {y} {z}\n')
            else:
                file.write(line)
        print(f'Scaled obj file has been saved to {new_file_path}.')
                    
if __name__ == "__main__":
    
    import time
    import argparse
    
    parser = argparse.ArgumentParser(description='Pose Estimation Server')
    parser.add_argument('--mesh_file', type=str, default='/home/ps/Projects/FoundationPose/demo_data/tote_transfering/box/mesh/box.obj')
    args = parser.parse_args()
    
    if True:
        ts = time.time()
        PE = PoseEstimationServer(mesh_file=args.mesh_file)
        te = time.time()
        print(f'Initialization time: {te-ts:.3f}s')
    
        PE.foundation_pose_server()
    
    if False:
        ts = time.time()
        scale_obj(file_path='/home/ps/Projects/galbot_ws/src/foundation_pose_ros/src/demo_data/tote_transfering/box_400_300_280.obj', y_scale_factor=(230/280))