# Copyright (c) Meta Platforms, Inc. and affiliates
# import logging
import os
# import argparse
import sys
import numpy as np
# from collections import OrderedDict
import torch
import rospy
from std_msgs.msg import Header
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
# import cv2
import message_filters

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

# logger = logging.getLogger("detectron2")

# To enable python to find the cubercnn package, we change working directory to the cubercnn folder
# TODO: This can be modified (path found using rospkg) after we make the package a ros package
# pkg_path = rospkg.RosPack().get_path('cubercnn')
# os.chdir(pkg_path)
os.chdir("/home/ariel/james_thesis/code/omni3d")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

#TURN THIS SCRIPT INTO A ROS NODE
#First we handle the arguments as global variables
INPUT_FOLDER = "datasets/coco_examples"
CONFIG_FILE = "cubercnn://omni3d/cubercnn_DLA34_FPN.yaml"
FOCAL_LENGTH = 0
PRINCIPAL_POINT = []
IMAGE_SIZE = (640, 480)
THRESHOLD = 0.6
MODEL_WEIGHTS = "cubercnn://omni3d/cubercnn_DLA34_FPN.pth"
OUTPUT_DIR = "output/demo"
DISPLAY = False

def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = CONFIG_FILE

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):    
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    # cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = MODEL_WEIGHTS
    cfg.OUTPUT_DIR = OUTPUT_DIR
    cfg.freeze()
    # default_setup(cfg, args)
    return cfg

#Now we create the ros node class
class CubeRCNNNode(object):
    def __init__(self):
        #Setup the model
        self.cfg = setup()
        self.model = build_model(self.cfg)
        
        DetectionCheckpointer(self.model, save_dir=self.cfg.OUTPUT_DIR).resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=True
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        #Inference parameters
        focal_length = FOCAL_LENGTH
        principal_point = PRINCIPAL_POINT
        h, w = IMAGE_SIZE
        self.thres = THRESHOLD

        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2

        if len(principal_point) == 0:
            px, py = w/2, h/2
        else:
            px, py = principal_point

        self.K = np.array([
            [focal_length, 0.0, px], 
            [0.0, focal_length, py], 
            [0.0, 0.0, 1.0]
        ])

        self.output_dir = self.cfg.OUTPUT_DIR
        
        min_size = self.cfg.INPUT.MIN_SIZE_TEST
        max_size = self.cfg.INPUT.MAX_SIZE_TEST
        self.augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

        category_path = os.path.join(util.file_parts(CONFIG_FILE)[0], 'category_meta.json')
        # store locally if needed
        if category_path.startswith(util.CubeRCNNHandler.PREFIX):
            category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

        metadata = util.load_json(category_path)
        self.cats = metadata['thing_classes']


        rospy.loginfo("CubeRCNNModel loaded successfully, device: {}".format(self.device))

        #Setup subscriber and publisher
        rospy.init_node("cubercnn_node", anonymous=True)
        # self.img_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_callback)
        # message filter for image and camera_info (use exact time sync since image and camera_info are published at the same time)
        self.img_sub = message_filters.Subscriber("/camera", Image)
        self.info_sub = message_filters.Subscriber("/camera_info", CameraInfo)
        self.ts = message_filters.TimeSynchronizer([self.img_sub, self.info_sub], 10)
        self.ts.registerCallback(self.img_callback)

        self.objects_pub = rospy.Publisher("/objects", Header, queue_size=10)
        self.detections_pub = rospy.Publisher("/detections", Image, queue_size=10)

        self.bridge = CvBridge()

        rospy.loginfo("CubeRCNNNode initialized successfully")

    def img_callback(self, data, camera_info):
        #Calculate inference rate
        t1 = rospy.Time.now()

        #Convert the camera_info to K
        self.K = np.array(camera_info.K).reshape((3,3))

        #Convert the image to cv2 format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        #Process the image
        with torch.no_grad():
            self.inference(cv_image)

        inference_time = rospy.Time.now() - t1
        rospy.loginfo("Inference time: {}, FPS: {}".format(inference_time.to_sec(), 1.0/inference_time.to_sec()))

    def start(self):
        rospy.loginfo("CubeRCNNNode started, waiting for image")
        rospy.spin()

    def inference(self, im):
        if im is None:
            return
        
        image_shape = im.shape[:2]  # h, w

        aug_input = T.AugInput(im)
        _ = self.augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(), 
            'height': image_shape[0], 'width': image_shape[1], 'K': self.K
        }]

        dets = self.model(batched)[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions, 
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):

                # skip
                if score < self.thres:
                    continue
                
                cat = self.cats[cat_idx]

                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)
        
        rospy.loginfo("Detected {} objects".format(len(meshes)))
        rospy.loginfo(meshes_text)

        if len(meshes) > 0:
            im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, self.K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)
            
            #Publish the detections
            try:
                im_drawn_rgb = im_drawn_rgb.astype(np.uint8)
                self.detections_pub.publish(self.bridge.cv2_to_imgmsg(im_drawn_rgb, "bgr8"))
            except CvBridgeError as e:
                print(e)

        #     if DISPLAY:
        #         im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
        #         vis.imshow(im_concat)

        #     util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
        #     util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))
        # else:
        #     util.imwrite(im, os.path.join(output_dir, im_name+'_boxes.jpg'))

if __name__ == "__main__":
    #Setup the node
    node = CubeRCNNNode()
    node.start()