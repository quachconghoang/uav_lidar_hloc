import numpy as np
from config import camera, RGBDImage, PointCloud, TriangleMesh, Image, Vector3dVector, draw_geometries
from config import gtsam, Point2, Point3, Rot3, Pose3, Cal3_S2, Values, PinholeCameraCal3_S2
import cv2 as cv

def generate3D(frame: dict):
    rgbd0 = RGBDImage.create_from_color_and_depth(
        color=Image(frame['color']),
        depth=Image(frame['depth']),
        depth_scale=1.0, depth_trunc=np.inf,
        convert_rgb_to_intensity=False)
    cloud = PointCloud.create_from_rgbd_image(image=rgbd0, intrinsic=frame['intr'], extrinsic=frame['extr'])
    cloud.transform(frame['transform'])
    # frame['cloud'] = cloud
    frame['point3D'] = np.asarray(cloud.points)

def cal_Point_Project_TartanCam(cam: PinholeCameraCal3_S2, pw: Point3):
    q = cam.pose().transformTo(pw)[[1,2,0]] # -> world xyz -> cam -> yzx [== camExtr]
    pn = cam.Project(q)
    pi = cam.calibration().uncalibrate(pn)
    # d = 1. / q[2]
    # print('debug: ',pn, pi, q, d)
    return np.rint(pi).astype(np.int32),q[2]

def cal_Point_Project_General(cam: PinholeCameraCal3_S2, p_src: Point3, cam_extr=np.eye(3)):
    q = cam.pose().transformTo(p_src)
    q_local = cam_extr.dot(q)
    pn = cam.Project(q_local)
    pi = cam.calibration().uncalibrate(pn)
    return pi # np.rint(pi).astype(np.int32)

#Calculate [expected] 2D location in target camera of 3D source points
def getPointsProject2D(pts_2d, pts_3d, target_cam):
    p_in_target = []
    p_valid = []

    for kp in pts_2d:
        x, y = kp[0], kp[1]
        p0w = Point3(pts_3d[int(y * 640 + x)])
        p0_target, _ = cal_Point_Project_TartanCam(target_cam, p0w)
        p_in_target.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            p_valid.append(True)
        else:
            p_valid.append(False)

    return p_in_target,p_valid

def getGroundTruth(src_p3d_world, src_p2d, target_cam):
    kpt0_gt = []
    kpt0_valid = []
    for kp in src_p2d:
        x,y = kp[0],kp[1]
        p0w = Point3(src_p3d_world[int(y * 640 + x)])
        p0_target,_ = cal_Point_Project_TartanCam(target_cam, p0w)
        kpt0_gt.append(p0_target)
        if (0 < p0_target[0] < 640) & (0 < p0_target[1] < 480):
            kpt0_valid.append(True)
        else:
            kpt0_valid.append(False)
    return kpt0_gt, kpt0_valid

# Get disparity map z = 80/disparity
# IDK 16 stand-for what
def getDisparityTartanAIR(img_l, img_r):
    stereo = cv.StereoSGBM_create(numDisparities=32, blockSize=15)
    if(len(img_l.shape) == 3):
        disparity = stereo.compute(cv.cvtColor(img_l, cv.COLOR_RGB2GRAY),
                                   cv.cvtColor(img_r, cv.COLOR_RGB2GRAY)).astype(np.float32)
    else:
        disparity = stereo.compute(img_l, img_r).astype(np.float32)

    return disparity/16

def get3DLocalFromDisparity(kps, disparity):
    # Tartan camera
    f = 320;cx = 320;cy = 240
    kps3D = []
    loss = 0

    for pt in kps:
        x, y = pt.astype(np.int)
        d = disparity[y, x]

        # Interpolation
        if d < 0:
            loss += 1
            d_left = -1
            d_right = -1
            # detect in epipolar line
            iter = 1
            while (iter<32):
                if (x-iter)>=0:
                    d_left = disparity[y, x-iter]
                if (x+iter)<640:
                    d_right = disparity[y, x+iter]
                if (d_left>0):
                    d = d_left;break
                if (d_right>0):
                    d = d_right;break
                iter +=1
                ...
        if(d<0.001):
            # print('failing d ... at:',pt)
            d=20.

        z = 80 / d
        p3 = [z,((x - cx) / f) * z,((y - cy)/ f ) * z] # to world format !!! -> 2-0-1 ### to cam 1-2-0
        # p3 = [((x - cx) / f) * z, ((y - cy) / f) * z, z] # Need EXTRINSIC
        # p3Vec = [z, ((x - cx) / f) * z, ((y - cy) / f) * z, 1]
        kps3D.append(p3)

    # print('loss = ', loss)
    return np.asarray(kps3D)