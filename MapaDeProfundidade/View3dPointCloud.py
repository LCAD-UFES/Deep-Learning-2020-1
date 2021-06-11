
import open3d as o3d
import matplotlib.pyplot as plt
from torch.nn import intrinsic
import deep_mapper as dm
import cv2
import numpy as np

if __name__ == "__main__":

    img = cv2.imread("test_imgs/input2.jpg")
    intrinsicMat = np.eye(3,dtype=np.float32)
    intrinsicMat[0,0]=427.11285934127466 # fx
    intrinsicMat[0,2]=340.29418100385436 # cx
    intrinsicMat[1,1]=567.3402504340095  # fy
    intrinsicMat[1,2]=253.19875767269184 # cy
    print(intrinsicMat)
    dists=np.zeros((5,1),np.float64)
    dists[0,0]=-0.4012898929082725
    dists[1,0]=0.2119039649987176
    dists[2,0]=-0.0017767161262766081
    dists[3,0]=-0.0033067056085569397
    dists[4,0]=-0.07115199650013011
    print(dists.shape)
    img2 = cv2.undistort(img ,intrinsicMat,dists,None,intrinsicMat)
    plt.imshow(img2)
    plt.show()
    cv2.imwrite("/tmp/input2.jpg",img2)
    dm.initialize('kitti')
    dm.inferenceDepth(img2)
    color_raw = o3d.io.read_image("test_imgs/input2.jpg")
    depth_raw = o3d.io.read_image("/tmp/output2.png")
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=25000)
    print(rgbd_image)
    
    plt.subplot(1, 2, 1)
    plt.title('Redwood grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('Redwood depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
             o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd])
