## KAM-Net: Keypoint-Aware and Keypoint-Matching Network for Vehicle Detection from 2D Point Cloud

This repository is the implementation and the related datasets of our paper `KAM-Net: Keypoint-Aware and Keypoint-Matching Network for Vehicle Detection from 2D Point Cloud`. If you benefit from this repository, please cite our paper.

```
@ARTICLE{,
        title = {KAM-Net: Keypoint-Aware and Keypoint-Matching Network for Vehicle Detection from 2D Point Cloud},
      journal = {IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE},
         year = {2021},
        pages = {},
      author  = {Tianpei, Zou and Guang, Chen and Zhijun, Li and Wei, He and Sanqing, Qu and Shangding, Gu and Alois, Knoll}, 
       eprint = {} 
}
```

### Abstract

2D LiDAR is an efficient alternative sensor for vehicle detection, which is one of the most critical tasks in autonomous driving. Compared to the fully-developed 3D LiDAR vehicle detection, 2D LiDAR vehicle detection has much room to improve. Most existing state-of-the-art works represent 2D point clouds as pseudo-images and then perform detection with traditional object detectors on 2D images. However, they ignore the sparse representation and geometric information of vehicles in the 2D cloud points. In this paper, we present Keypoint-Aware and Keypoint-Matching Network termed as KAM-Net, specifically focuses on better detecting the vehicles by explicitly capturing and extracting the sparse information of L-shape in 2D LiDAR point clouds. The whole framework consists of two stages, namely keypoint-aware stage and keypoint-matching stage. The keypoint-aware stage utilizes the heatmap and edge extraction module to simultaneously predict the position of L-shaped keypoints and inflection offset of L-shaped endpoints. The keypoint-matching stage is followed to group the keypoints and produce the oriented bounding boxes with axis by utilizing the endpoint-matching and Lshaped-matching methods. Further, we conduct extensive experiments on a recently released public dataset to evaluate the effectiveness of our approach. The results show that our KAM-Net achieves a new state-of-the-art performance.
