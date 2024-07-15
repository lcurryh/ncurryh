
<div align="center">
 

<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/1ttttttt.jpg" alt="ddq_arch" width="200">




 
</div>

# DIMD-DETR: DDQ-DETR with Improved Metric space for End-to-End Object Detector on Remote Sensing Aircraft

<div align="center">

<b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://openmmlab.com/">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
    <b><font size="5">GeForce RTX 3090</font></b>
    <sup>
      <a href="https://www.nvidia.cn/data-center/tesla-p100/">
        <i><font size="4">GET IT</font></i>
      </a>
    </sup>

  ![](https://img.shields.io/badge/python-3.8.18-red)
  [![](https://img.shields.io/badge/pytorch-1.10.1-red)](https://pytorch.org/)
  [![](https://img.shields.io/badge/torchvision-0.11.0-red)](https://pypi.org/project/torchvision/)
  [![](https://img.shields.io/badge/MMDetection-3.3.0-red)](https://github.com/open-mmlab/)
  
  

  [🛠️Installation Dependencies](https://blog.csdn.net/m0_46556474/article/details/130778016) |
  [🎤Introduction](https://github.com/open-mmlab/mmdetection) |
 
  [👀Download Dataset](https://pan.baidu.com/s/1ZZYeLK0vwzrXUt_AHgsn0w )) |
  
  [🌊Aircraft Detection](https://github.com/lcurryh/ncurryh/tree/main/DIMD-DETR%20DDQ-DETR%20with%20Improved%20Metric%20space%20for%20End-to-End%20Object%20Detector%20on%20Remote%20Sensing%20Aircraft) |
  [🚀Remote Sensing](https://github.com/lcurryh/ncurryh/tree/main/DIMD-DETR%20DDQ-DETR%20with%20Improved%20Metric%20space%20for%20End-to-End%20Object%20Detector%20on%20Remote%20Sensing%20Aircraft) |
  [🤔End-to-End](https://github.com/lcurryh/ncurryh/tree/main/DIMD-DETR%20DDQ-DETR%20with%20Improved%20Metric%20space%20for%20End-to-End%20Object%20Detector%20on%20Remote%20Sensing%20Aircraft) |
 

</div>

## Dependencies:

 - Python 3.8.18
 - [PyTorch](https://pytorch.org/) 1.10.1
 - [Torchvision](https://pypi.org/project/torchvision/) 0.11.2
 - [OpenCV](https://opencv.org/) 4.8.1
 - Ubuntu 20.04.5 LTS 
 - [MMCV](https://github.com/open-mmlab/mmcv)
 - GeForce RTX 3090
   

## Introduction

this paper proposes an end-to-end remote sensing aircraft detection method based on an improved metric space—DIMD-DETR. Firstly, the pyramid vision transformer V2 module, which combines transformer and pyramid structures, is integrated into the network backbone to enhance detection capabilities across various aircraft sizes. A bilayer layer targeted prediction method is implemented in the head decoder to dynamically interact the target object with the global image context by precisely fusing multi-angle information. Additionally, the augmentations library is utilized to augment image features before the training procedure, simulating various conditions to improve the model's generalization ability. An innovative metric space loss function is designed to enhance the model's accuracy and prediction stability in complex environments. Finally, an advanced dynamic learning rate adjustment framework is introduced to intelligently optimize the learning rate, ensuring high accuracy while accelerating model convergence. The methods are tested on a custom dataset, MDMF, which contains over 17,000 complex remote sensing images of aircraft, achieving an average precision (AP) of 67.5%, an AP50 of 94.9%, and APS of 53.1%. Compared to current popular networks, our model demonstrated superior detection performance with fewer number of parameters. Furthermore, to verify the model's generalization, it was tested on the public remote sensing aircraft datasets LEVIR and DIOR, with performance improvements of 1.9% and 2.4% over the baseline model, respectively. 

<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/2.png" alt="ddq_arch" width="700">
</div>

## Overview of our network architecture
A new end-to-end detection mechanism, DIMD-DETR, based on DDQ-DETR is developed, aiming at enhancing the accuracy of remote sensing aircraft recognition while balancing computing efficiency. The improvements include the use of the BLTP method to enhance the model's sensitivity to subtle scene changes; the incorporation of PVTV2 into the backbone network to boost detection capabilities; the design of a novel metric space loss function to optimize the model's prediction precision and accuracy; the use of the Albu library to reduce the model's tendency to overfit; and the introduction of a dynamic learning rate adjustment framework to ensure high accuracy while accelerating convergence.
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/3.jpg" alt="ddq_arch" width="700">
</div>

## Ours Aircraft dataset
To evaluate our method, we trained the model on four different large-scale datasets:

1)**LEVIR** : This dataset covers most types of ground features in human living environments and has a total of 11,028 labeled targets, including 4,724 airplanes, 3,025 ships, and 3,279 oil tanks. We extracted 1,900 airplane images and used data augmentation methods from Roboflow (https://roboflow.com/) to increase the number to 5,700 images. The dataset was randomly divided into training, validation, and test sets in a ratio of 6:2:2.

2)**DIOR**: This is a large-scale dataset for object detection, containing 20 categories. Each category has a substantial number of objects, images, and instances with complex backgrounds, adding to the dataset's difficulty. We extracted 1,386 airplane images, comprising over 11,000 airplanes. Using data augmentation techniques, we expanded the number of airplane images to 4,104. The dataset was randomly divided into training, validation, and test sets in a ratio of 6:2:2.

3)**MDMF**: We created a dataset called the multi-directional monitoring for complex flight (MDMF) from public resources and the Kaggle platform, including a total of 5,012 images from five open-source datasets and Kaggle. These sources include: RSOD (446 images), NWPU VHR-10 (100 images), LEVIR (1,900 images), DIOR (1,386 images), and an additional 1,200 images selected from Kaggle. Taking advantage of data augmentation techniques, we expanded the total number of images to 17,188. The dataset was divided into training, validation, and test sets in a ratio of 6:2:2.

4)**B-MDMF**: In the MDMF dataset, we added 4,500 fine-grained images from the FGVC-aircraft dataset, naming it the broad multi-directional monitoring for complex flight (B-MDMF). This dataset includes 5,012 images from MDMF and 4,500 images from FGVC-Aircraft, totaling 9,512 images. Again, making use of data augmentation techniques, we extended the total number of images to 19,012. The dataset was divided into training, validation, and test sets in a ratio of 6:2:2.

To enhance the model's performance of recognizing remote sensing airplane targets, nearly half of the training data in the B-MDMF dataset is sourced from FGVC-Aircraft. These data include images with more detailed and fine-grained annotations, potentially making the model more sensitive to detecting airplane targets in remote sensing images. 
###  Dataset example

<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/9.jpg" alt="ddq_arch" width="700">
</div>

</div>


## Result of experiment
Ablation experiments on the remote sensing aircraft detection frameworks, where ① represents the backbone network (PVTV2), ② represents the transfer mode between decoders (BLTP), ③ represents the Albu data augmentation algorithm (Albu), ④ represents the loss function of the metric space, and ⑤ is the contribution of the dynamic learning rate adjustment framework to improve the aircraft detection performance.
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/11.png" alt="ddq_arch" width="700">
</div>

### Metric space-based loss function
**Comparative analysis of different classification, regularization, and IOU methods on the MDMF dataset.**
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/33.png" alt="ddq_arch" width="700">
</div>

**The contribution of classification (GHMC), regularization (Smooth L1), and IOU (SIOU) techniques in enhancing remote sensing aircraft detection tasks.**
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/44.png" alt="ddq_arch" width="700">
</div>

the effects of classification (GHMC), regularization (Smooth L1), and IOU techniques (SIOU) on enhancing detection performance. In the baseline scenario without any techniques, the AP was 64.7%, and AP50 was 91.7%. When combining GHMC and SIOU, the AP and AP50 increased to 65.4% and 92.3%, respectively. Combining GHMC and Smooth L1 further improved these metrics to 65.6% and 92.6%. The combination of SIOU and Smooth L1 resulted in AP and AP50 values of 65.8% and 92.9%, respectively. When all three techniques were combined, the AP reached a maximum of 66.1%, and AP50 increased to 93.5%, demonstrating the most significant performance enhancement. This indicates that Smooth L1 contributed the most to performance improvement, especially when used in conjunction with other techniques. This is attributed to the contribution ratios of classification, regularization, and IOU being set at 1:5:2, making Smooth L1 regularization predominant in enhancing performance. We also attempted to adjust these ratios, but the results were suboptimal, confirming that the original ratio yielded the best performance.
## Feature enhancementh
To enhance the generalization of our model, we employed the popular Albu library  to increase the diversity of our training samples. This library offers a rich array of data augmentation techniques, including over 70 different structured image transformation methods. By incorporating variations such as rotation, cropping, flipping, and scaling, we can reduce the risk of model overfitting. Furthermore, through advanced techniques such as brightness and contrast adjustments, noise injection, and color transformations, the model can adapt to various lighting and meteorological conditions. Additionally, specialized enhancements like elastic transformations and grid distortion improve the accuracy of object detection by simulating different perspectives and scale changes. Our objective is to expand the application of our aircraft detection algorithm to a broader range of training scenarios. The related techniques employed include:

1）RGB Shift: Randomly alters the order of the image color channels, enhancing the model's adaptability to color variations.

2）Shift Scale Rotate: Applies affine transformations to images, including translation, scaling, and rotation, enhancing the model's robustness to changes in target position and scale.
	
3）Hue Saturation Value: Randomly adjusts the hue and saturation of images, helping the model better adapt to different environments and scenes.

4）	Random Brightness Contrast: Randomly adjusts the brightness and contrast of images to improve the model's adaptability to various lighting conditions.

5）Channel Shuffle: Randomly shuffles the RGB channels of images, adjusting the color distribution to help the model better recognize different color display methods.

6）Elastic Transform: Simulates the effect of images being distorted by elastic materials, aiding the model in learning to recognize non-rigid deformations that may be encountered in practical applications.

7）Grid Distortion: Distorts images by periodically or randomly moving grid points, simulating camera lens distortions or other visual distortions, and training the model to identify objects in deformed visual inputs.

Without enhancement preprocessing, the small airplane at position 'w' in  (b) was not detected, whereas it was successfully located in  (a). Additionally, in  (b), positions 'v' and 'u' mistakenly identified the shadow of the airplane as the aircraft itself, whereas  these objects was accurately detected in  (a). It indicated that applying feature enhancement on objects with the fine-scale features significantly improves the model's detection performance.
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/8.jpg" alt="ddq_arch" width="700">
</div>

## Comparisons of different datasets
To verify the generality and robustness of our methods, we evaluated the detection performance of the DIAD-DETR model compared to the baseline model across four datasets: LEVIR, DIOR, MDMF, and B-MDMF.  the DIAD-DETR model improved its AP from 55.7% on the baseline to 57.6% on LEVIR, from 68.9% to 71.5% on DIOR, from 64.7% to 67.5% on MDMF, and from 65.1% to 67.8% on B-MDMF. Moreover, the model demonstrated exceptional performance in detecting small objects, particularly on the MDMF dataset, where the APS significantly increased from 48.5% to 53.1%. These results indicated that the DIAD-DETR model not only achieved higher accuracy in target detection tasks across various scenarios but also exceled in identifying small targets.
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/55.png" alt="ddq_arch" width="700">
</div>



## Comparisons of models
The detection performance of various single-stage, two-stage, and end-to-end networks was compared on the MDMF aircraft dataset.among the two-stage networks, Dynamic R-CNN stands out with an AP of 65.1% and an APS of 51.2%, despite its higher parameter count and computational cost. In the single-stage networks, YOLO-based models such as YOLOV5-X and YOLOV8-X achieved a good balance between high performance and computational efficiency, with AP scores of 64.6% and 65.3%, respectively. However, their small object detection capability was relatively weak, with APS scores of only 47.9% and 48.3%. Additionally, end-to-end networks like DETR and DINO-DETR maintained lower parameter counts and computational costs while achieving AP and APS scores above 64% and 49%, respectively, reflecting a good performance-efficiency balance. Our model achieved the best performance across all metrics with lower parameter magnitudes  and computational costs, reaching an AP of 67.5% and an APS of 53.1%, demonstrating outstanding overall detection capability and small object recognition performance. 
<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/22.png" alt="ddq_arch" width="700">
</div>

**Comparison of detection results using different models on the MDMF dataset. Notably, models marked with  the right superscripts of the asterisk (*) and the hash (#)belong to the two-stage and the end-to-end networks, respectively, while those without any notation indicate single-stage networks.**

we evaluated the performance of the highest precision single-stage, two-stage, and end-to-end networks on a remote sensing aircraft image in a complex scene. The results show that although YOLOV9-E has high detection precision, its ability to detect small targets is still limited, as it missed small targets at the locations 'u' and 'v'. Dynamic R-CNN demonstrated strong detection capabilities but also failed to detect the small target at location 'u', misidentified a building at location 'z' as an aircraft, and missed the aircraft at location 'w' due to interference. DINO-DETR also exhibited high detection precision but made a false detection at location 'z' in the complex scene and missed the aircraft at location 'w' under environmental interference, as well as the small target at location 'u'. In contrast, our model, specifically designed for remote sensing aircraft, successfully detected all targets, further highlighting its superiority to discovering the small targets.

<div align="center">
<img src="https://github.com/lcurryh/ncurryh/blob/main/Main%20image/7.jpg" alt="ddq_arch" width="700">
</div>






  
