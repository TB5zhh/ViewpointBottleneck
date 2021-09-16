#  **Pointly-supervised 3D Scene Parsing with Viewpoint Bottleneck**

***

Created by Liyi Luo, Beiwen Tian, Hao Zhao and Guyue Zhou from [Institute for AI Industry Research (AIR), Tsinghua University, China](http://air.tsinghua.edu.cn/EN/).

<img src=".\doc\result7.png" style="zoom:70%;" />

<img src=".\doc\result1.png" style="zoom:70%;" />

<img src=".\doc\result2.png" alt="result2" style="zoom:70%;" />

<img src=".\doc\result3.png" alt="result3" style="zoom:70%;" />

<img src=".\doc\result4.png" alt="result4" style="zoom:70%;" />

<img src=".\doc\result5.png" alt="result5" style="zoom:70%;" />

<img src=".\doc\result6.png" alt="result6" style="zoom:70%;" />

## Introduction

------

Semantic understanding of 3D point clouds is important for various robotics applications. Given that point-wise semantic annotation is expensive, in our paper, we address the challenge of learning models with extremely sparse labels. The core problem is how to leverage numerous unlabeled points. 

In this repository, we propose a self-supervised 3D representation learning framework named viewpoint bottleneck. It optimizes a mutual-information based objective, which is applied on point clouds under different viewpoints. A principled analysis shows that viewpoint bottleneck leads to an elegant surrogate loss function that is suitable for large-scale point cloud data. Compared with former arts based upon contrastive learning, viewpoint bottleneck operates on the feature dimension instead of the sample dimension. This paradigm shift has several advantages: It is easy to implement and tune, does not need negative samples and performs better on our goal down-streaming task. We evaluate our method on the public benchmark ScanNet, under the pointly-supervised setting. We achieve the best quantitative results among comparable solutions. Meanwhile we provide an extensive qualitative inspection on various challenging scenes. They demonstrate that our models can produce fairly good scene parsing results for robotics applications. 



## Citation

------

If you find our work useful in your research, please consider citing:

- [Pointly-supervised 3D scene Parsing with Viewpoint Bottleneck]()  [[pdf]()] 

```
operates 
```

## Requirements

------

- Ubuntu 16.04 or higher
- CUDA 10.1 or higher
- pytorch 1.3 or higher
- python 3.6 or higher
- GCC 6 or higher

## Installation

------

You need to install `pytorch` and [`Minkowski Engine`](https://pypi.org/project/MinkowskiEngine/) either with `pip` or with anaconda.

Clone the repository and install the rest of the requirements

```
git clone https://github.com/OPEN-AIR-SUN/ViewpointBottleneck/

cd ViewpointBottlencek

pip install -r requirements.txt
```

### Troubleshooting

Please visit the MinkowskiEngine [issue pages](https://github.com/StanfordVL/MinkowskiEngine/) if you have difficulties installing Minkowski Engine.

## Training and evaluating

------



## Run Pre-trained Model 

------



## Model Zoo

------

| Feature Dimension | Supervision Points 200 | Supervision Points 100 | Supervision Points 50 | Supervision Points 20 |
| :---------------: | :--------------------: | :--------------------: | :-------------------: | :-------------------: |
|      [256]()      |        [68.4]()        |        [66.5]()        |       [63.3]()        |       [56.2]()        |
|      [512]()      |        [68.5]()        |        [66.8]()        |       [63.6]()        |       [57.0]()        |
|     [1024]()      |        [68.4]()        |        [66.5]()        |       [63.7]()        |       [56.3]()        |



## Acknowledgements

------

We want to thank codebase of [SpatioTemporalSegmentation](https://github.com/chrischoy/SpatioTemporalSegmentation). And also thanks to dataset [ScanNet](https://github.com/ScanNet/ScanNet) .

