# CFENet: Boosting Few-Shot Semantic Segmentation with Complementary Feature-Enhanced Network

This repo contains the code for our  [paper]() "*CFENet: Boosting Few-Shot Semantic Segmentation with Complementary Feature-Enhanced Network*" by Yun Wang, Lu Zhu，Yuan yuan Liu. 

> **Abstract:** *Few-shot semantic segmentation aims to extract information from few annotated support images to segment unknown class objects in the query image. Traditional algorithms may produce errors and insufficient feature extraction using multi-layer cosine similarity to extract correlation information, due to the large differences in appearance and posture between novel class objects, as well as the similarity in texture and shape among different categories. To address the above issue, we propose a Complementary Feature-Enhanced Network (CFENet). Specifically, we propose a correlation complementary extraction module (CCEM) to facilitate long-range information interaction between query features and support features in the intermediate layer, which contains detailed information. The generated multi-channel correlation information complements the prior information obtained through cosine similarity comparison. In addition, we propose a multi-branch feature enhancement module to capture long-range dependencies in aggregated features which are composed of prior association information and query features. The module effectively suppresses noise in the aggregated features and complementarily enhances the query target object feature from both global and local perspectives. Experiments of the network on the PASCAL-5^<sup>i</sup> and COCO-20^<sup>i</sup> datasets validate the effectiveness of our proposed method. *


## Get Started 
### Dependencies

- Python 3.8
- PyTorch 1.7.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14

### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

- We follow the lists generation as BAM and upload the Data lists into './lists'
- Run `util/get_mulway_base_data.py` to generate base annotations for **stage1**.

### Models

- Download our pre-trained backbones from [here](https://pan.baidu.com/s/1IIA52GjxHrsgghpPeDYjvQ?pwd=9njp) (password：9njp) and put them into the `CFENet/initmodel` directory. 
- Download our trained base learners from [here](https://pan.baidu.com/s/1MACa5yf2dL53pzOwf08D6Q?pwd=y5ob)(password: y5ob) and put them under `initmodel/PSPNet`. 
- We provide all trained CFENet [models](https://pan.baidu.com/s/1WPK6EoS2sfnZkWsbb7bIrw?pwd=octe)(password: octe) of COCO dataset for your convenience. For Pascal dataset, you can directly retrain the models since the traing time is not long. _Backbone: ResNet50 & ResNet101; Dataset: PASCAL-5<sup>i</sup> & COCO-20<sup>i</sup>; Setting: 1-shot & 5-shot_.

-Please note that to reproduct the results we reported in our paper, you can just download the corresponding models and run test script. But we still highly recommond you to retrain the model. Please note that the experimental results may vary due to different environments and settings. 

### Scripts

- Change configuration via the `.yaml` files in `CFENet/config`, then run the `tain.py` and 'test.py' for training and testing.

- **Stage1** *Pre-training*
  
  Train the base learner within the standard learning paradigm.
  
  ```
  sh train_base.sh
  ```

- **Stage2** *Meta-training*
  
  Train the meta learner and ensemble module within the meta-learning paradigm. 
  
  ```
  sh train.sh
  ```

- **Stage3** *Meta-testing*
  
  Test the proposed model under the standard few-shot setting. 
  
  ```
  sh test.sh
  ```


### To-Do List

- [x] Support different backbones
- [x] Support various annotations for training/testing
- [x] Multi-GPU training

## References

This repository owes its existence to the exceptional contributions of other projects:
[PFENet](https://github.com/dvlab-research/PFENet)
[BAM](https://github.com/chunbolang/BAM)
[MSANet](https://github.com/AIVResearch/MSANet)

Many thanks for their great work!
