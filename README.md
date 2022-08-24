# ESRGAN-pytorch

This repository implements a deep-running model for super resolution.
 Super resolution allows you to pass low resolution images to CNN and restore them to high resolution. 
 We refer to the following article.  
 [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://arxiv.org/abs/1809.00219)  
 
 ## architecture
 [Overall Architecture]
 ![ESRGAN architecture](./image/architecture.PNG)  
 [Basic block]  
 ![BasicBlock](./image/basicBlock.PNG)
 
 ### Test Code
 ```bash
python test.py --lr_dir LR_DIR --sr_dir SR_DIR
```
 
 ## Prepare dataset
 ### Use Flicker2K and DIV2K
```bash
cd datasets
python prepare_datasets.py
cd ..
```
### custom dataset
Make dataset like this; size of hr is 128x128 ans lr is 32x32
```
datasets/
    hr/
        0001.png
        sdf.png
        0002.png
        0003.png
        0004.png
        ...
    lr/
        0001.png
        sdf.png
        0002.png
        0003.png
        0004.png
        ...
```

## how to train
run main file
```bash
python main.py --is_perceptual_oriented True --num_epoch=10
python main.py --is_perceptual_oriented False --epoch=10
```

## Sample
we are in training on this code and train is not complete yet.
this is intermediate result.

<img src="image/lr.png" width="200">
<img src="image/hr.png" width="800">


第一步：使用data_tool内Fisheye_simu工具，从高清图仿真出模仿鱼眼矫正的退化图，高清图建议下载开源的DIV2K等数据集合
第二步：使用data_tool内coner_crop工具，同时切分高低清图四角区域形成配对
第三步：开启训练，如上述说明，使用FishSR_project内的main.py脚本，详细内容请参阅论文