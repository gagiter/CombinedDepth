# CombinedDepth

## Dataset

### NYU

reference to [bts](https://github.com/cogaplex-bts/bts)

### [EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)

### [Zurich Urban Micro Aerial Vehicle Dataset](http://rpg.ifi.uzh.ch/zurichmavdataset.html)

### Many [Slam Dataset](https://sites.google.com/view/awesome-slam-datasets/) that could use and [here](https://github.com/youngguncho/awesome-slam-datasets)

## Compile libwarp

```
python setup build
```
### Windows with pytorch 1.3.0
[issue](https://github.com/pytorch/extension-cpp/issues/37)

## train

python train --data_root /media/disk1.5T/gq15t/dataset/DepthTrain002 --batch_size 8 --model_name global_depth --gpu_id 1 --lr 0.0001 --summary_freq 10 --save_freq 100 --normal_weight 0.0 --plane_weight 0.0 --ref_weight 1.0 --depth_weight 1.0 --rotation_scale 0.25 --translation_scale 5.0 --down_times 4 --occlusion 1 --global_depth 0 --use_number 0 --resume 1
