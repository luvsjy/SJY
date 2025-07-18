# SGNet

 A Global-Local Feature Fusion Network for Randomly Missing Seismic Data Reconstruction ([Paper]()

## Prerequisites
- Python 3.8
- NVIDIA GeForce RTX 4070 
- PyTorch 2.4.1

## Run
1. train the model
```
python train1.py --dataroot D:/SoftWare/ChengxuXiangmu/LG_Data/SeismicData/Train --name celebahq_LGNet --model pix2pixglg --netG1 unet_256 --netG2 resnet_4blocks --netG3 unet256 --netD snpatch --gan_mode lsgan --input_nc 1 --no_dropout --direction AtoB --display_id 0 --gpu_ids 0 --n_epochs 70 --epoch_count 1 --n_epochs_decay 30
```
2. test the model
```
python test_and_save.py --dataroot D:/SoftWare/ChengxuXiangmu/LG_Data/Seismic_xianchang/Test --name celebahq_LGNet --model pix2pixgsg --netG1 unet_256 --netG2 resnet_4blocks --netG3 unet256 --gan_mode nogan --input_nc 1 --no_dropout --direction AtoB --gpu_ids 0 --num_test 1
```

## Download Datasets
We use [BP 2007 synthetic seismic data](https://wiki.seg.org/wiki/2007_BP_Anisotropic_Velocity_Benchmark) , [fild seismic data](https://wiki.seg.org/wiki/2004_BP_velocity_estimation_benchmark_model)

## Pretrained Models
You can download the pre-trained model from [BP 2007 synthetic seismic data](https://drive.google.com/file/d/1ZIXgfW1JrGvlVw8vmWci87dhgOIPapEg/view?usp=drive_link)

## Citation
If you find this useful for your research, please use the following.
