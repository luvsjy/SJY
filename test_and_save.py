import os
import shutil
import numpy as np
import torch
import glob
from options.test_options import TestOptions
from models import create_model
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import io
from contextlib import redirect_stdout
from PIL import Image


def calculate_l1(img1, img2):
    return np.mean(np.abs(np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)))


def calculate_snr(img1, img2):
    # # 计算信号（原始图像）的均方根值
    # signal_power = np.mean(np.square(img1))
    #
    # # 计算噪声（差异图像）的均方误差
    # noise = img1 - img2
    # noise_power = np.mean(np.square(noise))
    #
    # # 计算SNR（单位：dB）
    # if noise_power == 0:
    #     return float('inf')  # 如果噪声为零，SNR 无穷大
    #
    # snr = 10 * np.log10(signal_power / noise_power)
    # return snr
    img1_mean = np.mean(img1)
    tmp1 = img - img1_mean
    real_var = np.sum(tmp1 * tmp1)  # 计算真实数据的方差

    noise = img1 - img2
    noise_mean = np.mean(noise)
    tmp2 = noise - noise_mean
    noise_var = np.sum(tmp2 * tmp2)  # 计算噪声数据的方差

    if noise_var == 0 or real_var == 0:
        s = 999.99  # 如果噪声或真实数据的方差为0，返回一个异常的信噪比
    else:
        # 使用 np.log 计算以 10 为底的对数
        s = 10 * np.log10(real_var / noise_var)  # np.log10 直接计算以 10 为底的对数
    return s


def calculate_psnr(img1, img2):
    # 如果是灰度图像，设置 multichannel=False；如果是彩色图像，设置 multichannel=True
    return psnr(img1.squeeze(), img2.squeeze(),data_range=4)

def calculate_ssim(img1, img2):
    # 计算 SSIM
    # 如果是灰度图像，设置 multichannel=False；如果是彩色图像，设置 multichannel=True
    return ssim(img1.squeeze(), img2.squeeze(), channel_axis=1, multichannel=False,data_range=255)


def calculate_lpips(img1, img2, model='alex'):
    # 重定向标准输出，避免LPIPS加载时的提示信息
    f = io.StringIO()
    with redirect_stdout(f):
        lpips_model = lpips.LPIPS(net=model)  # 加载LPIPS模型
    # 将输入的npy格式图像（HxWxC）转换为Tensor
    img1_tensor = torch.from_numpy(img1).unsqueeze(0).float()   # 转换为 (1, C, H, W)
    img2_tensor = torch.from_numpy(img2).unsqueeze(0).float()   # 转换为 (1, C, H, W)
    # 检查是否是灰度图像（单通道）
    if img1_tensor.shape[1] == 1:
        img1_tensor = img1_tensor.repeat(1, 3, 1, 1)  # 复制通道，变为三通道
    if img2_tensor.shape[1] == 1:
        img2_tensor = img2_tensor.repeat(1, 3, 1, 1)  # 复制通道，变为三通道
    # 计算LPIPS距离
    distance = lpips_model(img1_tensor, img2_tensor)

    return distance.item()


def normalize_for_metrics(img):
    img = np.array(img.cpu(), dtype=np.float32)

    min_val = img.min()
    max_val = img.max()
    if max_val - min_val == 0:
        return img

    img_normalized = 2.0*(2.0 * (img - min_val) / (max_val - min_val) - 1)

    return img_normalized

def normalize_for_metrics1(img):
    img = np.array(img.cpu(), dtype=np.float32)

    min_val = img.min()
    max_val = img.max()
    if max_val - min_val == 0:
        return img

    img_normalized = (255 * (img - min_val) / (max_val - min_val)).astype(np.uint8)

    return img_normalized


def load_flist(flist):
    if isinstance(flist, list):
        return flist
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.npy'))
            flist.sort()
            return flist
        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
            except:
                return [flist]
    return []



val_image = "D:/SoftWare/ChengxuXiangmu/LG_Data/Seismic_xianchang/Test"


opt = TestOptions().parse()
opt.num_threads = 0
opt.batch_size = 1
opt.serial_batches = True
opt.no_flip = True
opt.display_id = -1


num_test = opt.num_test


model = create_model(opt)
model.setup(opt)
model.eval()

val_mask_suffix = ['mask_10%', 'mask_20%', 'mask_30%', 'mask_40%', 'mask_50%']
save_dir_suffix = ['010', '1020', '2030', '3040', '4050']


for suffix_idx in range(5):
    val_mask = 'D:/SoftWare/ChengxuXiangmu/LG_Data/Seismic_xianchang/test_mask_class/' + val_mask_suffix[suffix_idx]
    save_dir = os.path.join('D:/SoftWare/ChengxuXiangmu/LG_Data/results_xianchang_all', 'LGNet-' + save_dir_suffix[suffix_idx])

    test_image_flist = load_flist(val_image)
    test_mask_flist = load_flist(val_mask)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(os.path.join(save_dir, 'comp'), exist_ok=True)
    #os.makedirs(os.path.join(save_dir, 'comp_img'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'masked'), exist_ok=True)
    #os.makedirs(os.path.join(save_dir, 'masked_img'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'merged_npy1'), exist_ok=True)
    #os.makedirs(os.path.join(save_dir, 'merged_img1'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'merged_npy2'), exist_ok=True)
    #os.makedirs(os.path.join(save_dir, 'merged_img2'), exist_ok=True)
    psnr1 = []
    psnr_mergedimg1=[]
    psnr_mergedimg2 = []

    l1 = []
    l1_mergedimg1 = []
    l1_mergedimg2 = []

    ssim_vals = []
    ssim_mergedimg1 = []
    ssim_mergedimg2 = []

    lpips_vals = []
    lpips_mergedimg1 = []
    lpips_mergedimg2 = []

    snr=[]
    snr_mergedimg1 = []
    snr_mergedimg2 = []
    mask_num = len(test_mask_flist)



    num_test = min(num_test, len(test_image_flist))



    for idx in range(num_test):
        img = np.load(test_image_flist[idx])
        mask = np.load(test_mask_flist[idx % mask_num])


        images = torch.from_numpy(img).float()
        masks = torch.from_numpy(mask).float()
        if images.ndimension() == 2:
            images = images.unsqueeze(0).unsqueeze(0)
        if masks.ndimension() == 2:
            masks = masks.unsqueeze(0).unsqueeze(0)
        data = {'A': images, 'B': masks, 'A_paths': ''}


        model.set_input(data)
        with torch.no_grad():
            model.forward()


        orig_imgs = model.images
        mask_imgs = model.masked_images1
        merged_images1=model.merged_images1
        merged_images2 = model.merged_images2
        comp_imgs = model.merged_images3



        names = os.path.basename(test_image_flist[idx]).split('.')[0]
        comp_npy_path = os.path.join(save_dir, 'comp', names + '_comp.npy')
        #comp_img_path = os.path.join(save_dir, 'comp_img', names + '_img.png')
        #mask_img_path = os.path.join(save_dir, 'masked_img', names + '_mask_img.png')
        mask_npy_path = os.path.join(save_dir, 'masked', names + '_mask.npy')
        merged_images1_npy_path = os.path.join(save_dir, 'merged_npy1', names + '_merged_images1.npy')
        #merged_images1_img_path = os.path.join(save_dir, 'merged_img1', names + '_merged_images1.png')
        #merged_images2_img_path = os.path.join(save_dir, 'merged_img2', names + '_merged_images1.png')
        merged_images2_npy_path = os.path.join(save_dir, 'merged_npy2', names + '_merged_images2.npy')
        np.save(comp_npy_path, comp_imgs[0].cpu().numpy())  # 保存到本地时，将 tensor 转换为 NumPy 数组
        np.save(mask_npy_path, mask_imgs[0].cpu().numpy())  # 同上
        np.save(merged_images1_npy_path, merged_images1[0].cpu().numpy())  # 保存到本地时，将 tensor 转换为 NumPy 数组
        np.save(merged_images2_npy_path, merged_images2[0].cpu().numpy())


        # 归一化
        orig_imgs_normalized = normalize_for_metrics(orig_imgs[0])
        comp_imgs_normalized = normalize_for_metrics(comp_imgs[0])
        merged_images1_normalized1 = normalize_for_metrics(merged_images1[0])  # 用于保存图片
        merged_images2_normalized1 = normalize_for_metrics(merged_images2[0])

        mask_imgs_normalized = normalize_for_metrics1(mask_imgs[0])
        merged_images1_normalized = normalize_for_metrics1(merged_images1[0])#用于保存图片
        merged_images2_normalized = normalize_for_metrics1(merged_images2[0])

        orig_imgs_normalized1 = normalize_for_metrics1(orig_imgs[0])
        comp_imgs_normalized1 = normalize_for_metrics1(comp_imgs[0])


        #计算指标
        psnr_tmp = calculate_psnr(comp_imgs_normalized,orig_imgs_normalized)
        psnr_mergedimg1_tmp=calculate_psnr(merged_images1_normalized1,orig_imgs_normalized)
        psnr_mergedimg2_tmp = calculate_psnr(merged_images2_normalized1, orig_imgs_normalized)
        psnr1.append(psnr_tmp)
        psnr_mergedimg1.append(psnr_mergedimg1_tmp)
        psnr_mergedimg2.append(psnr_mergedimg2_tmp)

        # l1_tmp = calculate_l1(comp_imgs_normalized,orig_imgs_normalized)
        # l1_mergedimg1_tmp = calculate_l1(merged_images1_normalized1, orig_imgs_normalized)
        # l1_mergedimg2_tmp = calculate_l1(merged_images2_normalized1, orig_imgs_normalized)
        # l1.append(l1_tmp)
        # l1_mergedimg1.append(l1_mergedimg1_tmp)
        # l1_mergedimg2.append(l1_mergedimg2_tmp)

        ssim_tmp = calculate_ssim(comp_imgs_normalized1,orig_imgs_normalized1)
        ssim_mergedimg1_tmp = calculate_ssim(merged_images1_normalized, orig_imgs_normalized1)
        ssim_mergedimg2_tmp = calculate_ssim(merged_images2_normalized, orig_imgs_normalized1)
        ssim_vals.append(ssim_tmp)
        ssim_mergedimg1.append(ssim_mergedimg1_tmp)
        ssim_mergedimg2.append(ssim_mergedimg2_tmp)

        # lpips_tmp = calculate_lpips(comp_imgs_normalized,orig_imgs_normalized)
        # lpips_mergedimg1_tmp = calculate_lpips(merged_images1_normalized1, orig_imgs_normalized)
        # lpips_mergedimg2_tmp = calculate_lpips(merged_images2_normalized1, orig_imgs_normalized)
        # lpips_vals.append(lpips_tmp)
        # lpips_mergedimg1.append(lpips_mergedimg1_tmp)
        # lpips_mergedimg2.append(lpips_mergedimg2_tmp)

        snr_tmp = calculate_snr(comp_imgs_normalized, orig_imgs_normalized)
        snr_mergedimg1_tmp = calculate_snr(merged_images1_normalized1, orig_imgs_normalized)
        snr_mergedimg2_tmp = calculate_snr(merged_images2_normalized1, orig_imgs_normalized)
        snr.append(snr_tmp)
        snr_mergedimg1.append(snr_mergedimg1_tmp)
        snr_mergedimg2.append(snr_mergedimg2_tmp)


        # 保存图片
        # img = Image.fromarray(comp_imgs_normalized1.squeeze())
        # img.save(comp_img_path)
        #
        # img1 = Image.fromarray(mask_imgs_normalized.squeeze())
        # img1.save(mask_img_path)
        #
        # img2 = Image.fromarray(merged_images1_normalized.squeeze())
        # img2.save(merged_images1_img_path)
        #
        # img3 = Image.fromarray(merged_images2_normalized.squeeze())
        # img3.save(merged_images2_img_path)



    avg_psnr = np.mean(np.array(psnr1))
    avg_psnr_mergedimg1=np.mean(np.array(psnr_mergedimg1))
    avg_psnr_mergedimg2 = np.mean(np.array(psnr_mergedimg2))

    # avg_l1 = np.mean(np.array(l1))
    # avg_l1_mergedimg1 = np.mean(np.array(l1_mergedimg1))
    # avg_l1_mergedimg2 = np.mean(np.array(l1_mergedimg2))

    avg_ssim = np.mean(np.array(ssim_vals))
    avg_ssim_mergedimg1 = np.mean(np.array(ssim_mergedimg1))
    avg_ssim_mergedimg2 = np.mean(np.array(ssim_mergedimg2))

    # avg_lpips = np.mean(np.array(lpips_vals))
    # avg_lpips_mergedimg1 = np.mean(np.array(lpips_mergedimg1))
    # avg_lpips_mergedimg2 = np.mean(np.array(lpips_mergedimg2))

    avg_snr = np.mean(np.array(snr))
    avg_snr_mergedimg1 = np.mean(np.array(snr_mergedimg1))
    avg_snr_mergedimg2 = np.mean(np.array(snr_mergedimg2))
    print(f'Finish in {save_dir}')
    print(f'The avg psnr for mask range {val_mask_suffix[suffix_idx]} is {avg_psnr:.4f}')
    print(f'The avg psnr_mergedimg1 for mask range {val_mask_suffix[suffix_idx]} is {avg_psnr_mergedimg1:.4f}')
    print(f'The avg psnr_mergedimg2 for mask range {val_mask_suffix[suffix_idx]} is {avg_psnr_mergedimg2:.4f}')

    # print(f'The avg L1 for mask range {val_mask_suffix[suffix_idx]} is {avg_l1:.4f}')
    # print(f'The avg L1_mergedimg1 for mask range {val_mask_suffix[suffix_idx]} is {avg_l1_mergedimg1:.4f}')
    # print(f'The avg L1_mergedimg2 for mask range {val_mask_suffix[suffix_idx]} is {avg_l1_mergedimg2:.4f}')

    print(f'The avg SSIM for mask range {val_mask_suffix[suffix_idx]} is {avg_ssim:.4f}')
    print(f'The avg SSIM_mergedimg1 for mask range {val_mask_suffix[suffix_idx]} is {avg_ssim_mergedimg1:.4f}')
    print(f'The avg SSIM_mergedimg2 for mask range {val_mask_suffix[suffix_idx]} is {avg_ssim_mergedimg2:.4f}')

    # print(f'The avg LPIPS for mask range {val_mask_suffix[suffix_idx]} is {avg_lpips:.4f}')
    # print(f'The avg LPIPS_mergedimg1 for mask range {val_mask_suffix[suffix_idx]} is {avg_lpips_mergedimg1:.4f}')
    # print(f'The avg LPIPS_mergedimg2 for mask range {val_mask_suffix[suffix_idx]} is {avg_lpips_mergedimg2:.4f}')

    print(f'The avg SNR for mask range {val_mask_suffix[suffix_idx]} is {avg_snr:.4f}')
    print(f'The avg SNR_mergedimg1 for mask range {val_mask_suffix[suffix_idx]} is {avg_snr_mergedimg1:.4f}')
    print(f'The avg SNR_mergedimg2 for mask range {val_mask_suffix[suffix_idx]} is {avg_snr_mergedimg2:.4f}')
