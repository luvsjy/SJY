import segyio
import numpy as np
import os
from PIL import Image

class SeismicDataset:
    def __init__(self, sgy_file, output_dir, block_size=(256, 256), num_npy_files=None):
        self.sgy_file = sgy_file
        self.output_dir = output_dir
        self.npy_dir = os.path.join(output_dir, "Train")
        self.img_dir = os.path.join(output_dir, "img_blocks")
        self.mask_img_dir = os.path.join(output_dir, "mask_images")
        self.mask_npy_dir = os.path.join(output_dir, "mask_npy")
        self.test_mask_class_dir = os.path.join(output_dir, "test_mask_class")
        self.test_mask_img_class_dir = os.path.join(output_dir, "test_mask_img_class")
        self.block_size = block_size
        self.num_npy_files = num_npy_files  # 控制生成的npy文件数量
        self.data = self.load_sgy_file()

    def load_sgy_file(self):
        with segyio.open(self.sgy_file, "r", ignore_geometry=True) as f:
            data = np.array(f.trace.raw[:])
        print(f"数据形状: {data.shape} (道数, 每道采样点数)")
        return data

    def split_data_into_blocks(self):
        trace_samples = self.data.shape[1]
        num_traces = self.data.shape[0]

        blocks = []
        for start in range(0, num_traces, self.block_size[0]):
            block_data = self.data[start:start + self.block_size[0], :]
            for col in range(0, trace_samples, self.block_size[1]):
                if col + self.block_size[1] <= trace_samples:
                    block = block_data[:, col:col + self.block_size[1]]
                    blocks.append(block)
        return blocks

    def generate_random_mask(self, block, mask_ratio):
        num_columns_to_mask = int(mask_ratio * block.shape[1])
        mask_indices = np.random.choice(block.shape[1], num_columns_to_mask, replace=False)
        mask = np.zeros(block.shape, dtype=bool)
        mask[:, mask_indices] = True
        return mask

    def save_blocks_as_npy_and_images(self, blocks):
        if not os.path.exists(self.npy_dir):
            os.makedirs(self.npy_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
        if not os.path.exists(self.mask_img_dir):
            os.makedirs(self.mask_img_dir)
        if not os.path.exists(self.mask_npy_dir):
            os.makedirs(self.mask_npy_dir)
        if not os.path.exists(self.test_mask_class_dir):
            os.makedirs(self.test_mask_class_dir)
        if not os.path.exists(self.test_mask_img_class_dir):
            os.makedirs(self.test_mask_img_class_dir)

        mask_ratios = [0.1 * i for i in range(1, 6)]  # 掩码比例从10%到50%

        npy_count = 0  # 控制生成的 npy 文件数量
        for i, block in enumerate(blocks):
            if self.num_npy_files is not None and npy_count >= self.num_npy_files:
                break  # 达到指定的 npy 文件数量时停止

            npy_path = os.path.join(self.npy_dir, f"block_{i}.npy")
            np.save(npy_path, block)

            min_val = np.min(block)
            max_val = np.max(block)
            if max_val > min_val:
                normalized_block = ((block - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                normalized_block = np.full(block.shape, 128, dtype=np.uint8)

            img = Image.fromarray(normalized_block)
            img_path = os.path.join(self.img_dir, f"block_{i}.png")
            img.save(img_path)

            mask_ratio = mask_ratios[np.random.randint(0, len(mask_ratios))]
            mask = self.generate_random_mask(block, mask_ratio)

            mask_npy_path = os.path.join(self.mask_npy_dir, f"block_{i}_mask.npy")
            np.save(mask_npy_path, mask)

            masked_img = Image.fromarray(mask.astype(np.uint8) * 255)
            masked_img_path = os.path.join(self.mask_img_dir, f"block_{i}_mask.png")
            masked_img.save(masked_img_path)

            mask_class_dir = os.path.join(self.test_mask_class_dir, f"mask_{int(mask_ratio * 100)}%")
            if not os.path.exists(mask_class_dir):
                os.makedirs(mask_class_dir)

            test_msk_path = os.path.join(mask_class_dir, f"block_{i}_mask.npy")
            np.save(test_msk_path, mask)

            mask_img_class_dir = os.path.join(self.test_mask_img_class_dir, f"mask_{int(mask_ratio * 100)}%")
            if not os.path.exists(mask_img_class_dir):
                os.makedirs(mask_img_class_dir)

            test_msk_img_path = os.path.join(mask_img_class_dir, f"block_{i}_mask.png")
            masked_img.save(test_msk_img_path)

            npy_count += 1

    def process_and_save(self):
        print("开始切分数据...")
        blocks = self.split_data_into_blocks()
        print(f"数据切分完成，共生成 {len(blocks)} 个小块")
        print("开始保存 .npy 文件和图片...")
        self.save_blocks_as_npy_and_images(blocks)
        print("保存完成！")


# 使用示例
sgy_file = r"D:\SoftWare\ChengxuXiangmu\dataset\sesimic_data\shots0001_0200.segy\1_1"
output_dir = r"D:\SoftWare\ChengxuXiangmu\LG_Data\SeismicData_shot_Train"
num_npy_files = 5000  # 假设我们只想生成100个npy文件

dataset = SeismicDataset(sgy_file, output_dir, num_npy_files=num_npy_files)
dataset.process_and_save()
