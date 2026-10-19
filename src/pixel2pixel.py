
from .bandpass import bandpass_filter
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
from torch.optim.lr_scheduler import MultiStepLR
from torch import jit

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

from tqdm import tqdm
import os

from obspy import read
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch

import einops
from einops import rearrange

from tqdm import tqdm

import time


device = "cuda:0"

WINDOW_SIZE = 40
PATCH_SIZE = 7


NUM_NEIGHBORS = 100
loss_type = 'L2'
argsmm = 20



# -------------------------------
class DenoiseNet(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(DenoiseNet, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)
        self._initialize_weights()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv4(x))
        x = self.act(self.conv5(x))
        x = self.act(self.conv6(x))
        x = self.conv3(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


def mse_loss(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    return nn.MSELoss()(gt, pred)


loss_f = nn.L1Loss() if loss_type == 'L1' else nn.MSELoss()


def loss_func(model,img1, img2, loss_f=nn.MSELoss()):
    pred1 = model(img1)
    loss = loss_f(img2, pred1)
    return loss


# -------------------------------
def train(model, optimizer, img_bank):
    N, H, W, C = img_bank.shape

    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img1 = torch.gather(img_bank, 0, index1_exp)  # Shape: (1, H, W, C)
    img1 = img1.permute(0, 3, 1, 2)  # (1, C, H, W)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    img2 = torch.gather(img_bank, 0, index2_exp)
    img2 = img2.permute(0, 3, 1, 2)

    loss = loss_func(model, img1, img2, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, noisy_img, clean_img):
    with torch.no_grad():
        pred = torch.clamp(model(noisy_img), 0, 1)
        mse_val = mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val)
    return psnr, pred



def apply_border_division(matrix, border_size1, border_size2):   
    rows, cols = matrix.shape   
    processed_matrix = matrix.copy()  
    processed_matrix[:border_size1, border_size2:-border_size2] /= 10 
    processed_matrix[-border_size1:, border_size2:-border_size2] /= 10  
    return processed_matrix  



def construct_pixel_bank(img):
    # 固定参数
    WINDOW_SIZE = 40
    PATCH_SIZE = 7
    NUM_NEIGHBORS = 100
    # blk_sz = 128
    blk_sz = 64
    # blk_sz = 32
    # 关键：设置小批次大小，根据GPU显存调整（如4/8/16，显存越小批次越小）
    BATCH_PER_GPU = 4  

    # 设备配置：优先GPU，强制使用float16（比bfloat16更省显存）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    img = img.to(device, dtype=dtype)

    # 预计算固定参数
    pad_sz = WINDOW_SIZE // 2 + PATCH_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    is_window_size_even = (WINDOW_SIZE % 2 == 0)
    H_org, W_org = img.shape[-2], img.shape[-1]

    # 关闭梯度计算 + 禁止autograd追踪（进一步省显存）
    with torch.no_grad():
        # S1：图像填充与unfold（精简形状，避免冗余维度）
        img_pad = F.pad(img, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=PATCH_SIZE, padding=0, stride=1)
        H_new = H_org + WINDOW_SIZE
        W_new = W_org + WINDOW_SIZE
        img_unfold = rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_new, w=W_new)

        # S2：计算图像块总数，生成块索引
        num_blk_h = H_org // blk_sz
        num_blk_w = W_org // blk_sz
        total_blks = num_blk_h * num_blk_w
        blk_indices = list(range(total_blks))

        # S3：分批次处理图像块（核心显存优化）
        topk_list = []
        for idx in range(0, total_blks, BATCH_PER_GPU):
            # 取当前批次的块索引
            batch_indices = blk_indices[idx:idx + BATCH_PER_GPU]
            batch_topk = []

            for blk_idx in batch_indices:
                # 计算当前块的坐标（避免网格张量占用显存）
                blk_j = blk_idx // num_blk_w
                blk_i = blk_idx % num_blk_w
                start_h = blk_j * blk_sz
                end_h = (blk_j + 1) * blk_sz + WINDOW_SIZE
                start_w = blk_i * blk_sz
                end_w = (blk_i + 1) * blk_sz + WINDOW_SIZE

                # 提取子图像块 + 裁剪冗余行/列（立即裁剪，不存冗余数据）
                sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
                if is_window_size_even:
                    sub_img_uf = sub_img_uf[..., :-1, :-1]  # 裁剪后再处理，省显存

                # S4：Unfold补丁窗口（精简形状，减少维度嵌套）
                patch_windows = F.unfold(sub_img_uf, kernel_size=WINDOW_SIZE, padding=0, stride=1)
                patch_windows = rearrange(
                    patch_windows,
                    'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                    h=blk_sz, w=blk_sz
                )

                # S5：提取中心区域（直接切片，避免冗余维度扩展）
                img_center = rearrange(
                    sub_img_uf,
                    'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE,
                    h=sub_img_uf.shape[-2], w=sub_img_uf.shape[-1]
                )
                img_center = img_center[..., center_offset:center_offset+blk_sz, center_offset:center_offset+blk_sz]

                # S6：计算L2距离 + Top-K（裁剪距离张量维度，减少排序显存占用）
                l2_dis = torch.sum((img_center - patch_windows) ** 2, dim=1, keepdim=False)
                # 仅保留前NUM_NEIGHBORS，减少排序开销
                _, sort_indices = torch.topk(l2_dis, k=NUM_NEIGHBORS, largest=False, sorted=True, dim=-3)

                # S7：提取Top-K补丁（精简形状，避免大张量重组）
                patch_windows_reshape = rearrange(
                    patch_windows,
                    'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                    k1=PATCH_SIZE, k2=PATCH_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
                )
                patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
                
                # 索引扩展：使用expand（不复制数据），避免repeat
                sort_indices_exp = sort_indices.unsqueeze(1).expand(-1, patch_center.shape[1], -1, -1, -1)
                topk = torch.gather(patch_center, dim=-3, index=sort_indices_exp)
                batch_topk.append(topk.cpu())  # 立即迁移到CPU，释放GPU显存

                # 关键：删除当前块的中间张量，释放GPU显存
                del sub_img_uf, patch_windows, img_center, l2_dis, sort_indices, patch_center
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # 合并当前批次结果
            batch_topk = torch.cat(batch_topk, dim=0)
            topk_list.append(batch_topk)
            del batch_topk  # 释放批次中间张量

        # 合并所有批次结果
        topk = torch.cat(topk_list, dim=0)
        topk = rearrange(
            topk,
            '(w2 w1) c k h w -> k c (w2 h) (w1 w)',
            w1=num_blk_w, w2=num_blk_h
        )
        topk = topk.permute(2, 3, 0, 1)
        return topk




def pixel2pixel(data):
    matrix = data
    matH,matW = data.shape

    if matH%64==0 and matW%64==0:
        flagResize = False
    else:
        flagResize = True

    column_averages = np.mean(matrix, axis=0)
    matrix_std = np.std(matrix, axis=0)
    standard_matrix = (matrix - column_averages) / matrix_std

    print(matrix_std.shape)

    device = torch.device("cuda")
    # bandpass_filter
    print("Start to filter...")
    fs = 1000
    lowcut = 30.0  
    highcut = 100.0 
    filtered_matrix = bandpass_filter(standard_matrix, lowcut, highcut, fs)
    threshold = np.std(filtered_matrix) * 2.0
    filtered_matrix[np.abs(filtered_matrix) < threshold] /= 5

    border_size1 = 20 
    border_size2 = 50
    filtered_matrix = apply_border_division(filtered_matrix, border_size1, border_size2)

    resize_image = torch.from_numpy(filtered_matrix).float()[None, None, :, :]  

    print("resize_image:",resize_image.shape)

    if flagResize:
        resize_image = F.interpolate(
            resize_image,
            size=(1024, 1024),
            mode='bilinear',
            align_corners=False
        )

    # max_epoch = 3000
    max_epoch = 100
    lr = 0.001
    
    st_time = time.time()
    img_bank_arr = construct_pixel_bank(resize_image).numpy()
    ed_time = time.time()
    print(ed_time-st_time)

    if img_bank_arr.ndim == 3:
        img_bank_arr = np.expand_dims(img_bank_arr, axis=1)
    # Transpose to (k, H, W, c)
    img_bank = img_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
    # Use only the first mm banks for training
    img_bank = img_bank[:argsmm]
    img_bank = torch.from_numpy(img_bank).to(device)

    # Use the first bank as the noisy input (reshaped to (1, C, H, W))
    noisy_img = img_bank[0].unsqueeze(0).permute(0, 3, 1, 2)

    # n_chan = clean_img_tensor.shape[1]
    n_chan = 1
    model = DenoiseNet(n_chan).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # scheduler = MultiStepLR(optimizer, milestones=[1500, 2000, 2500], gamma=0.5)
    scheduler = MultiStepLR(optimizer, milestones=[int(0.5*max_epoch), int(0.66*max_epoch), int(0.87*max_epoch)], gamma=0.5)

    for epoch in tqdm(range(max_epoch)):
        train(model, optimizer, img_bank)
        scheduler.step()

    # PSNR, out_img = test(model, noisy_img, clean_img_tensor)
    with torch.no_grad():
        # out_img = torch.clamp(model(noisy_img), 0, 1)
        out_img = model(noisy_img)
    

    if flagResize:
        out_img = F.interpolate(
            out_img,
            size=(matH,matW),
            mode='bilinear',
            align_corners=False
        )

    # out_img_pil = to_pil_image(out_img.squeeze(0))
    out_img_pil = out_img.squeeze(0).squeeze(0).cpu().numpy()
    print(out_img_pil.shape)
    out_img_pil = out_img_pil * matrix_std
    
    return out_img_pil
    # return 0


if __name__ == '__main__':
    example1 = os.path.join("/home/user/workCodes/DAS_Denoise/MethodsCompare/data/","BPT1_UTC_20200117_013019.232.mseed")
    st_raw = read(example1)  

    st_raw_array = np.zeros((st_raw[0].stats.npts, len(st_raw)))  #(2048,986)
    for tr_no in range(len(st_raw)):
        st_raw_array[:,tr_no] = st_raw[tr_no].data

    # st_raw_array_resize = st_raw_array[:1000,:]

    # st_raw_array_resize = st_raw_array[:256,:256]
    start_idx = 180
    start_idy = 250
    wlen = 512
    st_raw_array_resize = st_raw_array[start_idy:start_idy+wlen,start_idx:start_idx+wlen]

    matrix = st_raw_array_resize

    curr_method = pixel2pixel
    denoised_matrix = curr_method(matrix)

    print(denoised_matrix.min(),denoised_matrix.max())
    
    plt.rcParams["figure.figsize"] = (9,9)
    plt.title("pixel2pixel",fontsize=28)

    plt.imshow(denoised_matrix, vmin=-250, vmax=250, cmap="seismic", aspect="auto", extent=[0,len(st_raw),1,0])

    # plt.imshow(denoised_matrix, cmap="seismic", aspect="auto", extent=[0,len(st_raw),1,0])
    plt.xlabel('DAS channel', fontsize=20)
    plt.ylabel('Time(s)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.colorbar()

    plt.show()   

