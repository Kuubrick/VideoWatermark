# %%
import subprocess
import os

result = subprocess.run(
    'bash -c "source /etc/network_turbo && env | grep proxy"',
    shell=True,
    capture_output=True,
    text=True,
)
output = result.stdout
for line in output.splitlines():
    if "=" in line:
        var, value = line.split("=", 1)
        os.environ[var] = value

# %%
import argparse
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import torch
from diffusers import DPMSolverMultistepScheduler
from tree_ring.inverse_stable_diffusion import InversableStableDiffusionPipeline
import open_clip
from tree_ring.optim_utils import *
from tree_ring.io_utils import *
import re

device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_id(string):
    match = re.search(r"sent(\d+)_frames", string)
    if match:
        return match.group(1)
    else:
        return None


# %%
def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_static:16.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_static:16.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_static:16.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_static:16.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(
            img1.size,
            scale=(args.crop_scale, args.crop_scale),
            ratio=(args.crop_ratio, args.crop_ratio),
        )(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(
            img2.size,
            scale=(args.crop_scale, args.crop_scale),
            ratio=(args.crop_ratio, args.crop_ratio),
        )(img2)

    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


# %%
parser = argparse.ArgumentParser(description="diffusion watermark")
parser.add_argument(
    "--model_id",
    default="/root/.cache/huggingface/diffusers/models--stabilityai--stable-diffusion-2-1-base/snapshots/5ede9e4bf3e3fd1cb0ef2f7a3fff13ee514fdf06/",
)
parser.add_argument("--test_num_inference_steps", default=None, type=int)
parser.add_argument("--num_inference_steps", default=50, type=int)

# watermark
parser.add_argument("--w_mask_shape", default="circle")

parser.add_argument("--w_measurement", default="l1_complex")
parser.add_argument("--w_radius", default=16)
parser.add_argument("--w_channel", default=0, type=int)

parser.add_argument(
    "--orig_img_no_w_path",
    default="../datas/videos/text2video-zero/static_radius:16/8frames_uniform",
    type=str,
)
parser.add_argument(
    "--orig_img_w_path",
    default="../datas/videos/text2video-zero/static_radius:16/8frames_uniform_w",
    type=str,
)

# for image distortion


parser.add_argument("--r_degree", default=None, type=float)
parser.add_argument("--jpeg_ratio", default=None, type=int)
# parser.add_argument('--crop_scale', default=0.75, type=float)
# parser.add_argument('--crop_ratio', default=0.75, type=float)
parser.add_argument("--crop_scale", default=None, type=float)
parser.add_argument("--crop_ratio", default=None, type=float)
parser.add_argument("--gaussian_blur_r", default=None, type=int)
parser.add_argument("--gaussian_std", default=None, type=float)
parser.add_argument("--brightness_factor", default=None, type=float)
parser.add_argument("--rand_aug", default=0, type=int)
args = parser.parse_args([])

if args.test_num_inference_steps is None:
    args.test_num_inference_steps = args.num_inference_steps

# %%
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    args.model_id, subfolder="scheduler"
)
pipe = InversableStableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
    revision="fp16",
)
pipe = pipe.to(device)


# %%
def get_watermarking_mask(init_latents_w, w_radius, device, w_channel=0):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)
    np_mask = circle_mask(init_latents_w.shape[-1], r=w_radius)
    torch_mask = torch.tensor(np_mask).to(device)

    if w_channel == -1:
        # all channels
        watermarking_mask[:, :] = torch_mask
    else:
        watermarking_mask[:, w_channel] = torch_mask
    return watermarking_mask


# %%
tester_prompt = ""  # assume at the detection time, the original prompt is unknown
text_embeddings = pipe.get_text_embedding(tester_prompt)

# ground-truth patch
gt_patch = torch.load("./tree_ring/key.pt").to(dtype=torch.complex32)
init_latents_w = pipe.get_random_latents()
watermarking_mask = get_watermarking_mask(init_latents_w, 16, device)
no_w_metrics = []
w_metrics = []

# %%
# for idx_video in range(618):
#     print(f'---------------------idx_video:{idx_video}----------------------------')
#     for idx_frame in range(8):
#         frame_path_no_w = f'{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg'
#         frame_path_w = f'{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg'

#         img_no_w = Image.open(frame_path_no_w)
#         img_w = Image.open(frame_path_w)

#         img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

#         img_no_w = transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
#         img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)


#         image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
#         image_latents_w = pipe.get_image_latents(img_w, sample=False)

#         reversed_latents_no_w = pipe.forward_diffusion(
#         latents=image_latents_no_w,
#         text_embeddings=text_embeddings,
#         guidance_scale=1,
#         num_inference_steps=50,
#         )

#         reversed_latents_w = pipe.forward_diffusion(
#         latents=image_latents_w,
#         text_embeddings=text_embeddings,
#         guidance_scale=1,
#         num_inference_steps=50,
#         )
#         no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)
#         no_w_metrics.append(-no_w_metric)
#         w_metrics.append(-w_metric)

# # %%
# preds = no_w_metrics +  w_metrics
# t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
# fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
# auc = metrics.auc(fpr, tpr)
# acc = np.max(1 - (fpr + (1 - tpr))/2)
# low = tpr[np.where(fpr<.01)[0][-1]]
# print(acc)
# import matplotlib.pyplot as plt

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# # 添加Accuracy信息到图中的中间偏下位置
# acc_text_pos = (0.5, 0.3)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
# plt.text(acc_text_pos[0], acc_text_pos[1], f'Accuracy = {acc*100:.2f}%', fontsize=12,
#          horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
# plt.legend(loc='lower right')
# plt.savefig('/root/VideoWatermark/img_static:16_no_distortion.png')
# plt.show()

# # %%
# # 分割原始列表，每8个元素为一组
# split_lists_no = [no_w_metrics[i:i+8] for i in range(0, len(no_w_metrics), 8)]

# # 计算每组的平均值
# average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# # 分割原始列表，每8个元素为一组
# split_lists_w = [w_metrics[i:i+8] for i in range(0, len(w_metrics), 8)]

# # 计算每组的平均值
# average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

# preds = average_values_no +  average_values_w
# t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
# fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
# auc = metrics.auc(fpr, tpr)
# acc = np.max(1 - (fpr + (1 - tpr))/2)
# low = tpr[np.where(fpr<.01)[0][-1]]
# print(acc)

# import matplotlib.pyplot as plt

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# # 添加Accuracy信息到图中的中间偏下位置
# acc_text_pos = (0.5, 0.3)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
# plt.text(acc_text_pos[0], acc_text_pos[1], f'Accuracy = {acc*100:.2f}%', fontsize=12,
#          horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
# plt.legend(loc='lower right')
# plt.savefig('/root/VideoWatermark/video_static:16_no_distortion.png')
# plt.show()

# %%
# no_w_metrics=[]
# w_metrics=[]
# args.jpeg_ratio=25
# for idx_video in range(618):
#     print(f'---------------------idx_video:{idx_video}----------------------------')
#     for idx_frame in range(8):
#         frame_path_no_w = f'{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg'
#         frame_path_w = f'{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg'

#         img_no_w = Image.open(frame_path_no_w)
#         img_w = Image.open(frame_path_w)

#         img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

#         img_no_w = transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
#         img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)


#         image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
#         image_latents_w = pipe.get_image_latents(img_w, sample=False)

#         reversed_latents_no_w = pipe.forward_diffusion(
#         latents=image_latents_no_w,
#         text_embeddings=text_embeddings,
#         guidance_scale=1,
#         num_inference_steps=50,
#         )

#         reversed_latents_w = pipe.forward_diffusion(
#         latents=image_latents_w,
#         text_embeddings=text_embeddings,
#         guidance_scale=1,
#         num_inference_steps=50,
#         )
#         no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)
#         no_w_metrics.append(-no_w_metric)
#         w_metrics.append(-w_metric)

# # %%
# preds = no_w_metrics +  w_metrics
# t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
# fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
# auc = metrics.auc(fpr, tpr)
# acc = np.max(1 - (fpr + (1 - tpr))/2)
# low = tpr[np.where(fpr<.01)[0][-1]]
# print(acc)
# import matplotlib.pyplot as plt

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# # 添加Accuracy信息到图中的中间偏下位置
# acc_text_pos = (0.5, 0.3)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
# plt.text(acc_text_pos[0], acc_text_pos[1], f'Accuracy = {acc*100:.2f}%', fontsize=12,
#          horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
# plt.legend(loc='lower right')
# plt.savefig('/root/VideoWatermark/img_static:16_jpeg.png')
# plt.show()

# # %%
# # 分割原始列表，每8个元素为一组
# split_lists_no = [no_w_metrics[i:i+8] for i in range(0, len(no_w_metrics), 8)]

# # 计算每组的平均值
# average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# # 分割原始列表，每8个元素为一组
# split_lists_w = [w_metrics[i:i+8] for i in range(0, len(w_metrics), 8)]

# # 计算每组的平均值
# average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

# preds = average_values_no +  average_values_w
# t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
# fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
# auc = metrics.auc(fpr, tpr)
# acc = np.max(1 - (fpr + (1 - tpr))/2)
# low = tpr[np.where(fpr<.01)[0][-1]]
# print(acc)

# import matplotlib.pyplot as plt

# # Plot ROC curve
# plt.figure(figsize=(8, 8))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.2f}')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate (FPR)')
# plt.ylabel('True Positive Rate (TPR)')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# # 添加Accuracy信息到图中的中间偏下位置
# acc_text_pos = (0.5, 0.3)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
# plt.text(acc_text_pos[0], acc_text_pos[1], f'Accuracy = {acc*100:.2f}%', fontsize=12,
#          horizontalalignment='center', verticalalignment='bottom', transform=plt.gca().transAxes)
# plt.legend(loc='lower right')
# plt.savefig('/root/VideoWatermark/video_static:16_jpeg.png')
# plt.show()

# %%
no_w_metrics = []
w_metrics = []
args.jpeg_ratio = None
args.crop_scale = 0.75
args.crop_ratio = 0.75
for idx_video in range(618):
    print(f"---------------------idx_video:{idx_video}----------------------------")
    for idx_frame in range(8):
        frame_path_no_w = (
            f"{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )
        frame_path_w = (
            f"{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )

        img_no_w = Image.open(frame_path_no_w)
        img_w = Image.open(frame_path_w)

        img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

        img_no_w = (
            transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        )
        img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)

        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )
        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )
        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

# %%
preds = no_w_metrics + w_metrics
t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)

# %%
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/img_static:16_crop.png")
plt.show()

# %%
# 分割原始列表，每8个元素为一组
split_lists_no = [no_w_metrics[i : i + 8] for i in range(0, len(no_w_metrics), 8)]

# 计算每组的平均值
average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# 分割原始列表，每8个元素为一组
split_lists_w = [w_metrics[i : i + 8] for i in range(0, len(w_metrics), 8)]

# 计算每组的平均值
average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

# %%
preds = average_values_no + average_values_w
t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)

# %%
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/video_static:16_crop.png")
plt.show()

# %%
no_w_metrics = []
w_metrics = []
args.jpeg_ratio = None
args.crop_scale = None
args.crop_ratio = None
args.r_degree = 75
for idx_video in range(618):
    print(f"---------------------idx_video:{idx_video}----------------------------")
    for idx_frame in range(8):
        frame_path_no_w = (
            f"{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )
        frame_path_w = (
            f"{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )

        img_no_w = Image.open(frame_path_no_w)
        img_w = Image.open(frame_path_w)

        img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

        img_no_w = (
            transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        )
        img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)

        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )
        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

# %%
preds = no_w_metrics + w_metrics
t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/img_static:16_rotate.png")
plt.show()

# %%
# 分割原始列表，每8个元素为一组
split_lists_no = [no_w_metrics[i : i + 8] for i in range(0, len(no_w_metrics), 8)]

# 计算每组的平均值
average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# 分割原始列表，每8个元素为一组
split_lists_w = [w_metrics[i : i + 8] for i in range(0, len(w_metrics), 8)]

# 计算每组的平均值
average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

preds = average_values_no + average_values_w
t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)

import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/video_static:16_rotate.png")
plt.show()

# %%
no_w_metrics = []
w_metrics = []
args.jpeg_ratio = None
args.crop_scale = None
args.crop_ratio = None
args.r_degree = None
args.gaussian_blur_r = 4
for idx_video in range(618):
    print(f"---------------------idx_video:{idx_video}----------------------------")
    for idx_frame in range(8):
        frame_path_no_w = (
            f"{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )
        frame_path_w = (
            f"{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )

        img_no_w = Image.open(frame_path_no_w)
        img_w = Image.open(frame_path_w)

        img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

        img_no_w = (
            transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        )
        img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)

        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )
        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)


# %%
preds = no_w_metrics + w_metrics
t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/img_static:16_gaussian_blur.png")
plt.show()

# %%
# 分割原始列表，每8个元素为一组
split_lists_no = [no_w_metrics[i : i + 8] for i in range(0, len(no_w_metrics), 8)]

# 计算每组的平均值
average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# 分割原始列表，每8个元素为一组
split_lists_w = [w_metrics[i : i + 8] for i in range(0, len(w_metrics), 8)]

# 计算每组的平均值
average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

preds = average_values_no + average_values_w
t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)

import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/video_static:16_gaussian_blur.png")
plt.show()

# %%
no_w_metrics = []
w_metrics = []
args.jpeg_ratio = None
args.crop_scale = None
args.crop_ratio = None
args.r_degree = None
args.gaussian_blur_r = None
args.gaussian_std = 0.1
for idx_video in range(618):
    print(f"---------------------idx_video:{idx_video}----------------------------")
    for idx_frame in range(8):
        frame_path_no_w = (
            f"{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )
        frame_path_w = (
            f"{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )

        img_no_w = Image.open(frame_path_no_w)
        img_w = Image.open(frame_path_w)

        img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

        img_no_w = (
            transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        )
        img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)

        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )
        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)


# %%
preds = no_w_metrics + w_metrics
t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/img_static:16_gaussian_std.png")
plt.show()

# %%
# 分割原始列表，每8个元素为一组
split_lists_no = [no_w_metrics[i : i + 8] for i in range(0, len(no_w_metrics), 8)]

# 计算每组的平均值
average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# 分割原始列表，每8个元素为一组
split_lists_w = [w_metrics[i : i + 8] for i in range(0, len(w_metrics), 8)]

# 计算每组的平均值
average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

preds = average_values_no + average_values_w
t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)

import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/video_static:16_gaussian_std.png")
plt.show()

# %%
no_w_metrics = []
w_metrics = []
args.jpeg_ratio = None
args.crop_scale = None
args.crop_ratio = None
args.r_degree = None
args.gaussian_blur_r = None
args.gaussian_std = None
args.brightness_factor = 0.1
for idx_video in range(618):
    print(f"---------------------idx_video:{idx_video}----------------------------")
    for idx_frame in range(8):
        frame_path_no_w = (
            f"{args.orig_img_no_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )
        frame_path_w = (
            f"{args.orig_img_w_path}/sent{idx_video}_frames/frame{idx_frame}.jpg"
        )

        img_no_w = Image.open(frame_path_no_w)
        img_w = Image.open(frame_path_w)

        img_no_w, img_w = image_distortion(img_no_w, img_w, 42, args)

        img_no_w = (
            transform_img(img_no_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        )
        img_w = transform_img(img_w).unsqueeze(0).to(text_embeddings.dtype).to(device)

        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=50,
        )

        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )
        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)


# %%
preds = no_w_metrics + w_metrics
t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)
import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/img_static:16_bright.png")
plt.show()

# %%
# 分割原始列表，每8个元素为一组
split_lists_no = [no_w_metrics[i : i + 8] for i in range(0, len(no_w_metrics), 8)]

# 计算每组的平均值
average_values_no = [sum(sub_list) / len(sub_list) for sub_list in split_lists_no]

# 分割原始列表，每8个元素为一组
split_lists_w = [w_metrics[i : i + 8] for i in range(0, len(w_metrics), 8)]

# 计算每组的平均值
average_values_w = [sum(sub_list) / len(sub_list) for sub_list in split_lists_w]

preds = average_values_no + average_values_w
t_labels = [0] * len(average_values_no) + [1] * len(average_values_w)
fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
auc = metrics.auc(fpr, tpr)
acc = np.max(1 - (fpr + (1 - tpr)) / 2)
low = tpr[np.where(fpr < 0.01)[0][-1]]
print(acc)

import matplotlib.pyplot as plt

# Plot ROC curve
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Receiver Operating Characteristic (ROC) Curve")
# 添加Accuracy信息到图中的中间偏下位置
acc_text_pos = (
    0.5,
    0.3,
)  # 这是文本标签的新位置，0.5代表水平居中，0.3代表垂直位置在中央偏下
plt.text(
    acc_text_pos[0],
    acc_text_pos[1],
    f"Accuracy = {acc*100:.2f}%",
    fontsize=12,
    horizontalalignment="center",
    verticalalignment="bottom",
    transform=plt.gca().transAxes,
)
plt.legend(loc="lower right")
plt.savefig("/root/VideoWatermark/video_static:16_bright.png")
plt.show()
