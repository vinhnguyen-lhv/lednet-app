from distutils.log import error
import os
import re
from turtle import down
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY

from inference_lednet import check_image_size


def lednet_inference(img, model="lednet"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------ set up LEDNet network -------------------
    down_factor = 8
    net = ARCH_REGISTRY.get("LEDNet")(channels=[32, 64, 128, 128], connection=False).to(
        device
    )
    ckpt_path = "weights/lednet.pth"
    # ckpt_path = 'weights/lednet_retrain_500000.pth'
    checkpoint = torch.load(
        "weights/lednet.pth", map_location=device, weights_only=True
    )["params"]
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start to processing ---------------------
    # prepare data
    img_t = img2tensor(img / 255.0, bgr2rgb=True, float32=True)

    # without [-1,1] normalization in lednet model (paper version)
    if not model == "lednet":
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

    img_t = img_t.unsqueeze(0).to(device)

    # lednet inference
    with torch.no_grad():
        # check_image_size
        H, W = img_t.shape[2:]
        img_t = check_image_size(img_t, down_factor)
        output_t = net(img_t)
        output_t = output_t[:, :, :H, :W]

        if model == "lednet":
            output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
        else:
            output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

    del output_t

    output = output.astype("uint8")
    return output
