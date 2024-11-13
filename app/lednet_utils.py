from distutils.log import error
import os
from turtle import down
import cv2
import argparse
import glob
import torch
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
import torch.nn.functional as F

from inference_lednet import check_image_size


def lednet_inference(input_path, output_path, model_path):
    # Load the model
    model = torch.load(model_path, map_location="cpu")
    model.eval()

    # Check the image size
    check_image_size(input_path)

    # Read the input image
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img2tensor(img)
    img = normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img = img.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(img)

    # Post-process the output
    output = F.interpolate(
        output, size=(img.size(2), img.size(3)), mode="bilinear", align_corners=False
    )
    output = output.squeeze(0)
    output = tensor2img(output, min_max=(-1, 1))

    # Save the output image
    imwrite(output, output_path)
