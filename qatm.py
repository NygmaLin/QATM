from pathlib import Path
import torch
import torchvision
from torchvision import models, transforms, utils
import argparse
from utils import *
import numpy as np
from tqdm import tqdm

# +
# import functions and classes from qatm_pytorch.py
print("import qatm_pytorch.py...")
import ast
import types
import sys
from qatm_pytorch import *

# with open("qatm_pytorch.py") as f:
#        p = ast.parse(f.read())
#
# for node in p.body[:]:
#     if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
#         p.body.remove(node)
#
# module = types.ModuleType("mod")
# code = compile(p, "mod.py", 'exec')
# sys.modules["mod"] = module
# exec(code,  module.__dict__)
#
# from mod import *
# -

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='QATM Pytorch Implementation')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-s', '--sample_image', default=r'E:\AIschool\data_full\data2\data\data\OK_Orgin_Images\OK_Orgin_Images\part3\Image_11.bmp')
    parser.add_argument('-t', '--template_images_dir', default=r'E:\AIschool\data_full\data2\data\data\TC_Images\TC_Images\part3\TC_Images')
    # parser.add_argument('-t', '--template_images_dir', default=r'E:\AIschool\QATM_pytorch_bak\t')
    parser.add_argument('--alpha', type=float, default=25)
    parser.add_argument('--thresh_csv', type=str, default='thresh_template.csv')
    args = parser.parse_args()
    
    template_dir = args.template_images_dir
    image_path = args.sample_image
    dataset = ImageDataset(Path(template_dir), image_path, resize_scale=1/4, thresh_csv='thresh_template.csv')

    save_path = r'E:\AIschool\QATM_pytorch_bak\save_path\part2\show'
    print("define model...")
    model = CreateModel(model=models.vgg19(pretrained=True).features, alpha=args.alpha, use_cuda=args.cuda)
    print("calculate score...")
    # scores, w_array, h_array, thresh_list = run_multi_sample(model, dataset)
    # print("nms...")
    # template_list = list(Path(template_dir).iterdir())

    # for idx in range(len(scores)):
    #     template_path = str(template_list[idx])
    #     score = np.array([scores[idx]])
    #     w = np.array([w_array[idx]])
    #     h = np.array([h_array[idx]])
    #     thresh = np.array([thresh_list[idx]])
    #     boxes, indices = nms_multi(score, w, h, thresh)
    #     compare_show(boxes[0], template_path, image_path, 1/4, save_path)
    for data in tqdm(dataset):
        template_path = data['template_name']
        score = run_one_sample(model, data['template'], data['image'], data['image_name'])
        w = np.array([data['template_w']])
        h = np.array([data['template_h']])
        thresh = np.array([data['thresh']])
        boxes, indices = nms_multi(score, w, h, thresh)

        compare_show(boxes[0], template_path, image_path, 1/4, save_path)

    _ = plot_result_multi(dataset.image_raw, boxes, indices, show=False, save_name='result.png')
    print("result.png was saved")


