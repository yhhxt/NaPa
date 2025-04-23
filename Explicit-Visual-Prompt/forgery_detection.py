import argparse
import os
from PIL import Image

import torch
from torchvision import transforms
import models
import yaml
from mmcv.runner import get_dist_info, init_dist, load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model', default='save/mit_b4.pth')
    parser.add_argument('--prompt', default='save/_train_segformer_evp_caisa/prompt_epoch_last.pth')
    parser.add_argument('--resolution', default='512,512')
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--config', default='configs/demo.yaml')
    parser.add_argument('--input_path', default='.', help='Path to process images from')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = models.make(config['model']).cuda()
    if 'segformer' in config['model']['name']:
        checkpoint = load_checkpoint(model.encoder, args.model)
        model.encoder.PALETTE = checkpoint
        if args.prompt != 'none':
            print('loading prompt...')
            checkpoint = torch.load(args.prompt)
            model.encoder.backbone.prompt_generator.load_state_dict(checkpoint['prompt'])
            model.encoder.decode_head.load_state_dict(checkpoint['decode_head'])
    else:
        model.encoder.load_state_dict(torch.load(args.model), strict=False)

    h, w = list(map(int, args.resolution.split(',')))
    img_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor(),
    ])

    inverse_transform = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                             std=[1, 1, 1])
    ])

    path = args.input_path
    file_list = os.listdir(path)
    
    for i in file_list:
        file_path = os.path.join(path, i)
        output_path = os.path.join(path, i+'_evp')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        if os.path.isdir(file_path):
            png_list = os.listdir(file_path)
            for png in png_list:
                if png.lower().endswith(('.jpg', '.jpeg', '.png')):
                    input = os.path.join(file_path, png)
                    output = os.path.join(output_path, png)

                    img = Image.open(input).convert('RGB')
                    img = img_transform(img)
                    img = img.cuda()

                    pred = model.encoder.forward_dummy(img.unsqueeze(0))
                    pred = torch.sigmoid(pred).view(1, h, w).cpu()

                    transforms.ToPILImage()(pred).save(output)