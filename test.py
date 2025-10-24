import torch
from torch.autograd import Variable
import utils
import numpy as np
import time
from fusenet import Fusenet
import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='SemFuse: Test semantic-guided infrared and visible image fusion on a directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM for semantic guidance.')
    parser.add_argument('--dataset_path', type=str, default='/content/kaist-dataset/', help='Path to the root DIRECTORY of the KAIST dataset')
    parser.add_argument('--output_path', type=str, default="/content/results/", help='Output directory for fused images')
    # ✅ MODIFIED: Added a new argument to limit the number of test images
    parser.add_argument('--test_num', type=int, default=None, help='Number of images to test. If not set, all images will be used.')
    return parser.parse_args()

def load_model(path, sam_checkpoint, sam_model_type, use_sam):
    if not os.path.exists(path):
        raise ValueError('Invalid model path: {}'.format(path))

    fuse_net = Fusenet(
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        use_sam=use_sam
    )
    fuse_net.load_state_dict(torch.load(path))
    para = sum([np.prod(list(p.size())) for p in fuse_net.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(fuse_net._get_name(), para * type_size / 1000 / 1000))
    print(f"Using SAM guidance: {'Yes' if use_sam else 'No'}")
    fuse_net.eval()
    fuse_net.cuda()
    return fuse_net

def generate_fuse_image(model, vi, ir):
    out = model(vi, ir)
    return out

def fuse_test(model, vi_path, ir_path, output_path_root, use_sam):
    if not os.path.exists(vi_path) or not os.path.exists(ir_path):
        print(f"Error: One or both image paths do not exist for {os.path.basename(vi_path)}. Skipping.")
        return

    vi_img = utils.get_test_images(vi_path, height=None, width=None)
    ir_img = utils.get_test_images(ir_path, height=None, width=None)
    out = utils.get_image(vi_path, height=None, width=None)

    vi_img = Variable(vi_img.cuda(), requires_grad=False)
    ir_img = Variable(ir_img.cuda(), requires_grad=False)

    with torch.no_grad():
        img_fusion = generate_fuse_image(model, vi_img, ir_img)

    original_filename = os.path.basename(vi_path).split('.')[0]
    file_name = f'fused_{original_filename}_{"withSAM" if use_sam else "noSAM"}.png'
    output_path = os.path.join(output_path_root, file_name)

    if torch.cuda.is_available():
        img = img_fusion.cpu().clamp(0, 255).numpy()
    else:
        img = img_fusion.clamp(0, 255).numpy()
    img = img.astype('uint8')
    utils.save_images(output_path, img, out)

def main():
    opt = parse_args()
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    with torch.no_grad():
        model = load_model(opt.model_path, opt.sam_checkpoint, opt.sam_model_type, opt.use_sam)

        print("Searching for all images in the dataset...")
        visible_image_paths = utils.list_images(opt.dataset_path)
        print(f"\nFound {len(visible_image_paths)} total images.")

        # ✅ MODIFIED: Add logic to limit the number of test images if the argument is provided
        if opt.test_num is not None:
            visible_image_paths = visible_image_paths[:opt.test_num]
            print(f"Limiting test to the first {len(visible_image_paths)} images as specified by --test_num.")
        
        total_images = len(visible_image_paths)
        print(f"Processing {total_images} images.")

        for i, visible_path in enumerate(visible_image_paths):
            infrared_path = visible_path.replace('/visible/', '/lwir/')
            filename = os.path.basename(visible_path)

            if not os.path.exists(infrared_path):
                print(f"Warning: Corresponding infrared image not found for {filename}. Skipping.")
                continue

            start = time.time()
            fuse_test(model, visible_path, infrared_path, opt.output_path, opt.use_sam)
            end = time.time()
            print(f"Processed ({i+1}/{total_images}): {filename} | Time: {end - start:.4f} seconds")

    print('\nSemFuse testing completed successfully.')

if __name__ == "__main__":
    main()
