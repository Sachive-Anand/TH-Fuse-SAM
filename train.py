import args
import time
import random
import torch
import torch.nn as nn
import utils
# import dataset  <- This is intentionally removed to fix the error
from fusenet import Fusenet
from tqdm import tqdm, trange
from torch.optim import Adam
from os.path import join
from loss import final_ssim, TV_Loss
from loss_p import VggDeep,VggShallow
import os
import argparse
from sam_guidance import SAMGuidance
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='SemFuse: Semantic-Aware Infrared and Visible Image Fusion')
    parser.add_argument('--sam_checkpoint', type=str, default='sam_vit_h_4b8939.pth', help='Path to SAM checkpoint')
    parser.add_argument('--sam_model_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type')
    parser.add_argument('--use_sam', action='store_true', help='Whether to use SAM for semantic guidance.')
    parser.add_argument('--semantic_loss_weight', type=float, default=0.3, help='Weight for semantic loss')
    parser.add_argument('--epochs', type=int, default=args.epochs, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=args.batch_size, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=args.learning_rate, help='Learning rate')
    parser.add_argument('--learning_rate_d', type=float, default=args.learning_rate_d, help='Learning rate for discriminator')
    parser.add_argument('--dataset_path', type=str, default=args.dataset_path, help='Dataset path')
    parser.add_argument('--save_model_path', type=str, default=args.save_model_path, help='Path to save model')
    parser.add_argument('--log_interval', type=int, default=args.log_interval, help='Log interval')
    parser.add_argument('--train_num', type=int, default=args.train_num, help='Number of training samples')
    parser.add_argument('--image_height', type=int, default=args.image_height, help='Image height')
    parser.add_argument('--image_width', type=int, default=args.image_width, help='Image width')
    return parser.parse_args()

class SemanticConsistencyLoss(nn.Module):
    def __init__(self):
        super(SemanticConsistencyLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, fused_semantic, ir_semantic, vi_semantic):
        combined_semantic = torch.max(ir_semantic, vi_semantic)
        return self.l1_loss(fused_semantic, combined_semantic)

def train(image_lists, opt):
    image_mode = 'L'
    fusemodel = Fusenet(sam_checkpoint=opt.sam_checkpoint, sam_model_type=opt.sam_model_type, use_sam=opt.use_sam)
    vgg_ir_model = VggDeep()
    vgg_vi_model = VggShallow()

    if opt.use_sam:
        print("Training with SAM semantic guidance")
        sam_guidance = SAMGuidance(sam_checkpoint=opt.sam_checkpoint, model_type=opt.sam_model_type)
        sam_guidance.cuda()
    else:
        print("Training WITHOUT SAM semantic guidance")
        sam_guidance = None

    L1_loss = nn.L1Loss()
    fusemodel.cuda()
    vgg_ir_model.cuda()
    vgg_vi_model.cuda()

    tbar = trange(opt.epochs, ncols=150)
    print('Start training for SemFuse...')

    for e in tbar:
        print('Epoch %d.....' % (e + 1))
        
        # This logic replaces the need for dataset.py, fixing the error
        random.shuffle(image_lists)
        image_set = image_lists
        batches = len(image_set) // opt.batch_size
        print(f"Number of batches per epoch: {batches}")
        
        fusemodel.train()
        count = 0
        all_model_loss = 0.
        all_semantic_loss = 0.

        if batches == 0:
            print("ERROR: 0 batches created. Your train_num might be smaller than the batch_size.")
            break

        for batch in range(batches):
            image_paths = image_set[batch * opt.batch_size:(batch * opt.batch_size + opt.batch_size)]
            
            path1 = [p for p in image_paths]
            path2 = [p.replace('/visible/', '/lwir/') for p in path1]

            img_vi = utils.get_train_images_auto(path1, height=opt.image_height, width=opt.image_width, mode=image_mode)
            img_ir = utils.get_train_images_auto(path2, height=opt.image_height, width=opt.image_width, mode=image_mode)
            count += 1

            optimizer_model = Adam(fusemodel.parameters(), opt.learning_rate)
            optimizer_model.zero_grad()
            optimizer_vgg_ir = Adam(vgg_ir_model.parameters(), opt.learning_rate_d)
            optimizer_vgg_ir.zero_grad()
            optimizer_vgg_vi = Adam(vgg_vi_model.parameters(), opt.learning_rate_d)
            optimizer_vgg_vi.zero_grad()

            img_vi = img_vi.cuda()
            img_ir = img_ir.cuda()
            
            outputs = fusemodel(img_vi, img_ir)
            
            if opt.use_sam and sam_guidance is not None:
                semantic_attention = sam_guidance(img_ir, img_vi)
                resized_outputs = F.interpolate(outputs, size=semantic_attention.shape[2:], mode='bilinear', align_corners=True)
                fused_semantic = sam_guidance(resized_outputs, resized_outputs)
                semantic_loss_value = SemanticConsistencyLoss()(fused_semantic, semantic_attention, semantic_attention)
            else:
                semantic_loss_value = torch.tensor(0.0, device=outputs.device)
            
            ssim_loss_value = 1 - final_ssim(img_ir, img_vi, outputs)
            
            if opt.use_sam:
                model_loss = ssim_loss_value + opt.semantic_loss_weight * semantic_loss_value
            else:
                model_loss = ssim_loss_value

            model_loss.backward(retain_graph=True)
            optimizer_model.step()
            
            vgg_ir_fuse_out = vgg_ir_model(outputs.detach())[2]
            vgg_ir_out = vgg_ir_model(img_ir)[2]
            per_loss_ir = L1_loss(vgg_ir_fuse_out, vgg_ir_out)
            per_loss_ir.backward(retain_graph=True)
            optimizer_vgg_ir.step()
            
            vgg_vi_fuse_out = vgg_vi_model(outputs.detach())[0]
            vgg_vi_out = vgg_vi_model(img_vi)[0]
            per_loss_vi = L1_loss(vgg_vi_fuse_out, vgg_vi_out)
            per_loss_vi.backward()
            optimizer_vgg_vi.step()

            all_model_loss += ssim_loss_value.item()
            all_semantic_loss += semantic_loss_value.item()

            if (batch + 1) % opt.log_interval == 0:
                mesg = f"{time.ctime()}\tEpoch {e+1}:[{count}/{batches}] fusemodel loss: {all_model_loss/((batch % opt.log_interval)+1):.5f}"
                if opt.use_sam:
                    mesg += f" semantic loss: {all_semantic_loss/((batch % opt.log_interval)+1):.5f}"
                tbar.set_description(mesg)
        
        print(f"\n--- Epoch {e + 1} complete. Saving model... ---")
        fusemodel.eval()
        fusemodel.cpu()
        save_model_filename = f"SemFuse_epoch_{e+1}{'_withSAM' if opt.use_sam else '_noSAM'}.model"
        save_model_path = os.path.join(opt.save_model_path, save_model_filename)
        torch.save(fusemodel.state_dict(), save_model_path)
        print(f"Model saved to: {save_model_path}\n")
        fusemodel.train()
        fusemodel.cuda()

    print("--- Training finished ---")

def main():
    opt = parse_args()
    print("Searching for images...")
    images_path = utils.list_images(opt.dataset_path)
    print(f"Found {len(images_path)} total images in the dataset.")
    
    train_num = opt.train_num
    images_path = images_path[:train_num]
    print(f"Using {len(images_path)} images for training as specified by --train_num.")
    
    train(images_path, opt)

if __name__ == "__main__":
    main()
