import os

import torch
import traceback
import csv
from contextlib import nullcontext
from torch import nn
from torch import optim
from torchvision.models import vgg19
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from Utils import save_checkpoint, load_checkpoint, wait_and_retry
from Dataset import ABDataset
from Model import Generator, MultiScaleDiscriminator
from torchmetrics.functional import structural_similarity_index_measure as ssim_loss

import gc
gc.collect()
torch.cuda.empty_cache()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def structural_loss(fake, real, alpha=0.84):
    l1_loss = nn.L1Loss()(fake, real)

    ssim_value = 1 - ssim_loss(fake, real)
    return alpha * l1_loss + (1 - alpha) * ssim_value

class TextureLoss(nn.Module):
    def __init__(self, layer_idx=5):
        super(TextureLoss, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(*list(vgg[:layer_idx])).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, fake, real):
        fake_features = self.feature_extractor(fake)
        real_features = self.feature_extractor(real)
        return nn.MSELoss()(fake_features, real_features)

def adjust_learning_rate(optimizer, epoch, initial_lr=1e-5, decay_start=300, decay_every=20, decay_factor=0.5):
    if epoch >= decay_start:
        num_decays = (epoch - decay_start) // decay_every
        new_lr = initial_lr * (decay_factor ** num_decays)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


def train_fn(disc_A, disc_B, gen_A, gen_B, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler, epoch, 
             texture_loss_fn, alpha=0.84, csv_file="training_losses.csv", patch_size=32, use_cross_domain=True):
    global count
    avg_dloss = 0
    avg_gloss = 0
    losses = []
    loop = tqdm(loader, leave=True)
    amp_ctx = torch.amp.autocast("cuda") if DEVICE.type == "cuda" else nullcontext()

    adjust_learning_rate(opt_gen, epoch, initial_lr=LEARNING_RATE)
    adjust_learning_rate(opt_disc, epoch, initial_lr=LEARNING_RATE)


    for idx, (a, b) in enumerate(loop):
        a = a.to(DEVICE)
        b = b.to(DEVICE)

        with amp_ctx:
            if use_cross_domain:
                with torch.no_grad():
                    cross_features_a = gen_B.patches[str(patch_size)](b)
                    cross_features_b = gen_A.patches[str(patch_size)](a)
                
                fake_a = gen_A(b, patch_size, cross_features_a)
                fake_b = gen_B(a, patch_size, cross_features_b)
            else:
                fake_a = gen_A(b, patch_size)
                fake_b = gen_B(a, patch_size)

            D_A_real_outputs = disc_A(a)  
            D_A_fake_outputs = disc_A(fake_a.detach())  
            D_B_real_outputs = disc_B(b)
            D_B_fake_outputs = disc_B(fake_b.detach())

            D_A_real_loss = sum([mse(pred, torch.ones_like(pred)) for pred in D_A_real_outputs])
            D_A_fake_loss = sum([mse(pred, torch.zeros_like(pred)) for pred in D_A_fake_outputs])
            D_B_real_loss = sum([mse(pred, torch.ones_like(pred)) for pred in D_B_real_outputs])
            D_B_fake_loss = sum([mse(pred, torch.zeros_like(pred)) for pred in D_B_fake_outputs])

            D_A_loss = D_A_real_loss + D_A_fake_loss
            D_B_loss = D_B_real_loss + D_B_fake_loss
            D_loss = ((D_A_loss + D_B_loss) / 2) * LAMBDA_ADV

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        with amp_ctx:
            if use_cross_domain:
                cross_features_a = gen_B.patches[str(patch_size)](b)  
                cross_features_b = gen_A.patches[str(patch_size)](a)  
                
                fake_a = gen_A(b, patch_size, cross_features_a)
                fake_b = gen_B(a, patch_size, cross_features_b)
            else:
                fake_a = gen_A(b, patch_size)
                fake_b = gen_B(a, patch_size)
            
            D_A_fake_outputs = disc_A(fake_a)  
            D_B_fake_outputs = disc_B(fake_b)
            loss_G_A = sum([mse(pred, torch.ones_like(pred)) for pred in D_A_fake_outputs])
            loss_G_B = sum([mse(pred, torch.ones_like(pred)) for pred in D_B_fake_outputs])

            if use_cross_domain:
                cycle_b = gen_B(fake_a, patch_size, cross_features_b)
                cycle_a = gen_A(fake_b, patch_size, cross_features_a)
            else:
                cycle_b = gen_B(fake_a, patch_size)
                cycle_a = gen_A(fake_b, patch_size)
            cycle_b_loss = l1(b, cycle_b)
            cycle_a_loss = l1(a, cycle_a)

            if use_cross_domain:
                identity_b = gen_B(b, patch_size, cross_features_b)
                identity_a = gen_A(a, patch_size, cross_features_a)
            else:
                identity_b = gen_B(b, patch_size)
                identity_a = gen_A(a, patch_size)
            identity_b_loss = l1(b, identity_b)
            identity_a_loss = l1(a, identity_a)

            texture_loss = texture_loss_fn(fake_a, a) + texture_loss_fn(fake_b, b)

            struct_loss_a = structural_loss(fake_a, a, alpha=alpha)
            struct_loss_b = structural_loss(fake_b, b, alpha=alpha)
            struct_loss = struct_loss_a + struct_loss_b

            G_loss = (
                loss_G_B * LAMBDA_ADV
                + loss_G_A * LAMBDA_ADV
                + cycle_b_loss * LAMBDA_CYCLE
                + cycle_a_loss * LAMBDA_CYCLE
                + identity_a_loss * LAMBDA_IDENTITY
                + identity_b_loss * LAMBDA_IDENTITY
                + texture_loss * LAMBDA_TEXTURE
                + struct_loss * LAMBDA_STRUCT
            )

            avg_dloss += D_loss.item()
            avg_gloss += G_loss.item()

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        losses.append({
            "G_loss": G_loss.item(),
            "D_H_loss": D_A_loss.item(),
            "D_Z_loss": D_B_loss.item(),
            "cycle_loss_3T": cycle_a_loss.item(),
            "cycle_loss_7T": cycle_b_loss.item(),
            "Identity_loss_3T": identity_a_loss.item(),
            "Identity_loss_7T": identity_b_loss.item(),
            "texture_loss": texture_loss.item(),
            "struct_loss": struct_loss.item(),
        })

        if idx % 3000 == 0:
            save_image(fake_a * 0.5 + 0.5, f"{path}/Generated from 7T/{count}_fake.png")
            save_image(fake_b * 0.5 + 0.5, f"{path}/Generated from 3T/{count}_fake.png")
            save_image(b * 0.5 + 0.5, f"{path}/Generated from 7T/{count}_real.png")
            save_image(a * 0.5 + 0.5, f"{path}/Generated from 3T/{count}_real.png")
            count += 1
        loop.set_postfix(epoch=epoch + 1, loss_g=avg_gloss / (idx + 1), loss_d=avg_dloss / (idx + 1))

    save_losses_to_csv(csv_file, epoch, losses)

    return losses



def save_losses_to_csv(csv_file, epoch, losses):
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "Epoch", "Batch", "G_loss", "D_H_loss", "D_Z_loss", 
                "Cycle_loss_3T", "Cycle_loss_7T", "Identity_loss_3T", 
                "Identity_loss_7T", "Texture_loss", "Struct_loss"
            ])

        for idx, loss in enumerate(losses):
            writer.writerow([
                epoch, idx, loss["G_loss"], loss["D_H_loss"], loss["D_Z_loss"], 
                loss["cycle_loss_3T"], loss["cycle_loss_7T"], 
                loss["Identity_loss_3T"], loss["Identity_loss_7T"], 
                loss["texture_loss"], loss["struct_loss"]
            ])


    

def main():
    disc_A = MultiScaleDiscriminator(in_channels=3).to(DEVICE)
    disc_B = MultiScaleDiscriminator(in_channels=3).to(DEVICE)
    gen_A = Generator(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, att_heads=NUM_HEADS, use_cross_domain=True).to(DEVICE)
    gen_B = Generator(width=IMAGE_WIDTH, height=IMAGE_HEIGHT, att_heads=NUM_HEADS, use_cross_domain=True).to(DEVICE)

    opt_disc = optim.Adam(
        list(disc_A.parameters()) + list(disc_B.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )
    opt_gen = optim.Adam(
        list(gen_A.parameters()) + list(gen_B.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    texture_loss_fn = TextureLoss(layer_idx=5).to(DEVICE)

    dataset = ABDataset(
        root_a=TRAIN_DIR + "/3T", root_b=TRAIN_DIR + "/7T", transform=transforms
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    patch_size = 8
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch -> {epoch}\nPatch Size -> {patch_size}\nNo. of Encoder Layers -> {gen_A.num_layers}")
        adjust_learning_rate(opt_gen, epoch, initial_lr=LEARNING_RATE)
        adjust_learning_rate(opt_disc, epoch, initial_lr=LEARNING_RATE)

        while True:
            try:
                train_fn(
                    disc_A,
                    disc_B,
                    gen_A,
                    gen_B,
                    loader,
                    opt_disc,
                    opt_gen,
                    L1,
                    mse,
                    d_scaler,
                    g_scaler,
                    epoch,
                    texture_loss_fn=texture_loss_fn,
                    alpha=0.84,
                    csv_file="training_losses_cross_domain.csv",
                    patch_size=patch_size,
                    use_cross_domain=True
                )
                break
            except torch.cuda.OutOfMemoryError:
                print(f"[Epoch {epoch}] CUDA OOM error. Retrying after delay...")
                wait_and_retry(delay_minutes=5)
            except RuntimeError as e:
                if "CUBLAS_STATUS_ALLOC_FAILED" in str(e):
                    print(f"[Epoch {epoch}] cuBLAS allocation failed. Retrying after delay...")
                    traceback.print_exc()
                    wait_and_retry(delay_minutes=5)
                else:
                    raise e

        if SAVE_MODEL and (epoch + 1) % 20 == 0:
            save_checkpoint(gen_A, opt_gen, filename=f"{path}/gen_A_epoch_{epoch+1}.pth")
            save_checkpoint(gen_B, opt_gen, filename=f"{path}/gen_B_epoch_{epoch+1}.pth")
            save_checkpoint(disc_A, opt_disc, filename=f"{path}/disc_A_epoch_{epoch+1}.pth")
            save_checkpoint(disc_B, opt_disc, filename=f"{path}/disc_B_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TRAIN_DIR = "training_dataset"
    path = "Results"
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-5
    LAMBDA_ADV = 1
    LAMBDA_IDENTITY = 5
    LAMBDA_TEXTURE = 5
    LAMBDA_STRUCT = 10
    LAMBDA_CYCLE = 15
    NUM_WORKERS = 4
    NUM_EPOCHS = 1000
    LOAD_MODEL = False
    SAVE_MODEL = True
    IMAGE_HEIGHT = 256
    IMAGE_WIDTH = 256
    count = 0
    NUM_HEADS = [8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32] 
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(f"{path}/Generated from 7T")
        os.mkdir(f"{path}/Generated from 3T")
    transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
            ToTensorV2(),
        ], is_check_shapes=False,
        additional_targets={"image0": "image"},
    )
    main()
