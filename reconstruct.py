import os
import time
import gc
from contextlib import nullcontext
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from Utils import load_checkpoint
from Dataset2 import ABDataset
from Model import Generator

gc.collect()
torch.cuda.empty_cache()

input_dir = "3T_testing_dataset"
output_dir = "fake_dataset_7T_780"
checkpoint = "Results/gen_B_epoch_780.pth"
image_width = 256
image_height = 256
batch_size = 1


def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir_exists(output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_ctx = torch.amp.autocast("cuda") if device.type == "cuda" else nullcontext()

def masking(a, b):
    l_top = l_bottom = 0
    a = a[0]
    b = b[0]

    for i in range(a.shape[1]):
        if torch.sum(a[:, i, :]) != 0:
            break
        l_top += 1

    for i in range(a.shape[1]):
        if torch.sum(a[:, a.shape[1] - i - 1, :]) != 0:
            break
        l_bottom += 1

    b[:, :l_top, :] = 0
    b[:, b.shape[1] - l_bottom:, :] = 0

    return a, b

NUM_HEADS = [8, 8, 8, 8, 16, 16, 16, 16, 32, 32, 32, 32] 
current_incremental_layer_index = 12
patch_size = 8

gen = Generator(
    img_channels=3,
    width=image_width,
    height=image_height,
    patch_sizes=[4,8,16,32],
    dim=1024,
    mlp_ratio=4,
    drop_rate=0.,
    att_heads=NUM_HEADS,
    use_cross_domain=True
).to(device)

gen.current_incremental_layer_index = current_incremental_layer_index
try:
    load_checkpoint(checkpoint, gen, None, 1e-5)
    print("Model loaded successfully!")
    
    print("\nAnalyzing checkpoint contents...")
    checkpoint_data = torch.load(checkpoint, map_location=device)
    state_dict = checkpoint_data["state_dict"]
    print(f"  Checkpoint keys: {len(state_dict.keys())} parameters")
    print(f"  Sample keys: {list(state_dict.keys())[:5]}...")
    
    cross_domain_keys = [k for k in state_dict.keys() if 'cross_domain' in k]
    print(f"  Cross-domain parameters: {len(cross_domain_keys)}")
    if cross_domain_keys:
        print(f"  Sample cross-domain keys: {cross_domain_keys[:3]}...")
    else:
        print("  WARNING: No cross-domain parameters found in checkpoint!")
        
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

gen.eval()

print("Note: Using input image as cross-domain features for reconstruction")

print("Verifying model architecture matches training:")
print(f"  img_channels: 3")
print(f"  width: {image_width}")
print(f"  height: {image_height}")
print(f"  patch_sizes: [4,8,16,32]")
print(f"  dim: 1024")
print(f"  mlp_ratio: 4")
print(f"  drop_rate: 0.0")
print(f"  att_heads: {NUM_HEADS}")
print(f"  use_cross_domain: True")
print(f"  current_incremental_layer_index: {current_incremental_layer_index}")
print(f"  patch_size (for inference): {patch_size}")

print(f"\nParameter mismatch analysis:")
model_keys = set(gen.state_dict().keys())
checkpoint_keys = set(state_dict.keys())

missing_in_model = checkpoint_keys - model_keys
missing_in_checkpoint = model_keys - checkpoint_keys

if missing_in_model:
    print(f"  Parameters in checkpoint but not in model: {len(missing_in_model)}")
    print(f"  Sample: {list(missing_in_model)[:3]}...")
if missing_in_checkpoint:
    print(f"  Parameters in model but not in checkpoint: {len(missing_in_checkpoint)}")
    print(f"  Sample: {list(missing_in_checkpoint)[:3]}...")
if not missing_in_model and not missing_in_checkpoint:
    print("  All parameter names match perfectly!")
print("="*50)

print("\nTesting model with random input...")
test_input = torch.randn(1, 3, image_height, image_width).to(device)
with torch.no_grad():
    test_cross_features = gen.patches[str(patch_size)](test_input)
    test_output = gen(test_input, patch_size, cross_domain_features=test_cross_features)
    print(f"Test input shape: {test_input.shape}")
    print(f"Test output shape: {test_output.shape}")
    print(f"Test output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    print(f"Test output mean: {test_output.mean():.3f}")
    print(f"Test output std: {test_output.std():.3f}")
    
    if test_output.abs().max() < 1e-6:
        print("Model is generating all zeros!")
    elif test_output.std() < 1e-6:
        print("Model has very low variance!")
    else:
        print("Model appears to be working normally")
print("="*50)

transforms = A.Compose(
    [
        A.Resize(width=image_width, height=image_height),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ]
)

val_dataset = ABDataset(root_a=input_dir, transform=transforms)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

loop = tqdm(val_loader, leave=True)

start = time.time()

for idx, (image, filename) in enumerate(loop):
    image = image.to(device)

    with amp_ctx:
        with torch.no_grad():
            cross_domain_features = gen.patches[str(patch_size)](image)
        
        gen_image = gen(image, patch_size, cross_domain_features=cross_domain_features)
        
        if idx < 3:
            print(f"\nImage {idx}:")
            print(f"  Input image shape: {image.shape}")
            print(f"  Input image range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"  Generated image shape: {gen_image.shape}")
            print(f"  Generated image range: [{gen_image.min():.3f}, {gen_image.max():.3f}]")
            print(f"  Generated image mean: {gen_image.mean():.3f}")
            print(f"  Generated image std: {gen_image.std():.3f}")
            
            if gen_image.abs().max() < 1e-6:
                print("  WARNING: Generated image appears to be all zeros!")
            elif gen_image.std() < 1e-6:
                print("  WARNING: Generated image has very low variance!")
            else:
                print("  Generated image looks normal")
        
        image, gen_image = masking(image*0.5+0.5, gen_image*0.5+0.5)

        for img, fname in zip(gen_image, filename):
            save_path = os.path.join(output_dir, fname)
            save_image(img, save_path)

end = time.time()

print(f"Processing completed in {round(end - start, 3)} seconds.")
