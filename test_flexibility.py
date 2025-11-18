import os
import torch
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from main import instantiate_from_config
from pytorch_lightning import seed_everything
from model.utils import load_checkpoint_adaptive
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def get_image_transform(image_h, image_w):
    """Returns a composed transform for image preprocessing."""
    return transforms.Compose(
        [
            transforms.Resize((image_h, image_w), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def create_concatenated_image(image_pred_batch, background_pred_batch):
    """Creates a concatenated image from 8 generated images and 2 backgrounds."""
    generated_images_np = []

    # Process 8 generated images
    for j in range(8):
        img_tensor = image_pred_batch[j].clamp(-1, 1)
        img_np = ((img_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
        generated_images_np.append(img_np)

    # Process 2 backgrounds
    bg1_tensor = background_pred_batch[0].clamp(-1, 1)
    bg1_np = ((bg1_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)

    bg2_tensor = background_pred_batch[1].clamp(-1, 1)
    bg2_np = ((bg2_tensor.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)

    # Create concatenated image: 2 rows, 5 columns (8 images + 2 backgrounds)
    row1 = np.concatenate(generated_images_np[:5], axis=1)
    row2 = np.concatenate(generated_images_np[5:] + [bg1_np, bg2_np], axis=1)
    concatenated_img = np.concatenate((row1, row2), axis=0)

    return concatenated_img


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--name", type=str, default="test", help="Experiment name.")
    parser.add_argument("--transformer_config", type=str, required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image_dir", type=str, required=True, help="Path to the directory containing images (i_s folder)."
    )
    parser.add_argument("--image_h", type=int, default=64)
    parser.add_argument("--image_w", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--task", type=str, default="permutation", help="Task name for output directory.")
    return parser


def main(args):
    seed_everything(args.seed)

    # Load model
    config = OmegaConf.load(args.transformer_config)
    model = instantiate_from_config(config.model).to(args.device)
    model = load_checkpoint_adaptive(model, args.resume)
    model.eval()

    transform = get_image_transform(args.image_h, args.image_w)

    # Read i_s.txt (source text labels)
    is_txt_path = os.path.join(os.path.dirname(args.image_dir), "i_s.txt")
    if not os.path.exists(is_txt_path):
        print(f"Error: i_s.txt not found at {is_txt_path}")
        return

    # Parse i_s.txt
    is_data = []
    with open(is_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                img_name, text = parts
                img_path = os.path.join(args.image_dir, img_name)
                if os.path.exists(img_path):
                    is_data.append((img_path, text))

    num_images = len(is_data)
    if num_images < 2:
        print("Error: Need at least 2 images for pairing.")
        return

    # Create pairs: (0,1), (2,3), (4,5), ...
    # If odd number, last image pairs with first
    pairs = []
    for i in range(0, num_images - 1, 2):
        pairs.append((i, i + 1))

    # If odd number of images, pair last with first
    if num_images % 2 == 1:
        pairs.append((num_images - 1, 0))

    print(f"Total images: {num_images}")
    print(f"Total pairs: {len(pairs)}")

    # Create output directory
    args.target_path = f"output/{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(args.target_path, exist_ok=True)
    print(f"Saving results to {args.target_path}")

    # Process each pair
    for pair_idx, (idx1, idx2) in enumerate(tqdm(pairs, desc="Processing image pairs")):
        img_path1, text1 = is_data[idx1]
        img_path2, text2 = is_data[idx2]

        # Load images
        image1 = Image.open(img_path1).convert("RGB")
        image2 = Image.open(img_path2).convert("RGB")

        # Transform to tensors
        image1_tensor = transform(image1).unsqueeze(0).to(args.device)
        image2_tensor = transform(image2).unsqueeze(0).to(args.device)

        # Generate synthesis
        with torch.no_grad():
            image_pred_batch, background_pred_batch = model.task_edit_style(
                image1_tensor, [text1], image2_tensor, [text2]
            )

        # Create and save concatenated image
        concatenated_img = create_concatenated_image(image_pred_batch, background_pred_batch)
        output_filename = f"pair_{pair_idx:03d}_{os.path.splitext(os.path.basename(img_path1))[0]}_{os.path.splitext(os.path.basename(img_path2))[0]}.png"
        Image.fromarray(concatenated_img).save(os.path.join(args.target_path, output_filename))

    print(f"Finished processing {len(pairs)} pairs.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
