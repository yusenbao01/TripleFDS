import os
import torch
import importlib
import argparse
import albumentations
import math
import numpy as np
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from datetime import datetime
from data.dataset import InferenceNewDataset
from main import instantiate_from_config, get_obj_from_str
from pytorch_lightning import seed_everything
from model.utils import load_checkpoint_adaptive


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("--transformer_config", type=str, default="configs/decouple_synth.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument(
        "--target_txt", type=str, required=True, help="Path to the text file with image names and target texts."
    )
    parser.add_argument("--image_h", type=int, default=64)
    return parser


def crop_and_resize(x, size, ori_size, k):
    img = x.detach().cpu()
    img = torch.clamp(img, -1.0, 1.0)
    img = (img + 1.0) / 2.0
    img = img.permute(0, 2, 3, 1)[k]
    img = (
        torch.nn.functional.interpolate(
            img.permute(2, 0, 1).unsqueeze(0), size=ori_size, mode="bilinear", align_corners=False  # [1, C, H, W]
        )
        .squeeze(0)
        .permute(1, 2, 0)
    )
    img = (img * 255).clamp(0, 255).to(torch.uint8).numpy()
    return img


def main(args):
    seed_everything(args.seed)

    config = OmegaConf.load(args.transformer_config)
    model = instantiate_from_config(config.model).to("cuda")
    model = load_checkpoint_adaptive(model, args.resume)
    model.eval()

    dataset = InferenceNewDataset(args.image_h, args.image_dir, args.target_txt)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Create target directories
    os.makedirs(os.path.join(args.target_path, "i_t"), exist_ok=True)
    os.makedirs(os.path.join(args.target_path, "concatenated"), exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            images = batch["image1"].to("cuda")
            texts = batch["rec2"]
            image_pred, background_pred = model.task_edit_content(images, texts)

            for k in range(images.shape[0]):
                edited_img = crop_and_resize(
                    image_pred, batch["image1_size"], (batch["ori_size"][1][k], batch["ori_size"][0][k]), k
                )
                background_edited_img = crop_and_resize(
                    background_pred, batch["image1_size"], (batch["ori_size"][1][k], batch["ori_size"][0][k]), k
                )
                # save edited image to i_t
                edited_img_path = os.path.join(args.target_path, "i_t", batch["img_name"][k])
                Image.fromarray(edited_img).save(edited_img_path)
                # process original image
                original_img = crop_and_resize(
                    images, batch["image1_size"], (batch["ori_size"][1][k], batch["ori_size"][0][k]), k
                )
                # save concatenated image to concatenated
                concatenated_img = np.concatenate((original_img, background_edited_img, edited_img), axis=1)
                target_file = os.path.join(args.target_path, "concatenated", batch["img_name"][k])
                Image.fromarray(concatenated_img).save(target_file)


if __name__ == "__main__":
    parser = get_parser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--task",
        type=str,
        default="edit_content",
    )
    args = parser.parse_args()
    args.target_path = f"output/{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(os.path.join(args.target_path, "i_t"), exist_ok=True)
    print(f"Saving results to {args.target_path}")
    main(args)
