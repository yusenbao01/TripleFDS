import os
import numpy as np
import albumentations
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Sampler, DistributedSampler
import pickle
import cv2
import json
import random
import torch
import torch.nn.functional as F
from tqdm import tqdm
import glob
import lightning as L
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.distributed as dist
import string


hw_ratio = 4


def is_main_process():
    """Checks if the current process is the main process (rank 0)."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True  # If not in a distributed environment, treat as the main process


class ProportionalBatchSampler(Sampler):
    """
    A custom sampler that proportionally draws batches from two datasets, with support for distributed training.

    It ensures that in every step of distributed training, all replicas (GPUs) process the same type of batch,
    while also guaranteeing that the data processed by each replica is unique.

    Args:
        dataset (ConcatDataset): A ConcatDataset containing two sub-datasets.
        batch_size (int): The batch size for **each replica (GPU)**.
        ratio (int, optional): The batch ratio of the first dataset to the second. If None, it's calculated automatically.
        shuffle (bool): Whether to shuffle the batch order every epoch.
        seed (int): The random seed for shuffling.
    """

    def __init__(self, dataset, batch_size, ratio=None, shuffle=True, seed=0):
        if not isinstance(dataset, ConcatDataset) or len(dataset.datasets) != 2:
            raise ValueError("This sampler requires a ConcatDataset with two datasets.")

        if dist.is_available() and dist.is_initialized():
            self.num_replicas = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        self.global_batch_size = self.batch_size * self.num_replicas

        # Store original indices, do not shuffle in __init__
        self._indices_ds1 = list(range(len(self.dataset.datasets[0])))
        self.offset_ds2 = len(self.dataset.datasets[0])
        self._indices_ds2_in_concat = [i + self.offset_ds2 for i in range(len(self.dataset.datasets[1]))]

        # Dynamically calculate the ratio
        num_batches_ds1 = len(self._indices_ds1) // self.global_batch_size
        num_batches_ds2 = len(self._indices_ds2_in_concat) // self.global_batch_size

        if ratio is None:
            if num_batches_ds2 > 0:
                calculated_ratio = num_batches_ds1 / num_batches_ds2
                # If DS1 has fewer batches than DS2, ensure at least a 1:1 ratio to avoid 0
                self.ratio = max(1, round(calculated_ratio))
            else:
                self.ratio = num_batches_ds1  # DS2 is empty, only use DS1
        else:
            self.ratio = ratio

        if is_main_process():
            print(f"Data global batch ratio (DS1:DS2) set to: {self.ratio}:1")
            print(
                f"Batch size per replica: {self.batch_size}, Num replicas: {self.num_replicas}, Global batch size: {self.global_batch_size}"
            )

        # Set up the state for the first epoch (0) at the end of __init__
        self.set_epoch(0)

    def _create_schedule(self):
        """Creates a global batch schedule (with fix for infinite loop)."""
        iter_ds1 = iter(self.global_batches_ds1)
        iter_ds2 = iter(self.global_batches_ds2)

        self.schedule = []
        # Continue looping until both iterators are exhausted
        while True:
            batches_from_ds1 = []
            if self.ratio > 0:
                for _ in range(self.ratio):
                    batch = next(iter_ds1, None)
                    if batch:
                        batches_from_ds1.append(batch)
                    else:
                        break  # DS1 is exhausted

            batch_from_ds2 = next(iter_ds2, None)

            # Only add to the schedule if at least one dataset still has batches
            if batches_from_ds1 or batch_from_ds2:
                if batches_from_ds1:
                    self.schedule.extend(batches_from_ds1)
                if batch_from_ds2:
                    self.schedule.append(batch_from_ds2)
            else:
                # If neither iterator returned any batches in this round, both are exhausted
                break

    def __iter__(self):
        # The iterator is only responsible for splitting and yielding batches for the current replica
        for global_batch in self.schedule:
            # Split the local batch for the current rank from the global batch
            start_idx = self.rank * self.batch_size
            end_idx = start_idx + self.batch_size
            local_batch = global_batch[start_idx:end_idx]
            yield local_batch

    def __len__(self):
        return len(self.schedule)

    def set_epoch(self, epoch: int):
        """
        Called by the DataLoader at the beginning of each epoch to ensure different shuffle orders.
        This method is now the sole entry point for preparing the state for each epoch.
        """
        self.epoch = epoch

        if self.shuffle:
            g1 = torch.Generator()
            g1.manual_seed(self.seed + self.epoch)
            shuffled_indices_ds1 = [self._indices_ds1[i] for i in torch.randperm(len(self._indices_ds1), generator=g1)]

            g2 = torch.Generator()
            g2.manual_seed(self.seed + self.epoch + 1)
            shuffled_indices_ds2 = [
                self._indices_ds2_in_concat[i] for i in torch.randperm(len(self._indices_ds2_in_concat), generator=g2)
            ]
        else:
            shuffled_indices_ds1 = self._indices_ds1
            shuffled_indices_ds2 = self._indices_ds2_in_concat

        # Recreate global batches from shuffled indices
        self.global_batches_ds1 = [
            shuffled_indices_ds1[i : i + self.global_batch_size]
            for i in range(0, len(shuffled_indices_ds1), self.global_batch_size)
        ]
        self.global_batches_ds2 = [
            shuffled_indices_ds2[i : i + self.global_batch_size]
            for i in range(0, len(shuffled_indices_ds2), self.global_batch_size)
        ]

        # Filter out incomplete global batches
        self.global_batches_ds1 = [b for b in self.global_batches_ds1 if len(b) == self.global_batch_size]
        self.global_batches_ds2 = [b for b in self.global_batches_ds2 if len(b) == self.global_batch_size]

        # Recreate the schedule
        self._create_schedule()


class SCBReconstructDataset(Dataset):
    def __init__(self, root, transform=None, mask_transform=None, split="train", padding=False):
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split

        # Get all subfolders (each subfolder is a sample)
        self.sample_folders = sorted(glob.glob(os.path.join(root, split, "*")))

        # Preload all sample information
        self.samples = []
        for folder in tqdm(self.sample_folders, desc="Loading dataset samples", disable=not is_main_process()):
            # Read sample information from JSON file
            info_path = os.path.join(folder, "info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        info_data = json.load(f)

                    # Get dimensions, fonts and texts from JSON
                    dimensions = info_data.get("dimensions", [2, 2, 2])
                    fonts = info_data.get("fonts", [])
                    texts = info_data.get("texts", [])
                    standard_font = info_data.get("standard_font", "")
                    cropped_bbox = info_data.get("cropped_bbox", [])

                    # Calculate total combinations
                    font_num, text_num, bg_num = dimensions
                    total_combinations = font_num * text_num * bg_num
                except Exception as e:
                    if is_main_process():
                        print(f"Error reading JSON file {info_path}: {e}")
                    continue

            # Store sample information (images will be loaded on demand in __getitem__)
            self.samples.append(
                {
                    "type": "scb_reconstruct",  # Add type identifier
                    "folder": folder,
                    "dimensions": dimensions,
                    "fonts": fonts,
                    "texts": texts,
                    "standard_font": standard_font,
                    "cropped_bbox": cropped_bbox,
                    # Image paths
                    "text_image_paths": [os.path.join(folder, f"{i}.png") for i in range(total_combinations)],
                    "bg_image_paths": [os.path.join(folder, f"bg{i}.png") for i in range(dimensions[2])],
                    "mask_image_paths": [
                        os.path.join(folder, f"mask{i}.png") for i in range(dimensions[0] * dimensions[1])
                    ],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        folder = sample_info["folder"]
        dimensions = sample_info["dimensions"]

        # Read text images
        text_images = []
        for img_path in sample_info["text_image_paths"]:
            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                text_images.append(img)
            else:
                # Print warning only on the main process
                if is_main_process():
                    print(f"Warning: Image not found {img_path}")

        # Read background images
        ori_bg_images = []
        bg_images = []
        for bg_path in sample_info["bg_image_paths"]:
            if os.path.exists(bg_path):
                bg = Image.open(bg_path).convert("RGB")
                ori_bg_images.append(bg)
            else:
                if is_main_process():
                    print(f"Warning: Background image not found {bg_path}")
        font_num, text_num, bg_num = dimensions
        for i, bbox in enumerate(sample_info["cropped_bbox"]):
            # PIL Image uses crop method for slicing, parameter is (left, top, right, bottom)
            cropped_bg = ori_bg_images[i % bg_num].crop((bbox[2], bbox[0], bbox[3], bbox[1]))
            bg_images.append(self.transform(cropped_bg))

        mask_images = []
        for mask_path in sample_info["mask_image_paths"]:
            mask = Image.open(mask_path).convert("L")
            for i in range(bg_num):
                mask_transformed = self.mask_transform(mask)
                # Dilate the mask by 2 units
                mask_transformed = F.max_pool2d(
                    mask_transformed.unsqueeze(0), kernel_size=5, stride=1, padding=2
                ).squeeze(0)
                mask_images.append(mask_transformed)

        texts = []
        for i in range(font_num):
            for j in range(text_num):
                for k in range(bg_num):
                    texts.append(sample_info["texts"][j])

        text_images = (
            torch.stack(text_images) if text_images and isinstance(text_images[0], torch.Tensor) else text_images
        )
        bg_images = torch.stack(bg_images) if bg_images and isinstance(bg_images[0], torch.Tensor) else bg_images
        mask_images = (
            torch.stack(mask_images) if mask_images and isinstance(mask_images[0], torch.Tensor) else mask_images
        )
        sample = {
            "type": "scb_reconstruct",  # Add type identifier
            "folder": folder,
            "text_images": text_images,
            "bg_images": bg_images,
            "mask_images": mask_images,
            "fonts": sample_info["fonts"],
            "texts": texts,
            "num_scb": dimensions,
            "standard_font": sample_info["standard_font"],
        }

        return sample


class SceneVTGDataset(Dataset):
    """Dataset class specifically for loading the scenevtg_cropped dataset."""

    def __init__(self, data_root, annotation_file, transform=None, repeat_count=8):
        """
        Args:
            data_root (str): The root directory of the dataset (e.g., 'datasets/scenevtg_cropped').
            annotation_file (str): The path to the annotation file (e.g., 'train.json').
            transform (callable, optional): Transformations to apply to the images.
            repeat_count (int): The number of samples to bundle together to match the structure of the SCB dataset.
        """
        self.data_root = data_root
        self.transform = transform
        self.repeat_count = repeat_count

        # Load annotation file
        with open(os.path.join(data_root, annotation_file), "r", encoding="utf-8") as f:
            self.annotations = json.load(f)

        if len(self.annotations) == 0:
            raise ValueError(f"Annotation file '{annotation_file}' is empty or failed to load.")

    def __len__(self):
        # Each sample consists of repeat_count real, distinct image pairs
        return len(self.annotations) // self.repeat_count

    def __getitem__(self, idx):
        # Get a group of self.repeat_count image pairs based on the index
        start_annotation_idx = idx * self.repeat_count

        orig_images = []
        bg_images = []
        texts = []

        for i in range(self.repeat_count):
            current_annotation_idx = start_annotation_idx + i
            sample_info = self.annotations[current_annotation_idx]

            orig_path = os.path.join(self.data_root, sample_info["orig_path"])
            bg_path = os.path.join(self.data_root, sample_info["bg_path"])

            # Strictly require files to exist, if any file in a group is missing, the entire sample loading fails
            # The DataLoader will automatically skip this batch
            orig_img = Image.open(orig_path).convert("RGB")
            bg_img = Image.open(bg_path).convert("RGB")

            if self.transform:
                orig_img = self.transform(orig_img)
                bg_img = self.transform(bg_img)

            orig_images.append(orig_img)
            bg_images.append(bg_img)
            texts.append(sample_info["text"])

        # Stack the list of images into a tensor
        orig_images_stacked = torch.stack(orig_images, dim=0)
        bg_images_stacked = torch.stack(bg_images, dim=0)

        return {
            "type": "scenevtg",
            "text_images": orig_images_stacked,
            "bg_images": bg_images_stacked,
            "texts": texts,
        }


class MostelDataset(Dataset):
    def __init__(self, root, subsets, transform=None, mask_transform=None, split="training", repeat_count=8):
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split
        self.repeat_count = repeat_count
        self.samples = []

        # --- Text validation configuration ---
        ALLOWED_CHARS = set(string.digits + string.ascii_letters + string.punctuation + " ")
        MIN_LEN = 3
        MAX_LEN = 14

        sub_folders_to_process = subsets if subsets else []
        src_image_dir_name = "i_s"
        tgt_image_dir_name = "t_f"
        bg_image_dir_name = "t_b"
        mask_image_dir_name = "mask_t"
        txt_dir_name = "txt"
        image_extensions = [".png", ".jpg", ".jpeg"]

        for folder in sub_folders_to_process:
            sub_folder_path = os.path.join(self.root, self.split, folder)
            txt_dir_path = os.path.join(sub_folder_path, txt_dir_name)

            if not os.path.isdir(txt_dir_path):
                if is_main_process():
                    print(f"Warning: '{txt_dir_name}' folder not found in {sub_folder_path}")
                continue

            for txt_filename in sorted(os.listdir(txt_dir_path)):
                if not txt_filename.endswith(".txt"):
                    continue

                base_name = os.path.splitext(txt_filename)[0]
                txt_file_path = os.path.join(txt_dir_path, txt_filename)

                try:
                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip().split()
                        if len(content) == 2:
                            src_text, tgt_text = content
                        else:
                            if is_main_process():
                                print(f"Warning: Incorrect text file format, should contain two words: {txt_file_path}")
                            continue
                except Exception as e:
                    if is_main_process():
                        print(f"Warning: Error reading text file {txt_file_path}: {e}")
                    continue

                # Validate text
                if not (MIN_LEN <= len(src_text) <= MAX_LEN and MIN_LEN <= len(tgt_text) <= MAX_LEN):
                    continue
                if not (set(src_text).issubset(ALLOWED_CHARS) and set(tgt_text).issubset(ALLOWED_CHARS)):
                    if is_main_process():
                        print(f"Warning: Text contains disallowed characters '{src_text}', '{tgt_text}' in {txt_file_path}")
                    continue

                # Find matching image file
                image_name = None
                for ext in image_extensions:
                    potential_image_name = base_name + ext
                    src_image_path = os.path.join(sub_folder_path, src_image_dir_name, potential_image_name)
                    if os.path.exists(src_image_path):
                        image_name = potential_image_name
                        break

                if image_name is None:
                    if is_main_process():
                        print(f"Warning: No corresponding image file found for {base_name} in {src_image_dir_name}")
                    continue

                # Build full paths and check for existence
                src_image_path = os.path.join(sub_folder_path, src_image_dir_name, image_name)
                tgt_image_path = os.path.join(sub_folder_path, tgt_image_dir_name, image_name)
                bg_image_path = os.path.join(sub_folder_path, bg_image_dir_name, image_name)
                mask_image_path = os.path.join(sub_folder_path, mask_image_dir_name, image_name)
                if os.path.exists(src_image_path) and os.path.exists(tgt_image_path) and os.path.exists(bg_image_path):
                    self.samples.append(
                        {
                            "src_image_path": src_image_path,
                            "tgt_image_path": tgt_image_path,
                            "bg_image_path": bg_image_path,
                            "mask_image_path": mask_image_path,
                            "src_text": src_text,
                            "tgt_text": tgt_text,
                        }
                    )
                else:
                    if is_main_process():
                        print(f"Warning: Image file missing for base name: {image_name}")

    def __len__(self):
        return len(self.samples) // self.repeat_count

    def __getitem__(self, idx):
        start_sample_idx = idx * self.repeat_count

        src_images = []
        tgt_images = []
        bg_images = []
        mask_images = []
        src_texts = []
        tgt_texts = []

        for i in range(self.repeat_count):
            current_sample_idx = start_sample_idx + i
            sample_info = self.samples[current_sample_idx]

            src_image_path = sample_info["src_image_path"]
            tgt_image_path = sample_info["tgt_image_path"]
            bg_image_path = sample_info["bg_image_path"]
            mask_image_path = sample_info["mask_image_path"]
            src_text = sample_info["src_text"]
            tgt_text = sample_info["tgt_text"]

            # Let exceptions be raised to be handled by the DataLoader
            src_image = Image.open(src_image_path).convert("RGB")
            tgt_image = Image.open(tgt_image_path).convert("RGB")
            bg_image = Image.open(bg_image_path).convert("RGB")
            mask_image = Image.open(mask_image_path).convert("L")
            if self.transform:
                src_image = self.transform(src_image)
                tgt_image = self.transform(tgt_image)
                bg_image = self.transform(bg_image)
            if self.mask_transform:
                mask_image = self.mask_transform(mask_image)
                # Dilate the mask by 2 units
                mask_image = F.max_pool2d(mask_image.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

            src_images.append(src_image)
            tgt_images.append(tgt_image)
            bg_images.append(bg_image)
            mask_images.append(mask_image)
            src_texts.append(src_text)
            tgt_texts.append(tgt_text)

        src_images_stacked = torch.stack(src_images, dim=0)
        tgt_images_stacked = torch.stack(tgt_images, dim=0)
        bg_images_stacked = torch.stack(bg_images, dim=0)
        mask_images_stacked = torch.stack(mask_images, dim=0)

        return {
            "type": "mostel",
            "text_images": src_images_stacked,
            "target_images": tgt_images_stacked,
            "bg_images": bg_images_stacked,
            "mask_images": mask_images_stacked,
            "texts": src_texts,
            "tgt_texts": tgt_texts,
        }


class ContrastiveSCBDataset(L.LightningDataModule):
    def __init__(
        self,
        scb_reconstruct_dir=None,
        scenevtg_data_dir=None,
        mostel_data_dir=None,
        mostel_subsets=None,
        batch_size=32,
        num_workers=16,
        image_height_size=64,
        hw_ratio=4,
        sampling_ratio=None,
        scb_repeat_count=8,
    ):

        super().__init__()
        self.scb_reconstruct_dir = scb_reconstruct_dir
        self.scenevtg_data_dir = scenevtg_data_dir
        self.mostel_data_dir = mostel_data_dir
        self.mostel_subsets = mostel_subsets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_height_size = image_height_size
        self.image_width_size = image_height_size * hw_ratio
        self.hw_ratio = hw_ratio
        self.sampling_ratio = sampling_ratio
        self.scb_repeat_count = scb_repeat_count

    def setup(self, stage="fit"):
        """Prepare datasets."""
        # Data preprocessing for scene text tasks
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_height_size, self.image_width_size), interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_height_size, self.image_width_size), interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )
        # Choose different dataset classes based on the stage
        if stage in ["fit", "train"]:
            all_train_datasets = []

            # 1. SCB Dataset (optional)
            if self.scb_reconstruct_dir and os.path.exists(self.scb_reconstruct_dir):
                train_dataset_scb = SCBReconstructDataset(
                    root=self.scb_reconstruct_dir,
                    transform=self.transform,
                    mask_transform=self.mask_transform,
                    split="train",
                )
                if len(train_dataset_scb) > 0:
                    all_train_datasets.append(train_dataset_scb)
                    if is_main_process():
                        print(f"SCB training set size: {len(train_dataset_scb)}")
                elif is_main_process():
                    print(f"Warning: SCB dataset directory exists, but the dataset is empty: {self.scb_reconstruct_dir}")

            # 2. SceneVTG Dataset (optional)
            if self.scenevtg_data_dir and os.path.exists(os.path.join(self.scenevtg_data_dir, "train.json")):
                train_dataset_scenevtg = SceneVTGDataset(
                    data_root=self.scenevtg_data_dir,
                    annotation_file="train.json",
                    transform=self.transform,
                    repeat_count=self.scb_repeat_count,
                )
                if len(train_dataset_scenevtg) > 0:
                    all_train_datasets.append(train_dataset_scenevtg)
                    if is_main_process():
                        print(f"SceneVTG training set size: {len(train_dataset_scenevtg)}")

            # 3. Mostel Dataset (optional)
            if self.mostel_data_dir and self.mostel_subsets and os.path.exists(self.mostel_data_dir):
                train_mostel_dataset = MostelDataset(
                    root=self.mostel_data_dir,
                    subsets=self.mostel_subsets,
                    transform=self.transform,
                    mask_transform=self.mask_transform,
                    split="training",
                    repeat_count=self.scb_repeat_count,
                )
                if len(train_mostel_dataset) > 0:
                    all_train_datasets.append(train_mostel_dataset)
                    if is_main_process():
                        print(f"Mostel training set size: {len(train_mostel_dataset)}")

            # Construct the final self.train_dataset based on the number of loaded datasets
            if not all_train_datasets:
                self.train_dataset = None
                if is_main_process():
                    print("Warning: No training datasets were loaded.")
            elif len(all_train_datasets) == 1:
                self.train_dataset = all_train_datasets[0]
                if is_main_process():
                    print("Using a single dataset for training.")
            elif len(all_train_datasets) == 2:
                self.train_dataset = ConcatDataset(all_train_datasets)
                if is_main_process():
                    print("Merging 2 datasets for training using a proportional sampler.")
            else:  # len > 2
                # To be compatible with ProportionalBatchSampler (which only supports 2 datasets),
                # we will use the first dataset as the primary and merge the rest as the secondary.
                primary_dataset = all_train_datasets[0]
                secondary_datasets = ConcatDataset(all_train_datasets[1:])
                self.train_dataset = ConcatDataset([primary_dataset, secondary_datasets])
                if is_main_process():
                    print(
                        f"Merging {len(all_train_datasets)} datasets for training (1 primary + {len(all_train_datasets)-1} secondary) using a proportional sampler."
                    )

        if stage in ["fit", "validate", "test"]:
            # Set up validation sets accordingly, but do not merge them
            self.val_scb_dataset = None
            self.val_scenevtg_dataset = None

            if self.scb_reconstruct_dir and os.path.exists(os.path.join(self.scb_reconstruct_dir, "val")):
                self.val_scb_dataset = SCBReconstructDataset(
                    root=self.scb_reconstruct_dir,
                    transform=self.transform,
                    mask_transform=self.mask_transform,
                    split="val",
                )
                if is_main_process() and len(self.val_scb_dataset) > 0:
                    print(f"SCB validation set size: {len(self.val_scb_dataset)}")

            if self.scenevtg_data_dir and os.path.exists(os.path.join(self.scenevtg_data_dir, "val.json")):
                self.val_scenevtg_dataset = SceneVTGDataset(
                    data_root=self.scenevtg_data_dir,
                    annotation_file="val.json",
                    transform=self.transform,
                    repeat_count=self.scb_repeat_count,
                )
                if is_main_process() and len(self.val_scenevtg_dataset) > 0:
                    print(f"SceneVTG validation set size: {len(self.val_scenevtg_dataset)}")

    def get_random_samples(self, num_samples=5, dataset_type="train"):
        """
        Get random samples for visualization and debugging.

        Args:
            num_samples: The number of samples to get.
            dataset_type: The dataset to get samples from ("train" or "val").

        Returns:
            A list of random samples.
        """
        if not hasattr(self, "train_dataset"):
            self.setup("fit")

        dataset = self.train_dataset if dataset_type == "train" else self.val_dataset

        if hasattr(dataset, "get_random_sample"):
            return [dataset.get_random_sample() for _ in range(num_samples)]
        else:
            indices = torch.randperm(len(dataset))[:num_samples].tolist()
            return [dataset[idx] for idx in indices]

    def get_random_batch(self, num_samples=5, dataset_type="train", device=None):
        """
        Get a random batch and move it to the specified device.

        Args:
            num_samples: The number of samples.
            dataset_type: The type of dataset ('train' or 'val').
            device: The target device. If None, the data is not moved.

        Returns:
            Batch data on the device.
        """
        samples = self.get_random_samples(num_samples, dataset_type)
        batch = self.collate_fn(samples) if self.stage == "pretrain" else self.collate_fn_train(samples)

        if device is not None:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)
            print(f"Batch data moved to device: {device}")

        return batch

    def get_fixed_samples(self, num_samples=4, dataset_type="train"):
        """
        Get the first few samples of the dataset for visualization and debugging.

        Args:
            num_samples: The number of samples to get.
            dataset_type: The dataset to get samples from ("train", "val", or "test").

        Returns:
            A list of the first few fixed samples.
        """
        if not hasattr(self, "train_dataset"):
            self.setup("fit")

        if dataset_type == "train":
            dataset = self.train_dataset
        elif dataset_type == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        indices = list(range(min(num_samples, len(dataset))))
        return [dataset[idx] for idx in indices]

    def get_fixed_batch(self, num_samples=4, dataset_type="train", device=None):
        """
        Get the first few samples of the dataset as a batch and move it to the specified device.

        Args:
            num_samples: The number of samples.
            dataset_type: The type of dataset ('train', 'val', or 'test').
            device: The target device. If None, the data is not moved.

        Returns:
            Batch data on the device.
        """
        samples = self.get_fixed_samples(num_samples, dataset_type)
        batch = self.collate_fn(samples) if self.stage == "pretrain" else self.collate_fn_train(samples)

        if device is not None:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

        return batch

    def train_dataloader(self):
        """Returns the training dataloader."""
        if not hasattr(self, "train_dataset"):
            self.setup("fit")

        if isinstance(self.train_dataset, ConcatDataset):
            if is_main_process():
                print("Using ProportionalBatchSampler to load mixed data.")
            batch_sampler = ProportionalBatchSampler(
                self.train_dataset, batch_size=self.batch_size, ratio=self.sampling_ratio
            )
            return DataLoader(
                self.train_dataset,
                batch_sampler=batch_sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn,
            )
        else:
            is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            if is_main_process():
                if is_distributed:
                    print("Using DistributedSampler to load a single dataset.")
                else:
                    print("Using a standard sampler to load a single dataset.")
            
            sampler = DistributedSampler(self.train_dataset, shuffle=True) if is_distributed else None
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn,
                drop_last=True,
            )

    def val_dataloader(self):
        """Returns the validation dataloader."""
        dataloaders = []
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if self.val_scenevtg_dataset:
            val_scenevtg_sampler = DistributedSampler(self.val_scenevtg_dataset, shuffle=False) if is_distributed else None
            num_workers_vtg = min(self.num_workers, 4) if len(self.val_scenevtg_dataset) < 1000 else self.num_workers
            vtg_loader = DataLoader(
                self.val_scenevtg_dataset,
                batch_size=self.batch_size,
                sampler=val_scenevtg_sampler,
                shuffle=False,
                num_workers=num_workers_vtg,
                pin_memory=True,
                collate_fn=self.collate_fn,
                drop_last=True,
            )
            dataloaders.append(vtg_loader)

        if self.val_scb_dataset:
            val_scb_sampler = DistributedSampler(self.val_scb_dataset, shuffle=False) if is_distributed else None
            dataloaders.append(
                DataLoader(
                    self.val_scb_dataset,
                    batch_size=self.batch_size,
                    sampler=val_scb_sampler,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    collate_fn=self.collate_fn,
                    drop_last=True,
                )
            )

        if not dataloaders:
            if is_main_process():
                print("Warning: No available validation datasets found.")
        return dataloaders

    def test_dataloader(self):
        """Returns the test dataloader."""
        if not hasattr(self, "test_dataset"):
            if is_main_process():
                print("Warning: No test set found.")
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def collate_fn(self, batch):
        """
        A generic collate_fn that can handle batches from different datasets.
        It adds a 'type' key to identify the data source and does not need to remove it from the samples.
        """
        if not batch:
            return None

        batch_type = batch[0].get("type", "scb_reconstruct")

        if batch_type in ["scb_reconstruct", "scenevtg", "mostel"]:
            # --- Generic processing logic ---
            texts_batch = []
            for item in batch:
                texts_batch.extend(item["texts"])

            text_images_list = [item["text_images"] for item in batch]
            text_images_batch = torch.cat(text_images_list, dim=0)

            bg_images_list = [item["bg_images"] for item in batch]
            bg_images_batch = torch.cat(bg_images_list, dim=0)

            # --- Safe handling of SCB-specific fields ---
            folders = [item.get("folder", "N/A") for item in batch]
            fonts_batch = [item.get("fonts", []) for item in batch]
            standard_fonts = [item.get("standard_font", "") for item in batch]
            num_scb = batch[0].get("num_scb")

            collated_batch = {
                "folders": folders,
                "text_images": text_images_batch,
                "bg_images": bg_images_batch,
                "fonts": fonts_batch,
                "texts": texts_batch,
                "standard_fonts": standard_fonts,
                "num_scb": num_scb,
                "batch_size": len(batch),
            }

            if "mask_images" in batch[0]:
                mask_images_list = [item["mask_images"] for item in batch]
                mask_images_stacked = torch.cat(mask_images_list, dim=0)
                collated_batch["mask_images"] = mask_images_stacked

            if batch_type == "mostel":
                target_texts_batch = []
                for item in batch:
                    target_texts_batch.extend(item["tgt_texts"])
                collated_batch["target_texts"] = target_texts_batch
                target_images_list = [item["target_images"] for item in batch]
                target_images_stacked = torch.cat(target_images_list, dim=0)
                collated_batch["target_images"] = target_images_stacked
            
            collated_batch["type"] = batch_type
        else:
            collated_batch = torch.utils.data.dataloader.default_collate(batch)

        return collated_batch


class InferenceNewDataset(Dataset):
    def __init__(self, size, image_dir, target_txt):
        self.image_paths = []
        self.target_texts = []
        self.max_h = size
        self.max_w = size * hw_ratio

        with open(target_txt, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    img_name, text = parts
                    self.image_paths.append(os.path.join(image_dir, img_name))
                    self.target_texts.append(text)

        self.transforms = transforms.Compose(
            [
                transforms.Resize((self.max_h, self.max_w), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image_path):
        img = Image.open(image_path)
        if not img.mode == "RGB":
            img = img.convert("RGB")

        width, height = img.size
        ori_size = (width, height)
        img = self.transforms(img)
        height, width = img.shape[1], img.shape[2]
        return img, ori_size, (width, height)

    def __getitem__(self, i):
        item = {}
        image1, ori_size, image1_size = self.preprocess_image(self.image_paths[i])
        item["image1"] = image1
        item["ori_size"] = ori_size
        item["image1_size"] = image1_size
        item["rec2"] = self.target_texts[i]
        item["img_name"] = os.path.basename(self.image_paths[i])
        return item


class _MostelEditingDataset(Dataset):
    """
    An internal dataset class for loading samples from the Mostel dataset specifically for text editing tasks.
    It treats source/target pairs as editing operations.
    """

    def __init__(self, root, subsets, transform=None, split="training"):
        self.root = root
        self.transform = transform
        self.split = split
        self.samples = []

        # --- Text validation configuration ---
        ALLOWED_CHARS = set(string.digits + string.ascii_letters + string.punctuation + " ")
        MIN_LEN = 3
        MAX_LEN = 14

        sub_folders_to_process = subsets if subsets else []
        src_image_dir_name = "i_s"
        tgt_image_dir_name = "t_f"
        txt_dir_name = "txt"
        image_extensions = [".png", ".jpg", ".jpeg"]

        for folder in sub_folders_to_process:
            sub_folder_path = os.path.join(self.root, self.split, folder)
            txt_dir_path = os.path.join(sub_folder_path, txt_dir_name)

            if not os.path.isdir(txt_dir_path):
                if is_main_process():
                    print(f"Warning: '{txt_dir_name}' folder not found in {sub_folder_path}")
                continue

            for txt_filename in sorted(os.listdir(txt_dir_path)):
                if not txt_filename.endswith(".txt"):
                    continue

                base_name = os.path.splitext(txt_filename)[0]
                txt_file_path = os.path.join(txt_dir_path, txt_filename)

                try:
                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip().split()
                        if len(content) == 2:
                            src_text, tgt_text = content
                        else:
                            if is_main_process():
                                print(f"Warning: Incorrect text file format, should contain two words: {txt_file_path}")
                            continue
                except Exception as e:
                    if is_main_process():
                        print(f"Warning: Error reading text file {txt_file_path}: {e}")
                    continue

                # Validate text
                if not (MIN_LEN <= len(src_text) <= MAX_LEN and MIN_LEN <= len(tgt_text) <= MAX_LEN):
                    continue
                if not (set(src_text).issubset(ALLOWED_CHARS) and set(tgt_text).issubset(ALLOWED_CHARS)):
                    if is_main_process():
                        print(f"Warning: Text contains disallowed characters '{src_text}', '{tgt_text}' in {txt_file_path}")
                    continue

                # Find matching image file
                image_name = None
                for ext in image_extensions:
                    potential_image_name = base_name + ext
                    src_image_path = os.path.join(sub_folder_path, src_image_dir_name, potential_image_name)
                    if os.path.exists(src_image_path):
                        image_name = potential_image_name
                        break

                if image_name is None:
                    continue

                # Build full paths and check for existence
                src_image_path = os.path.join(sub_folder_path, src_image_dir_name, image_name)
                tgt_image_path = os.path.join(sub_folder_path, tgt_image_dir_name, image_name)

                if os.path.exists(src_image_path) and os.path.exists(tgt_image_path):
                    self.samples.append(
                        {
                            "image1_path": src_image_path,
                            "image2_path": tgt_image_path,
                            "rec1": src_text,
                            "rec2": tgt_text,
                        }
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        try:
            image1 = Image.open(sample_info["image1_path"]).convert("RGB")
            image2 = Image.open(sample_info["image2_path"]).convert("RGB")

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)

            return {
                "type": "mostel_editing",
                "image1": image1,
                "image2": image2,
                "rec1": sample_info["rec1"],
                "rec2": sample_info["rec2"],
            }
        except Exception as e:
            if is_main_process():
                print(f"Error loading sample {idx} ({sample_info['image1_path']}): {e}")
            # Return the next sample to avoid training interruption
            return self.__getitem__((idx + 1) % len(self))


class _SCBEditingDataset(Dataset):

    def __init__(self, root, transform=None, mask_transform=None, split="train", rsste=False):
        """
        Args:
            root (str): The root directory of the dataset (e.g., 'datasets/scb_pretrain').
            transform (callable, optional): Transformations to apply to the images.
            split (str): The dataset split ('train' or 'val').
        """
        self.root = root
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split
        self.samples = []
        self.rsste = rsste
        sample_folders = sorted(glob.glob(os.path.join(self.root, self.split, "*")))

        for folder in tqdm(sample_folders, desc=f"Building editing pairs for '{self.split}' set", disable=not is_main_process()):
            info_path = os.path.join(folder, "info.json")
            if not os.path.exists(info_path):
                continue

            try:
                with open(info_path, "r", encoding="utf-8") as f:
                    info_data = json.load(f)

                dimensions = info_data.get("dimensions", [2, 2, 2])
                texts = info_data.get("texts", [])
                font_num, text_num, bg_num = dimensions

                if text_num < 2:  # Must have at least two different texts to form an editing pair
                    continue

                # Helper function to convert (f, t, b) coordinates to a linear index
                def get_linear_idx(f, t, b):
                    return f * (text_num * bg_num) + t * bg_num + b

                # --- Generate editing pairs ---
                # Keep font and background constant, only change text
                for f_idx in range(font_num):
                    for b_idx in range(bg_num):
                        for t1_idx in range(text_num):
                            for t2_idx in range(text_num):
                                if t1_idx == t2_idx:
                                    continue

                                idx1 = get_linear_idx(f_idx, t1_idx, b_idx)
                                idx2 = get_linear_idx(f_idx, t2_idx, b_idx)

                                img1_path = os.path.join(folder, f"{idx1}.png")
                                img2_path = os.path.join(folder, f"{idx2}.png")
                                bg_path = os.path.join(folder, f"bg{b_idx}.png")
                                mask_path = os.path.join(folder, f"mask{idx2//bg_num}.png")
                                rec1 = texts[t1_idx]
                                rec2 = texts[t2_idx]

                                if os.path.exists(img1_path) and os.path.exists(img2_path):
                                    self.samples.append(
                                        {
                                            "image1_path": img1_path,
                                            "image2_path": img2_path,
                                            "bg_path": bg_path,
                                            "mask_path": mask_path,
                                            "rec1": rec1,
                                            "rec2": rec2,
                                        }
                                    )
            except Exception as e:
                if is_main_process():
                    print(f"Error processing folder {folder}: {e}")
                continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]

        try:
            image1 = Image.open(sample_info["image1_path"]).convert("RGB")
            image2 = Image.open(sample_info["image2_path"]).convert("RGB")
            bg = Image.open(sample_info["bg_path"]).convert("RGB")
            mask = Image.open(sample_info["mask_path"]).convert("L")

            if self.transform:
                image1 = self.transform(image1)
                image2 = self.transform(image2)
                bg = self.transform(bg)

            if self.mask_transform:
                mask = self.mask_transform(mask)
                # Dilate the mask by 2 units
                mask = F.max_pool2d(mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)

            if self.rsste:
                return {
                    "type": "scb_editing",
                    "image1": image1,
                    "image2": image2,
                    "rec1": sample_info["rec1"],
                    "rec2": sample_info["rec2"],
                }
            else:
                return {
                    "type": "scb_editing",
                    "text_images": image1,
                    "target_images": image2,
                    "bg_images": bg,
                    "mask_images": mask,
                    "texts": sample_info["rec1"],
                    "target_texts": sample_info["rec2"],
                }
        except Exception as e:
            if is_main_process():
                print(f"Error loading sample {idx} ({sample_info['image1_path']}): {e}")
            # Return the next sample to avoid training interruption
            return self.__getitem__((idx + 1) % len(self))


class SCBEditingDataset(L.LightningDataModule):
    def __init__(
        self,
        scb_edit_dir=None,
        mostel_data_dir=None,
        mostel_subsets=None,
        batch_size=32,
        num_workers=16,
        image_height_size=64,
        hw_ratio=4,
        rsste=False,
    ):
        super().__init__()
        self.scb_edit_dir = scb_edit_dir
        self.mostel_data_dir = mostel_data_dir
        self.mostel_subsets = mostel_subsets
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_height_size = image_height_size
        self.image_width_size = image_height_size * hw_ratio
        self.transform = None
        self.train_dataset = None
        self.val_dataset = None
        self.rsste = rsste

    def setup(self, stage=None):
        # Create common image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_height_size, self.image_width_size), interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.image_height_size, self.image_width_size), interpolation=InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )

        if stage == "fit" or stage is None:
            train_datasets = []
            val_datasets = []

            # 1. Load SCB Editing dataset
            if self.scb_edit_dir:
                scb_train = _SCBEditingDataset(
                    root=self.scb_edit_dir,
                    transform=self.transform,
                    mask_transform=self.mask_transform,
                    split="train",
                    rsste=self.rsste,
                )
                if len(scb_train) > 0:
                    train_datasets.append(scb_train)
                    if is_main_process():
                        print(f"SCB Editing training set size: {len(scb_train)}")

                val_path = os.path.join(self.scb_edit_dir, "val")
                if os.path.exists(val_path):
                    scb_val = _SCBEditingDataset(
                        root=self.scb_edit_dir,
                        transform=self.transform,
                        mask_transform=self.mask_transform,
                        split="val",
                        rsste=self.rsste,
                    )
                    if len(scb_val) > 0:
                        val_datasets.append(scb_val)
                        if is_main_process():
                            print(f"SCB Editing validation set size: {len(scb_val)}")

            # 2. Load Mostel Editing dataset
            if self.mostel_data_dir and self.mostel_subsets:
                mostel_train = _MostelEditingDataset(
                    root=self.mostel_data_dir,
                    subsets=self.mostel_subsets,
                    transform=self.transform,
                    split="training",
                )
                if len(mostel_train) > 0:
                    train_datasets.append(mostel_train)
                    if is_main_process():
                        print(f"Mostel Editing training set size: {len(mostel_train)}")

            # Merge datasets
            if train_datasets:
                self.train_dataset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
                if is_main_process():
                    print(f"Total training samples: {len(self.train_dataset)}")
            else:
                self.train_dataset = None
                if is_main_process():
                    print("Warning: No training datasets were loaded.")

            if val_datasets:
                self.val_dataset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
                if is_main_process():
                    print(f"Total validation samples: {len(self.val_dataset)}")
            else:
                self.val_dataset = None
                if is_main_process():
                    print("Warning: No validation datasets were loaded.")

    def train_dataloader(self):
        if not self.train_dataset:
            if is_main_process():
                print("Warning: Training dataset is empty, cannot create DataLoader.")
            return None
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if not self.val_dataset:
            if is_main_process():
                print("Warning: Validation dataset is empty, cannot create DataLoader.")
            return []
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
