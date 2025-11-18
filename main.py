import os, sys, argparse, datetime, time
import importlib
import numpy as np
import torch, torchvision

from torch.utils.data import DataLoader

# Updated imports for Lightning 2.0+ using official convention
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.utilities import rank_zero_info

from omegaconf import OmegaConf, ListConfig
from PIL import Image, ImageDraw, ImageFont
from model.utils import load_checkpoint_adaptive
from collections import OrderedDict
import shutil


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def generate_experiment_name(transformer_config, lightning_config, vae_ckpt_path, learning_rate=None):
    """Generates an experiment name automatically from the configuration."""
    model_params = transformer_config.model.params
    transformer_params = model_params.transformer_config.params
    data_params = transformer_config.data.params

    max_text_len = model_params.get("max_text_len", "_")
    style_len = model_params.get("style_len", "_")
    background_len = model_params.get("background_len", "_")
    n_head = transformer_params.get("n_head", "_")
    n_embd = model_params.get("n_embd", "_")
    accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", "_")

    scb_data_dir = data_params.get("scb_edit_dir", "")
    if scb_data_dir == "":
        scb_data_dir = data_params.get("scb_reconstruct_dir", "")
    if data_params.get("mostel_data_dir", None):
        other_dataset = "with_mostel"
    else:
        other_dataset = "only_scb"
    data_dir_parts = scb_data_dir.split("-")
    data_size = "unk"
    data_structure = "unk"
    difficulty = "unk"
    if len(data_dir_parts) > 2:
        data_size = data_dir_parts[2]
        data_structure = data_dir_parts[3]
        difficulty = data_dir_parts[4]

    if "kl-f4" in vae_ckpt_path:
        kl_factor = "klf4"
    elif "kl-f8" in vae_ckpt_path:
        kl_factor = "klf8"
    else:
        kl_factor = "klfx"

    name_parts = [
        f"{lightning_config.name}-{data_size}-{data_structure}-{difficulty}-{other_dataset}",
        f"c{max_text_len}s{style_len}b{background_len}",
        f"d{n_embd}h{n_head}",
        f"lr{learning_rate}",
        f"{kl_factor}",
        f"agb{accumulate_grad_batches}",
    ]

    return "-".join(name_parts)


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        num_workers=None,
        shuffle_test_loader=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.shuffle_test_loader = shuffle_test_loader
        self.shuffle_val_dataloader = shuffle_val_dataloader

        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)

    def train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=None,
            shuffle=self.shuffle_val_dataloader,
        )

    def test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=None,
            shuffle=self.shuffle_test_loader,
        )


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.configs_saved = False

    def _save_configs(self):
        """Private method to save configuration files."""
        if self.configs_saved:
            return  # Avoid saving multiple times

        os.makedirs(self.logdir, exist_ok=True)
        os.makedirs(self.ckptdir, exist_ok=True)
        os.makedirs(self.cfgdir, exist_ok=True)

        shutil.copyfile("./model/model.py", os.path.join(self.cfgdir, "model.py"))
        shutil.copyfile("./model/minGPT.py", os.path.join(self.cfgdir, "minGPT.py"))
        shutil.copyfile("./model/module.py", os.path.join(self.cfgdir, "module.py"))

        try:
            OmegaConf.save(self.config, os.path.join(self.cfgdir, "project.yaml"))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "lightning.yaml"),
            )
            self.configs_saved = True
        except Exception as e:
            print(f"Warning: Failed to save configs: {e}")

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self._save_configs()

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            self._save_configs()
            if (
                "callbacks" in self.lightning_config
                and "metrics_over_trainsteps_checkpoint" in self.lightning_config["callbacks"]
            ):
                os.makedirs(os.path.join(self.ckptdir, "trainstep_checkpoints"), exist_ok=True)
        else:
            if not self.resume and os.path.exists(self.logdir):
                dst = os.path.join(os.path.dirname(self.logdir), "child_runs", os.path.basename(self.logdir))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        max_images,
        clamp=True,
        rescale=True,
        disabled=False,
        log_images_kwargs=None,
        num_train_logs_per_epoch=1,
        num_val_logs_per_epoch=1,
    ):
        super().__init__()
        self.rescale = rescale
        self.max_images = max_images
        self.logger_log_images = {L.pytorch.loggers.TensorBoardLogger: self._tensorboard}
        self.clamp = clamp
        self.disabled = disabled
        self.log_images_kwargs = log_images_kwargs or {}
        self.num_train_logs_per_epoch = num_train_logs_per_epoch
        self.num_val_logs_per_epoch = num_val_logs_per_epoch
        self.train_log_indices = None
        self.val_log_indices = None

    def on_train_epoch_start(self, trainer, pl_module):
        """At the beginning of each training epoch, randomly select batch indices to log."""
        if self.disabled:
            return

        num_batches = trainer.num_training_batches
        if self.num_train_logs_per_epoch > num_batches:
            indices = list(range(num_batches))
        else:
            indices = np.random.choice(num_batches, self.num_train_logs_per_epoch, replace=False)
        self.train_log_indices = sorted(list(indices))

    def on_validation_epoch_start(self, trainer, pl_module):
        """At the beginning of each validation epoch, randomly select batch indices to log."""
        if self.disabled:
            return

        self.val_log_indices = []
        for num_batches in trainer.num_val_batches:
            if self.num_val_logs_per_epoch > num_batches:
                indices = list(range(num_batches))
            else:
                indices = np.random.choice(num_batches, self.num_val_logs_per_epoch, replace=False)
            self.val_log_indices.append(sorted(list(indices)))

    @staticmethod
    def create_text_image(text, w, h, nc):
        """Creates a black and white image tensor for the given text."""
        img = Image.new("L", (w, h), "black")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("./SCB-Datagen/arial.ttf", size=w // 15)
        except IOError:
            font = ImageFont.load_default()

        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            textwidth, textheight = right - left, bottom - top
        except AttributeError:
            textwidth, textheight = draw.textsize(text, font=font)

        x, y = (w - textwidth) / 2, (h - textheight) / 2
        draw.text((x, y), text, fill="white", font=font)

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
        if nc > 1:
            img_tensor = img_tensor.expand(-1, nc, -1, -1)
        return img_tensor

    @rank_zero_only
    def _tensorboard(self, pl_module, grid, split):
        tag = f"{split}/combined"
        pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, logger, split, grid, global_step, current_epoch, batch_idx):
        root_dir = os.getcwd()
        if logger and hasattr(logger, "save_dir") and logger.save_dir:
            root_dir = logger.save_dir
        elif logger and hasattr(logger, "log_dir") and logger.log_dir:
            root_dir = logger.log_dir

        root = os.path.join(root_dir, "images", split)
        grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
        grid = (grid * 255).astype(np.uint8)

        filename = f"gs-{global_step:07}_e-{current_epoch:04}_b-{batch_idx:07}_combined.png"
        path = os.path.join(root, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if not (hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0):
            return

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

        for k in images:
            N = min(images[k].shape[0], self.max_images)
            images[k] = images[k][:N].detach().cpu()
            if self.clamp:
                images[k] = torch.clamp(images[k], -1.0, 1.0)
                images[k] = (images[k] + 1.0) / 2.0

        desired_order = ["img", "ori_img", "bg", "ori_bg"]
        image_keys = [k for k in desired_order if k in images]
        image_keys.extend(sorted([k for k in images if k not in desired_order]))

        if not image_keys:
            if is_train:
                pl_module.train()
            return

        all_images_orig = [images[k] for k in image_keys]
        if all_images_orig[0].shape[0] == 0:
            if is_train:
                pl_module.train()
            return

        nrow = 8
        padded_image_rows = []
        for img_tensor in all_images_orig:
            padding_size = nrow - img_tensor.shape[0]
            if padding_size > 0:
                padding = torch.zeros(
                    padding_size, *img_tensor.shape[1:], device=img_tensor.device, dtype=img_tensor.dtype
                )
                img_tensor = torch.cat([img_tensor, padding], dim=0)
            padded_image_rows.append(img_tensor)

        _, num_channels, height, width = all_images_orig[0].shape
        device = all_images_orig[0].device

        if batch["type"] == "scb_reconstruct":
            text_labels = [batch["texts"][0], batch["texts"][2], batch["fonts"][0][0], batch["fonts"][0][1]]
        else:
            text_labels = [batch["texts"][0], batch["texts"][1], "null", "null"]
        text_row_list = []
        text_image_width = width * 2
        for label in text_labels:
            text_img_double_width = self.create_text_image(label, text_image_width, height, num_channels).to(device)
            text_row_list.append(text_img_double_width[..., :width])
            text_row_list.append(text_img_double_width[..., width:])

        padded_image_rows.append(torch.cat(text_row_list, dim=0))

        combined_tensor = torch.cat(padded_image_rows, dim=0)
        grid = torchvision.utils.make_grid(combined_tensor, nrow=nrow)

        self.log_local(pl_module.logger, split, grid, pl_module.global_step, pl_module.current_epoch, batch_idx)

        logger_log_images = self.logger_log_images.get(type(pl_module.logger))
        if logger_log_images:
            logger_log_images(pl_module, grid, split)

        if is_train:
            pl_module.train()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not self.disabled and self.train_log_indices is not None:
            if batch_idx in self.train_log_indices:
                if trainer.global_rank == 0:
                    print(
                        f"ImageLogger: Logging training images at global_step={pl_module.global_step}, batch_idx={batch_idx}"
                    )
                self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not self.disabled and self.val_log_indices is not None:
            if batch_idx in self.val_log_indices[dataloader_idx]:
                if trainer.global_rank == 0:
                    print(
                        f"ImageLogger: Logging validation images at global_step={pl_module.global_step}, batch_idx={batch_idx} (dataloader {dataloader_idx})"
                    )
                self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled:
            self.log_img(pl_module, batch, batch_idx, split="test")


class CUDACallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            device = trainer.strategy.root_device
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            device = trainer.strategy.root_device
            torch.cuda.synchronize(device)
            max_memory = torch.cuda.max_memory_allocated(device) / 2**20
            epoch_time = time.time() - self.start_time

            if hasattr(trainer.strategy, "reduce"):
                max_memory = trainer.strategy.reduce(max_memory)
                epoch_time = trainer.strategy.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f} MiB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with custom parameters.")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated list of GPU indices to use.")
    parser.add_argument(
        "--disentanglement_weight",
        type=float,
        default=1.0,
        help="Global weight for both inter and intra contrastive losses.",
    )
    parser.add_argument(
        "--use_adaptive_weight",
        action="store_true",
        help="Enable adaptive weighting between inter and intra loss.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/lightning.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    transformer_config = OmegaConf.load(config.model_configs.transformer_config)
    vae_ckpt_path = config.model_configs.vae_ckpt_path
    trainer_config = config.get("trainer", OmegaConf.create())

    if args.gpus:
        try:
            gpu_indices = [int(g.strip()) for g in args.gpus.split(",")]
            trainer_config.devices = gpu_indices
        except ValueError:
            raise ValueError(f"Invalid format for --gpus. Expected comma-separated integers, but got '{args.gpus}'.")

    transformer_config.model.params.disentanglement_weight = args.disentanglement_weight
    transformer_config.model.params.use_adaptive_weight = args.use_adaptive_weight

    if "interW" in config.name or "intraW" in config.name or "disW" in config.name:
        import re
        config.name = re.sub(r"(interW|intraW|disW)[\d.-]+", "disW" + str(args.disentanglement_weight), config.name)
        if args.use_adaptive_weight:
            if "-adaptive" not in config.name:
                config.name += "-adaptive"

    bs = transformer_config.data.params.batch_size
    base_lr = transformer_config.model.base_learning_rate
    lr_config = config.get("learning_rate", OmegaConf.create())
    scale_lr = lr_config.get("scale_lr", True)
    devices_config = trainer_config.get("devices", 1)
    accumulate_grad_batches = trainer_config.get("accumulate_grad_batches", 1)
    if isinstance(devices_config, (list, ListConfig)):
        ngpu = len(devices_config)
    elif isinstance(devices_config, int):
        ngpu = devices_config
    else:
        ngpu = 1
    scb_factor = 1
    learning_rate = (scb_factor * accumulate_grad_batches * ngpu * bs * base_lr) if scale_lr else base_lr

    new_name = generate_experiment_name(transformer_config, config, vae_ckpt_path, learning_rate=learning_rate)
    config.name = new_name
    print(f"Generated experiment name: {new_name}")

    ckpt_path = config.get("resume_from_checkpoint")
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise ValueError(f"Checkpoint path {ckpt_path} is not a valid file.")
        logdir = os.path.dirname(os.path.dirname(os.path.dirname(ckpt_path)))
    else:
        nowname = now + ("_" + config.name if config.name else "")
        logdir = os.path.join(config.logdir, nowname)
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(config.seed)

    vae_config = OmegaConf.load(config.model_configs.vae_config)
    transformer_config.model.params.ckpt_path = ckpt_path
    transformer_config.model.params.vae_config = vae_config.model
    transformer_config.model.params.vae_ckpt_path = config.model_configs.vae_ckpt_path

    model = instantiate_from_config(transformer_config.model)

    trainer_kwargs = {}
    logger_config = config.logger
    logger_config.params.save_dir = logdir
    trainer_kwargs["logger"] = instantiate_from_config(logger_config)

    callbacks_cfg = config.get("callbacks", OmegaConf.create())
    if "setup_callback" in callbacks_cfg:
        callbacks_cfg.setup_callback.params.update(
            {
                "resume": ckpt_path,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": transformer_config,
                "lightning_config": config,
            }
        )
    if "checkpoint_callback" in callbacks_cfg:
        if hasattr(model, "monitor"):
            callbacks_cfg.checkpoint_callback.params.monitor = model.monitor
        callbacks_cfg.checkpoint_callback.params.dirpath = os.path.join(ckptdir, "trainstep_checkpoints")

    trainer_kwargs["callbacks"] = (
        [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg] if callbacks_cfg else []
    )

    trainer_config_dict = OmegaConf.to_container(trainer_config, resolve=True)
    trainer_config_dict.update(trainer_kwargs)
    trainer_config_dict["use_distributed_sampler"] = False
    trainer = Trainer(gradient_clip_val=1.0, **trainer_config_dict)

    data = instantiate_from_config(transformer_config.data)
    if not isinstance(data, L.LightningDataModule):
        data = DataLoader(
            data,
            batch_size=transformer_config.data.batch_size,
            num_workers=transformer_config.data.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    model.learning_rate = learning_rate
    if trainer.global_rank == 0:
        print("--- Learning Rate Configuration ---")
        print(f"  Batch Size: {bs}")
        print(f"  Base LR: {base_lr}")
        print(f"  Devices: {devices_config} (ngpu: {ngpu})")
        print(f"  Accumulate Grad Batches: {accumulate_grad_batches}")
        print(f"  SCB Factor: {scb_factor}")
        print(f"  Scale LR: {scale_lr}")
        print(f"  Final Learning Rate: {model.learning_rate:.2e}")

    if config.get("load_checkpoint_adaptive"):
        model = load_checkpoint_adaptive(model, config.load_checkpoint_adaptive, trainer=trainer)
        trainer.fit(model, data)
    else:
        trainer.fit(model, data, ckpt_path=ckpt_path)
