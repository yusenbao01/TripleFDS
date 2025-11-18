import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchvision import models, transforms
from main import instantiate_from_config
import math
from model.utils import load_checkpoint_adaptive
from model.module import (
    SimpleEncoder,
    DecoderHead,
    PerceptualLoss,
    StrLabelConverter,
    SimpleProjection,
    StandardProjectionHead,
)


class TripleFDS(L.LightningModule):
    def __init__(
        self,
        transformer_config,
        decoder_config=None,
        alphabet="data/alphabet/en.txt",
        special_tokens="data/alphabet/special_tokens.txt",
        ckpt_path=None,
        max_text_len=32,
        style_len=64,
        background_len=256,
        n_embd=384,
        feature_dim=1024,
        tau=0.2,
        vae_config=None,
        vae_ckpt_path=None,
        disentanglement_weight=1.0,
        use_adaptive_weight=False,
    ):
        super().__init__()
        self.content_len = max_text_len
        self.style_len = style_len
        self.background_len = background_len
        self.conv_h = int(math.sqrt(background_len / 4))
        self.conv_w = self.conv_h * 4
        self.n_embd = n_embd
        self.feature_dim = feature_dim
        self.logit_scale_inter = torch.tensor(1 / tau)
        self.disentanglement_weight = disentanglement_weight
        self.use_adaptive_weight = use_adaptive_weight
        # self.conv = nn.Conv2d(in_channels=3, out_channels=n_embd, kernel_size=8, stride=8, bias=False)
        # self.conv_o = nn.Conv2d(in_channels=n_embd, out_channels=4, kernel_size=1)
        # self.conv = SimpleEncoder(n_embd=n_embd)

        self.query_i = nn.Embedding(background_len, n_embd)
        self.query_s = nn.Embedding(style_len, n_embd)
        self.query_c = nn.Embedding(max_text_len, n_embd)
        self.query_b = nn.Embedding(background_len, n_embd)
        self.str_converter = StrLabelConverter(alphabet, max_text_len, 0, special_tokens)
        self.str_embd = nn.Embedding(len(self.str_converter.idx2name), n_embd)
        self.perceptual_loss_fn = PerceptualLoss()

        # Initialize VAE, used for both encoder and decoder
        if vae_config:
            vae = instantiate_from_config(vae_config)
            vae = load_checkpoint_adaptive(vae, vae_ckpt_path)
            del vae.loss
            self.vae = vae
            self.vae_post_projection = nn.Conv2d(self.vae.embed_dim, self.n_embd, 1)
            self.conv = None
        else:
            # Fallback for when no VAE is provided
            self.conv = SimpleEncoder(n_embd=n_embd)
            self.init_decoder_from_config(decoder_config)
        self.init_transformer_from_config(transformer_config, ckpt_path)
        self.conv_o = DecoderHead(in_channels=n_embd, out_channels=self.vae.embed_dim)

        self.contrastive_head_c = StandardProjectionHead(
            input_dim=self.n_embd,
            output_dim=self.feature_dim,
        )
        self.contrastive_head_s = StandardProjectionHead(
            input_dim=self.n_embd,
            output_dim=self.feature_dim,
        )
        self.contrastive_head_b = StandardProjectionHead(
            input_dim=self.n_embd,
            output_dim=self.feature_dim,
        )
        self.init_indices()

    def init_indices(self):
        self.indices1, self.content_indices, self.style_indices, self.background_indices, self.tokens_len = (
            self.get_indices_with_special_tokens()
        )
        self.indices2, self.image_indices = self.get_special_tokens()
        self.cat_len1 = [self.tokens_len[3] + 2, self.tokens_len[1] + 2, self.tokens_len[2] + 2, self.tokens_len[3] + 2]
        self.cat_len2 = [self.tokens_len[3] + 2, self.tokens_len[2] + 2, self.tokens_len[1] + 2, self.tokens_len[3] + 2]

        # Use the dedicated flag to enable adaptive weighting
        if self.use_adaptive_weight:
            print("Enabling adaptive weight adjustment...")
            self._init_reference_layers()
            # Initialize EMA for gradient norms
            self.grad_norm_ema = {}
            # Register a buffer to store the latest weight ratio to ensure it can be used during validation
            self.register_buffer("lambda_intra_ratio", torch.tensor(1.0))
        else:
            print("Using fixed loss weights.")

    def _init_reference_layers(self):
        """
        Initialize reference layers for each loss for gradient consistency calculation.
        """
        # Directly and explicitly specify the reference layer as the last linear layer of each projection head
        self.last_layer_inter_intra = []
        for head in [self.contrastive_head_b, self.contrastive_head_s, self.contrastive_head_c]:
            # The structure of StandardProjectionHead is fixed: nn.Sequential(nn.Linear, nn.SiLU, nn.Linear)
            # So the last linear layer is self.projection[2]
            last_linear_layer = head.projection[2]
            self.last_layer_inter_intra.append(last_linear_layer.weight)

    def calculate_adaptive_weight(self, main_grad_norm, secondary_grad_norm, secondary_loss_key, decay=0.95):
        """
        A simplified adaptive weight calculation method.
        Directly performs EMA smoothing on the gradient norms and then calculates their ratio.
        """
        # Update EMA gradient norm for the main loss
        if "main" not in self.grad_norm_ema:
            self.grad_norm_ema["main"] = main_grad_norm
        else:
            self.grad_norm_ema["main"] = decay * self.grad_norm_ema["main"] + (1 - decay) * main_grad_norm

        # Update EMA gradient norm for the secondary loss
        if secondary_loss_key not in self.grad_norm_ema:
            self.grad_norm_ema[secondary_loss_key] = secondary_grad_norm
        else:
            self.grad_norm_ema[secondary_loss_key] = (
                decay * self.grad_norm_ema[secondary_loss_key] + (1 - decay) * secondary_grad_norm
            )

        # Calculate the weight so that the weighted gradient norms are roughly equal
        # weight * ema_secondary_grad = ema_main_grad -> weight = ema_main_grad / ema_secondary_grad
        eps = 1e-8
        weight = self.grad_norm_ema["main"] / (self.grad_norm_ema[secondary_loss_key] + eps)

        # Clamp the weight range to prevent extreme values
        clamped_weight = torch.clamp(weight, 0.1, 10.0)

        return clamped_weight, weight

    def init_decoder_from_config(self, config):
        model = instantiate_from_config(config)
        model.eval()
        self.vae = model

    def init_transformer_from_config(self, config, ckpt_path):
        self.transformer = instantiate_from_config(config)
        if ckpt_path is not None:
            print(f"Transformer restored from {ckpt_path}")
            from collections import OrderedDict

            sd = torch.load(ckpt_path)["state_dict"]
            keys = list(sd.keys())
            new_keys = OrderedDict()
            model_state_dict = self.state_dict()
            for k in keys:
                new_k = k
                if k.startswith("vqgan"):
                    new_k = new_k.replace("vqgan", "decoder")
                if new_k in model_state_dict:
                    new_keys[new_k] = sd[k]
            self.load_state_dict(new_keys, strict=True)

    def get_input(self, images):
        x = images
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            x = x
        if x.dtype == torch.double:
            x = x.float()

        if self.conv is None:  # VAE encoder path
            posterior = self.vae.encode(x)
            z = posterior.mode()
            x = self.vae_post_projection(z)
        else:  # Original SimpleEncoder path
            x = self.conv(x)
        return x  # [bs, n_embd, 8, 32]

    @torch.no_grad()
    def log_images(self, batch, **kwargs):
        log = dict()
        # Get input data
        text_images = batch["text_images"]
        texts = batch["texts"]
        # decouple task
        background_tokens, content_tokens, style_tokens, rec_embd, target_ids1, logits1 = self.decouple(
            text_images, texts, target_texts=batch.get("target_texts", None)
        )
        # synth task
        image_tokens, target_ids2, logits2 = self.synth(
            background_tokens, rec_embd, style_tokens, stage="train", dataset_type=batch["type"]
        )
        # decode background and image
        background_latent = self.conv_o(
            background_tokens.permute(0, 2, 1)
            .contiguous()
            .view(background_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        bg_pred = self.vae.decode(background_latent)
        background = batch["bg_images"]
        image_latent = self.conv_o(
            image_tokens.permute(0, 2, 1).contiguous().view(image_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        image_pred = self.vae.decode(image_latent)
        image = batch.get("target_images", batch["text_images"])
        # Decode text from the content part of mode 1
        content_indices_pred = torch.topk(logits1, k=1, dim=-1)[1].view(text_images.shape[0], -1)
        pred_text = self.str_converter.decode(content_indices_pred)
        # Log results
        log["bg"] = bg_pred  # Background decoupled in mode 1
        log["img"] = image_pred  # Reconstructed image synthesized in mode 2
        log["ori_bg"] = background  # Original background
        log["ori_img"] = image  # Original image
        # Log predicted text
        self.logger.experiment.add_text("train/pred_text", str(pred_text), global_step=self.global_step)
        return log

    def forward(self, batch, stage="train"):
        output = dict()
        text_images = batch["text_images"]
        texts = batch["texts"]
        # decouple task
        background_tokens, content_tokens, style_tokens, rec_embd, target_ids1, logits1 = self.decouple(
            text_images, texts, target_texts=batch.get("target_texts", None)
        )
        # synth task(replace content tokens with rec_embd)
        attn_mask = None
        image_tokens, target_ids2, logits2 = self.synth(
            background_tokens,
            rec_embd,
            style_tokens,
            stage=stage,
            dataset_type=batch["type"],
            attn_mask=attn_mask,
        )
        # decode background and image
        background_latent = self.conv_o(
            background_tokens.permute(0, 2, 1)
            .contiguous()
            .view(background_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        bg_pred = self.vae.decode(background_latent)
        background = batch["bg_images"]

        image_latent = self.conv_o(
            image_tokens.permute(0, 2, 1).contiguous().view(image_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        image_pred = self.vae.decode(image_latent)
        image = batch.get("target_images", batch["text_images"])
        # output
        output["bg_pred"] = bg_pred
        output["bg_images"] = background
        output["image_pred"] = image_pred
        output["image"] = image
        output["content_tokens"] = content_tokens
        output["style_tokens"] = style_tokens
        output["background_tokens"] = background_tokens
        output["logits1"] = logits1
        output["logits2"] = logits2
        output["rec_embd"] = rec_embd
        output["target_ids1"] = target_ids1
        output["target_ids2"] = target_ids2
        return output

    # decouple task
    def decouple(self, text_images, texts, target_texts=None):
        img_latent = self.get_input(text_images).flatten(2).permute(0, 2, 1)
        rec_ids, _ = self.str_converter.encode(texts)
        rec_ids = rec_ids.to(self.device)
        if target_texts is None:
            rec_embd = self.str_embd(rec_ids)
        else:
            rec_ids_tgt, _ = self.str_converter.encode(target_texts)
            rec_ids_tgt = rec_ids_tgt.to(self.device)
            rec_embd = self.str_embd(rec_ids_tgt)
        query_c = self.query_c.weight[None, :, :].expand(img_latent.shape[0], -1, -1)
        query_s = self.query_s.weight[None, :, :].expand(img_latent.shape[0], -1, -1)
        query_b = self.query_b.weight[None, :, :].expand(img_latent.shape[0], -1, -1)
        inputs1, target_ids1 = self.wrap_decouple_tokens(img_latent, query_c, query_s, query_b, rec_ids)
        embeddings1, logits1 = self.transformer(inputs1, mode="decouple", cat_len=self.cat_len1)
        content_tokens = embeddings1[:, self.content_indices, :]
        style_tokens = embeddings1[:, self.style_indices, :]
        background_tokens = embeddings1[:, self.background_indices, :]
        logits1 = logits1[:, self.indices1, :]
        return background_tokens, content_tokens, style_tokens, rec_embd, target_ids1, logits1

    # synth task
    def synth(
        self,
        background_tokens,
        content_tokens,
        style_tokens,
        stage="train",
        dataset_type="scb_reconstruct",
        attn_mask=None,
    ):
        query_i = self.query_i.weight[None, :, :].expand(background_tokens.shape[0], -1, -1)
        inputs2, target_ids2 = self.wrap_synth_tokens(
            background_tokens, content_tokens, style_tokens, query_i, stage, dataset_type
        )

        embeddings2, logits2 = self.transformer(inputs2, mode="synth", cat_len=self.cat_len2, attn_mask=attn_mask)
        image_tokens = embeddings2[:, self.image_indices, :]
        logits2 = logits2[:, self.indices2, :]
        return image_tokens, target_ids2, logits2

    def task_edit_content(self, images, texts):
        background_tokens, content_tokens, style_tokens, rec_embd, target_ids1, logits1 = self.decouple(images, texts)
        # replace content tokens with rec_embd
        content_tokens = rec_embd
        image_tokens, target_ids2, logits2 = self.synth(background_tokens, content_tokens, style_tokens, stage="test")
        image_latent = self.conv_o(
            image_tokens.permute(0, 2, 1).contiguous().view(image_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        image_pred = self.vae.decode(image_latent)
        background_latent = self.conv_o(
            background_tokens.permute(0, 2, 1)
            .contiguous()
            .view(background_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        background_pred = self.vae.decode(background_latent)
        return image_pred, background_pred

    def task_edit_style(self, images, texts, cond_images, cond_texts):
        b1, c1, s1, t1, _, _ = self.decouple(images, texts)
        b2, c2, s2, t2, _, _ = self.decouple(cond_images, cond_texts)

        style_batch = torch.cat([s1, s1, s1, s1, s2, s2, s2, s2], dim=0)
        content_batch = torch.cat([t1, t1, t2, t2, t1, t1, t2, t2], dim=0)
        background_batch = torch.cat([b1, b2, b1, b2, b1, b2, b1, b2], dim=0)

        image_tokens, _, _ = self.synth(background_batch, content_batch, style_batch, stage="test")
        image_latent = self.conv_o(
            image_tokens.permute(0, 2, 1).contiguous().view(image_tokens.shape[0], -1, self.conv_h, self.conv_w)
        )
        background_latent = self.conv_o(
            background_batch[:2]
            .permute(0, 2, 1)
            .contiguous()
            .view(background_batch[:2].shape[0], -1, self.conv_h, self.conv_w)
        )
        image_pred = self.vae.decode(image_latent)
        background_pred = self.vae.decode(background_latent)
        return image_pred, background_pred

    def training_step(self, batch, batch_idx, stage="train"):
        # Create contrastive learning positive sample index dictionary
        if not hasattr(self, "positive_indices") and batch["type"] == "scb_reconstruct":
            self.set_positive_indices_dict(batch)
        output = self(batch, stage)
        # decouple loss
        background_mse_loss = F.mse_loss(output["bg_pred"], output["bg_images"])
        perceptual_loss1 = self.perceptual_loss_fn(output["bg_pred"], output["bg_images"])
        rec_loss1 = F.cross_entropy(
            output["logits1"].contiguous().view(-1, self.transformer.config.vocab_size),
            output["target_ids1"].view(-1),
        )

        # # inter loss
        b_features_norm = F.normalize(self.contrastive_head_b(output["background_tokens"]), dim=-1)
        s_features_norm = F.normalize(self.contrastive_head_s(output["style_tokens"]), dim=-1)
        c_features_norm = F.normalize(self.contrastive_head_c(output["content_tokens"]), dim=-1)

        inter_loss = torch.tensor(0.0, device=self.device)
        intra_loss = torch.tensor(0.0, device=self.device)
        if batch["type"] == "scb_reconstruct":
            inter_loss_b = self.inter_sample_loss(b_features_norm, "background")
            inter_loss_s = self.inter_sample_loss(s_features_norm, "style")
            inter_loss_c = self.inter_sample_loss(c_features_norm, "content")

            inter_loss_raw = (inter_loss_b + inter_loss_s + inter_loss_c) / 3
            intra_loss_raw = self.intra_sample_loss(b_features_norm, s_features_norm, c_features_norm)

            # Default to a 1:1 ratio
            inter_loss = inter_loss_raw
            intra_loss = intra_loss_raw
            raw_lambda_intra_ratio = None  # Initialize for logging

            if self.use_adaptive_weight:
                # During the training phase, calculate and update the adaptive weight ratio
                if stage == "train" and len(self.last_layer_inter_intra) > 0:
                    inter_grads = torch.autograd.grad(inter_loss_raw, self.last_layer_inter_intra, retain_graph=True)
                    inter_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in inter_grads])).detach()

                    intra_grads = torch.autograd.grad(intra_loss_raw, self.last_layer_inter_intra, retain_graph=True)
                    intra_grad_norm = torch.norm(torch.stack([torch.norm(g) for g in intra_grads])).detach()

                    clamped_lambda_intra, raw_lambda_intra_ratio = self.calculate_adaptive_weight(
                        inter_grad_norm, intra_grad_norm, "intra"
                    )
                    # Update the stored ratio for use in the validation phase
                    self.lambda_intra_ratio.copy_(clamped_lambda_intra.detach())

                # Use the latest stored ratio in all phases (train/val/test)
                intra_loss = self.lambda_intra_ratio * intra_loss_raw
        else:
            inter_loss_b = inter_loss_s = inter_loss_c = torch.tensor(0.0, device=self.device)

        # # intra loss
        inter_loss = self.disentanglement_weight * inter_loss
        intra_loss = self.disentanglement_weight * intra_loss
        loss1 = 10 * background_mse_loss + perceptual_loss1 + rec_loss1 + inter_loss + intra_loss

        # synth loss
        mask_images = batch.get("mask_images", None)
        if mask_images is not None:
            # Create weight map: text area (mask=1) weight is 2, background (mask=0) weight is 1
            weight_map = 1.0 + mask_images
            # reduction='none' makes mse_loss return a tensor of the same shape as the input, so we can apply weights
            pixel_wise_mse = F.mse_loss(output["image_pred"], output["image"], reduction="none")
            img_mse_loss = (pixel_wise_mse * weight_map).mean()
            # Pass the mask to the custom PerceptualLoss function
            perceptual_loss2 = self.perceptual_loss_fn(output["image_pred"], output["image"], mask=mask_images.float())
        else:
            # --- Standard loss calculation (no mask) ---
            img_mse_loss = F.mse_loss(output["image_pred"], output["image"])
            perceptual_loss2 = self.perceptual_loss_fn(output["image_pred"], output["image"])

        rec_loss2 = F.cross_entropy(
            output["logits2"].contiguous().view(-1, self.transformer.config.vocab_size),
            output["target_ids2"].view(-1),
        )
        loss2 = 10 * img_mse_loss + perceptual_loss2 + rec_loss2

        loss = loss1 + loss2
        if batch["type"] == "scb_reconstruct":
            self.log(f"{stage}/inter_loss_b", inter_loss_b, on_step=True, on_epoch=False, sync_dist=True)
            self.log(f"{stage}/inter_loss_s", inter_loss_s, on_step=True, on_epoch=False, sync_dist=True)
            self.log(f"{stage}/inter_loss_c", inter_loss_c, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/inter_loss", inter_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/intra_loss", intra_loss, on_step=True, on_epoch=False, sync_dist=True)
        if self.use_adaptive_weight:
            if stage == "train" and raw_lambda_intra_ratio is not None:
                self.log(
                    f"{stage}/lambda_intra_raw_ratio",
                    raw_lambda_intra_ratio,
                    on_step=True,
                    on_epoch=False,
                    sync_dist=True,
                )
        self.log(f"{stage}/background_mse_loss", background_mse_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/perceptual_loss1", perceptual_loss1, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/rec_loss1", rec_loss1, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/img_mse_loss", img_mse_loss, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/perceptual_loss2", perceptual_loss2, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/rec_loss2", rec_loss2, on_step=True, on_epoch=False, sync_dist=True)
        self.log(f"{stage}/loss1", loss1, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}/loss2", loss2, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.training_step(batch, batch_idx, stage="val")

    def configure_optimizers(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            torch.nn.LeakyReLU,
            torch.nn.BatchNorm2d,
            torch.nn.InstanceNorm1d,
        )
        param_dict = {}
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                fpn = "%s.%s" % (mn, pn) if mn else pn
                # Exclude decoder and perceptual loss parameters from the main optimizer groups
                if not fpn.startswith("decoder") and not fpn.startswith("perceptual_loss_fn"):
                    if pn.endswith("bias"):
                        no_decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                        decay.add(fpn)
                    elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                        no_decay.add(fpn)
                    elif isinstance(p, nn.Parameter):
                        no_decay.add(fpn)
                    param_dict[fpn] = p

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)

        # --- Create optimizer groups ---
        # Main model parameters
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 1e-5},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        # Freeze vae decoder and perceptual loss parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.perceptual_loss_fn.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1),
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_indices_with_special_tokens(self):
        # output: [<ignore>] [ignore_tokens] [</ignore>] [<c>] [content_tokens] [</c>] [<s>] [style_tokens] [</s>] [<b>] [background_tokens] [</b>]
        indices = []
        content_indices = []
        style_indices = []
        background_indices = []
        i = 0
        indices.append(i)
        i += self.background_len + 1
        indices.append(i)
        i += 1
        indices.append(i)
        # content
        i += 1
        indices.extend(range(i, i + self.content_len))
        content_indices.extend(range(i, i + self.content_len))
        # content
        i += self.content_len
        indices.append(i)
        i += 1
        indices.append(i)
        i += 1
        style_indices.extend(range(i, i + self.style_len))
        i += self.style_len
        indices.append(i)
        i += 1
        indices.append(i)
        i += 1
        background_indices.extend(range(i, i + self.background_len))
        i += self.background_len
        indices.append(i)
        return (
            indices,
            content_indices,
            style_indices,
            background_indices,
            (len(indices), len(content_indices), len(style_indices), len(background_indices)),
        )

    def get_special_tokens(self):
        # output: [<ignore>] [ignore_tokens] [</ignore>] [<i>] [image_tokens] [</i>]
        indices = []
        image_indices = []
        i = 0
        indices.append(i)
        i += self.background_len + 2 + self.content_len + 2 + self.style_len + 1
        indices.append(i)
        i += 1
        indices.append(i)
        i += 1
        image_indices.extend(range(i, i + self.background_len))
        i += self.background_len
        indices.append(i)
        return indices, image_indices

    def expand_token_id(self, token_name, bs):
        return torch.tensor(self.str_converter.name2idx[token_name], device=self.device).expand(bs, 1)

    def expand_token(self, token_name, bs):
        token_idx = torch.tensor(self.str_converter.name2idx[token_name], device=self.device)
        return self.str_embd(token_idx).unsqueeze(0).unsqueeze(1).expand(bs, 1, self.n_embd)

    def wrap_decouple_tokens(self, img_latent, query_c, query_s, query_b, rec_ids):
        # input: [<i>] [image_tokens] [</i>] [<q_c>] [<query_content_tokens>] [</q_c>] [<q_s>] [<query_style_tokens>] [</q_s>] [<q_b>] [<query_background_tokens>] [</q_b>]
        # output: [<ignore>] [ignore_tokens] [</ignore>] [<c>] [content_tokens] [</c>] [<s>] [style_tokens] [</s>] [<b>] [background_tokens] [</b>]
        bs = img_latent.shape[0]

        ig = self.expand_token_id("<ignore>", bs)
        _ig = self.expand_token_id("</ignore>", bs)
        c = self.expand_token_id("<c>", bs)
        _c = self.expand_token_id("</c>", bs)
        s = self.expand_token_id("<s>", bs)
        _s = self.expand_token_id("</s>", bs)
        b = self.expand_token_id("<b>", bs)
        _b = self.expand_token_id("</b>", bs)
        rec_ids = torch.cat([ig, _ig, c, rec_ids, _c, s, _s, b, _b], dim=1)

        i = self.expand_token("<i>", bs)
        _i = self.expand_token("</i>", bs)
        q_c = self.expand_token("<q_c>", bs)
        _q_c = self.expand_token("</q_c>", bs)
        q_s = self.expand_token("<q_s>", bs)
        _q_s = self.expand_token("</q_s>", bs)
        q_b = self.expand_token("<q_b>", bs)
        _q_b = self.expand_token("</q_b>", bs)
        inputs = torch.cat([i, img_latent, _i, q_c, query_c, _q_c, q_s, query_s, _q_s, q_b, query_b, _q_b], dim=1)
        return inputs, rec_ids

    def wrap_synth_tokens(self, background_tokens, content_tokens, style_tokens, query_i, stage, dataset_type):
        # input: [<b>] [background_tokens] [</b>] [<c>] [content_tokens] [</c>] [<s>] [style_tokens] [</s>] [<q_i>] [<query_image_tokens>] [</q_i>]
        # output: [<ignore>] [ignore_tokens] [</ignore>] [<i>] [image_tokens] [</i>]
        bs = background_tokens.shape[0]

        ig = self.expand_token_id("<ignore>", bs)
        _ig = self.expand_token_id("</ignore>", bs)
        i = self.expand_token_id("<i>", bs)
        _i = self.expand_token_id("</i>", bs)
        rec_ids = torch.cat([ig, _ig, i, _i], dim=1)

        b = self.expand_token("<b>", bs)
        _b = self.expand_token("</b>", bs)
        c = self.expand_token("<c>", bs)
        _c = self.expand_token("</c>", bs)
        s = self.expand_token("<s>", bs)
        _s = self.expand_token("</s>", bs)
        q_i = self.expand_token("<q_i>", bs)
        _q_i = self.expand_token("</q_i>", bs)

        if (stage == "train" or stage == "val") and dataset_type == "scb_reconstruct":
            # Get positive sample indices
            pos_b = self.positive_indices["background_difficult"]
            pos_c = self.positive_indices["content_difficult"]
            pos_s = self.positive_indices["style_difficult"]
            # Calculate the average of positive samples
            temp_b = torch.stack([background_tokens[pos_b[i]].mean(dim=0) for i in range(bs)])
            temp_c = torch.stack([content_tokens[pos_c[i]].mean(dim=0) for i in range(bs)])
            temp_s = torch.stack([style_tokens[pos_s[i]].mean(dim=0) for i in range(bs)])
        else:
            temp_b, temp_c, temp_s = background_tokens, content_tokens, style_tokens

        input = torch.cat([b, temp_b, _b, s, temp_s, _s, c, temp_c, _c, q_i, query_i, _q_i], dim=1)
        return input, rec_ids

    def intra_sample_loss(self, b_features, s_features, c_features):
        # [N, D] * [N, D] -> [N, D], then sum over dimension D -> [N]
        sim_bs = torch.sum(b_features * s_features, dim=1)
        sim_bc = torch.sum(b_features * c_features, dim=1)
        sim_sc = torch.sum(s_features * c_features, dim=1)
        # The mean of absolute values is generally more numerically stable than the mean of squares
        loss = (torch.abs(sim_bs) + torch.abs(sim_bc) + torch.abs(sim_sc)).mean()
        return loss

    def inter_sample_loss(self, features, feature_type):
        """
        Calculate inter-group contrastive loss - InfoNCE style implementation (efficient vectorized version)
        This implementation replaces the original for loop with matrix operations and masks,
        significantly improving computational efficiency on the GPU while maintaining
        mathematically equivalent logic.
        """
        N = features.size(0)
        assert N > 1, f"Number of samples for {feature_type} features must be greater than 1"

        # --- 1. Preparation ---
        # Get pre-defined positive pair relationships and temperature parameter from self
        positive_indices_dict = self.positive_indices[feature_type]
        logit_scale = self.logit_scale_inter
        epsilon = 1e-8

        # --- 2. Key step: Create masks instead of loop conditions ---
        # a. Create a boolean mask for positive pairs (pos_mask)
        # pos_mask[i, j] = True if (i, j) is a positive pair
        pos_mask = torch.zeros(N, N, dtype=torch.bool, device=features.device)
        for anchor_idx, pos_indices in positive_indices_dict.items():
            if pos_indices:
                # Ensure indices are within a valid range
                valid_pos_indices = [pi for pi in pos_indices if pi < N]
                if valid_pos_indices:
                    pos_mask[anchor_idx, valid_pos_indices] = True
        # b. If there are no valid positive pairs in the entire batch, return 0 loss directly
        num_positive_pairs = pos_mask.sum()
        if num_positive_pairs == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        # c. Create masks to exclude self and negative samples
        self_mask = torch.eye(N, dtype=torch.bool, device=features.device)
        neg_mask = ~pos_mask & ~self_mask

        # --- 3. Core calculation: Use matrix operations instead of loops ---
        # a. Calculate the full similarity matrix (logits) and take the exponent
        sim_matrix = torch.matmul(features, features.t()) * logit_scale
        exp_sim = torch.exp(sim_matrix)
        # b. Calculate the sum of exp(sim) for all negative samples corresponding to each anchor at once
        # neg_exp_sum_per_anchor[i] stores the sum of exp(sim) for all negative samples of the i-th sample
        neg_exp_sum_per_anchor = (exp_sim * neg_mask.float()).sum(dim=1)
        # c. Extract the exp(sim) values of all positive pairs as numerators
        numerators = exp_sim[pos_mask]
        # d. Find the corresponding neg_exp_sum in the denominator for each numerator
        #    Align each positive pair with its anchor's negative sum through advanced indexing
        anchor_indices_of_pos_pairs = torch.where(pos_mask)[0]
        corresponding_neg_sums = neg_exp_sum_per_anchor[anchor_indices_of_pos_pairs]
        # e. Calculate the loss for each positive pair
        denominators = numerators + corresponding_neg_sums
        pair_losses = -torch.log(numerators / (denominators + epsilon))

        # --- 4. Return the average loss of all positive pairs ---
        final_loss = pair_losses.sum() / num_positive_pairs
        return final_loss

    def _get_linear_idx(self, font_coord, text_coord, bg_coord):
        """Converts 3D coordinates (font_idx, text_idx, bg_idx) to a linear index."""
        # self.text_num and self.bg_num are set in the class's __init__ or set_positive_indices_dict
        return font_coord * (self.text_num * self.bg_num) + text_coord * self.bg_num + bg_coord

    def _precompute_replacement_map(self):
        """
        Precomputes the feature replacement map within a single "large sample" (contrastive group).
        This map is based on local indices relative to the start of the large sample.
        """
        self._precomputed_replacement_map = {}

        # Internal helper function to get local linear index during precomputation
        def get_local_linear_idx_in_precompute(font_coord, text_coord, bg_coord):
            return font_coord * (self.text_num * self.bg_num) + text_coord * self.bg_num + bg_coord

        for i in range(self.font_num):
            for j in range(self.text_num):
                for k in range(self.bg_num):
                    local_original_idx = get_local_linear_idx_in_precompute(i, j, k)

                    s_replacements = []
                    c_replacements = []
                    b_replacements = []

                    # Style replacements (keep font_idx=i constant, change text_idx and bg_idx)
                    if self.text_num > 1 and self.bg_num > 1:
                        for other_j in range(self.text_num):
                            if other_j == j:
                                continue
                            for other_k in range(self.bg_num):
                                if other_k == k:
                                    continue
                                s_replacements.append(get_local_linear_idx_in_precompute(i, other_j, other_k))

                    # Content replacements (keep text_idx=j constant, change font_idx and bg_idx)
                    if self.font_num > 1 and self.bg_num > 1:
                        for other_i in range(self.font_num):
                            if other_i == i:
                                continue
                            for other_k in range(self.bg_num):
                                if other_k == k:
                                    continue
                                c_replacements.append(get_local_linear_idx_in_precompute(other_i, j, other_k))

                    # Background replacements (keep bg_idx=k constant, change font_idx and text_idx)
                    if self.font_num > 1 and self.text_num > 1:
                        for other_i in range(self.font_num):
                            if other_i == i:
                                continue
                            for other_j in range(self.text_num):
                                if other_j == j:
                                    continue
                                b_replacements.append(get_local_linear_idx_in_precompute(other_i, other_j, k))

                    self._precomputed_replacement_map[local_original_idx] = {
                        "s_replacements": s_replacements,
                        "c_replacements": c_replacements,
                        "b_replacements": b_replacements,
                    }

    def set_positive_indices_dict(self, batch):
        self.positive_indices = {}  # Clear previous calculation results

        # Calculate parameters - avoid multiple calls
        # If the num_scb dimension changes (or it's the first call), re-initialize and precompute
        if (
            not hasattr(self, "font_num")
            or self.font_num is None
            or self.font_num != batch["num_scb"][0]
            or self.text_num != batch["num_scb"][1]
            or self.bg_num != batch["num_scb"][2]
        ):
            self.font_num, self.text_num, self.bg_num = batch["num_scb"]
            self.num_images_per_group = self.font_num * self.text_num * self.bg_num
            self._precompute_replacement_map()  # Only re-precompute when dimensions change

        self.batch_size = batch["batch_size"]  # batch_size can change with each call

        # Get from instance attributes to ensure consistency
        batch_size = self.batch_size
        font_num = self.font_num
        text_num = self.text_num
        bg_num = self.bg_num
        num_images = font_num * text_num * bg_num

        # Initialize dictionaries for different feature types
        all_feature_types = ["background", "style", "content"]
        for feature_type in all_feature_types:
            self.positive_indices[feature_type] = {}
            self.positive_indices[feature_type + "_include_self"] = {}
            self.positive_indices[feature_type + "_difficult"] = {}  # For storing custom replacement indices

        # Define a nested helper function
        # It will receive a list containing (global index, original font_idx, original text_idx, original bg_idx)
        # This allows it to internally calculate the three types of positive sample indices for each sample
        def select_positive_indices_dict(
            current_feature_type, grouped_items_with_coords  # Format: [(global_idx, f_orig, t_orig, b_orig), ...]
        ):
            # Get references to the corresponding positive sample dictionaries
            positive_indices_dict_normal = self.positive_indices[current_feature_type]
            positive_indices_dict_include_self = self.positive_indices[current_feature_type + "_include_self"]
            positive_indices_dict_difficult = self.positive_indices[current_feature_type + "_difficult"]

            # Iterate through each sample in the current group
            for global_idx, f_orig, t_orig, b_orig in grouped_items_with_coords:
                # 1. Calculate regular positive sample indices (excluding self)
                positive_indices_dict_normal[global_idx] = [
                    item[0] for item in grouped_items_with_coords if item[0] != global_idx
                ]

                # 2. Calculate positive sample indices including self
                positive_indices_dict_include_self[global_idx] = [item[0] for item in grouped_items_with_coords]

                # 3. Calculate 'difficult' (replacement source) indices
                # Use the precomputed _precomputed_replacement_map to get them.
                # local_original_idx is the relative index of the current sample within a single large sample (contrastive group)
                local_original_idx = self._get_linear_idx(f_orig, t_orig, b_orig)

                # Determine the starting offset of the large sample to which the current item belongs
                batch_offset_for_current_item = global_idx - local_original_idx

                repl_local_indices = []
                if current_feature_type == "background":
                    repl_local_indices = self._precomputed_replacement_map[local_original_idx]["b_replacements"]
                elif current_feature_type == "style":
                    repl_local_indices = self._precomputed_replacement_map[local_original_idx]["s_replacements"]
                elif current_feature_type == "content":
                    repl_local_indices = self._precomputed_replacement_map[local_original_idx]["c_replacements"]

                # Convert local replacement indices to global indices and store them
                positive_indices_dict_difficult[global_idx] = [
                    batch_offset_for_current_item + idx for idx in repl_local_indices
                ]

        # Iterate through each "large sample" (contrastive group) in the batch
        for batch_offset_idx in range(batch_size):
            batch_offset = batch_offset_idx * num_images

            # --- Process 'background' type positive sample indices ---
            current_grouped_items_with_coords = []
            # Iterate through all samples sharing the same background_idx (within a large sample)
            for bg_idx_orig in range(bg_num):
                for font_idx_orig in range(font_num):
                    for text_idx_orig in range(text_num):
                        global_idx = batch_offset + self._get_linear_idx(font_idx_orig, text_idx_orig, bg_idx_orig)
                        current_grouped_items_with_coords.append(
                            (global_idx, font_idx_orig, text_idx_orig, bg_idx_orig)
                        )
                # Call the helper function for all samples corresponding to the current bg_idx_orig
                select_positive_indices_dict("background", current_grouped_items_with_coords)
                current_grouped_items_with_coords = []  # Reset the list for the next bg_idx

            # --- Process 'style' type positive sample indices ---
            current_grouped_items_with_coords = []
            # Iterate through all samples sharing the same font_idx (within a large sample)
            for font_idx_orig in range(font_num):
                for text_idx_orig in range(text_num):
                    for bg_idx_orig in range(bg_num):
                        global_idx = batch_offset + self._get_linear_idx(font_idx_orig, text_idx_orig, bg_idx_orig)
                        current_grouped_items_with_coords.append(
                            (global_idx, font_idx_orig, text_idx_orig, bg_idx_orig)
                        )
                # Call the helper function for all samples corresponding to the current font_idx_orig
                select_positive_indices_dict("style", current_grouped_items_with_coords)
                current_grouped_items_with_coords = []  # Reset the list for the next font_idx

            # --- Process 'content' type positive sample indices ---
            current_grouped_items_with_coords = []
            # Iterate through all samples sharing the same text_idx (within a large sample)
            for text_idx_orig in range(text_num):
                for font_idx_orig in range(font_num):
                    for bg_idx_orig in range(bg_num):
                        global_idx = batch_offset + self._get_linear_idx(font_idx_orig, text_idx_orig, bg_idx_orig)
                        current_grouped_items_with_coords.append(
                            (global_idx, font_idx_orig, text_idx_orig, bg_idx_orig)
                        )
                # Call the helper function for all samples corresponding to the current text_idx_orig
                select_positive_indices_dict("content", current_grouped_items_with_coords)
                current_grouped_items_with_coords = []  # Reset the list for the next text_idx
