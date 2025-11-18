import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class StrLabelConverter(object):
    def __init__(self, alphabet, max_text_len, start_id=0, specials=None):

        self.max_text_len = max_text_len
        if alphabet.endswith(".txt"):
            with open(alphabet, "r", encoding="utf-8") as f:
                alphabet = f.read()
        else:
            raise NotImplementedError("Only .txt alphabet files are supported")

        if specials.endswith(".txt"):
            with open(specials, "r", encoding="utf-8") as f:
                specials = [line.strip() for line in f if line.strip()]
        else:
            raise NotImplementedError("Only .txt special tokens files are supported")

        self.alphabet = alphabet
        self.specials = specials
        self.start_id = start_id
        self.specials = specials if specials is not None else []
        self.idx2name = []
        self.name2idx = {}

        # Add normal characters
        for char in self.alphabet:
            self.name2idx[char] = len(self.idx2name) + self.start_id
            self.idx2name.append(char)

        # Add special characters
        for special in self.specials:
            self.name2idx[special] = len(self.idx2name) + self.start_id
            self.idx2name.append(special)

        # For easy access to pad_id, etc.
        self.pad_id = self.name2idx.get("<pad>", None)

    def encode(self, text):
        if isinstance(text, str):
            try:
                text = [self.name2idx[char] for char in text]
            except KeyError:
                new_text = ""
                for c in text:
                    if c in self.alphabet:
                        new_text += c
                text = [self.name2idx[char] for char in new_text]
            length = min(len(text), self.max_text_len)

            text = text[: self.max_text_len]
            text = text + [self.pad_id] * (self.max_text_len - length)
            return text, length

        elif isinstance(text, list):
            rec = []
            length = []
            for t in text:
                t, l = self.encode(t)
                rec.append(t)
                length.append(l)

            return torch.tensor(rec), length

    def decode(self, t, scires=None):
        if t.ndim == 1:
            try:
                str = ""
                for c in t:
                    if int(c) == self.pad_id:
                        continue
                    str += self.idx2name[int(c) - self.start_id]
                return str
            except IndexError:
                return "###"
        elif t.ndim == 2:
            results = []
            for i in range(t.shape[0]):
                results.append(self.decode(t[i]))
            return results


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True, allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)
            std = flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3).permute(1, 0, 2, 3)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class SimpleProjection(nn.Module):
    def __init__(self, input_dim, output_dim, sequence_len):
        super().__init__()
        self.sequence_len = sequence_len
        self.projection = nn.Sequential(
            nn.InstanceNorm1d(sequence_len),
            nn.Linear(input_dim, output_dim, bias=False),
            nn.Conv1d(in_channels=sequence_len, out_channels=1, kernel_size=1),
        )

    def forward(self, x):
        if x.size(1) > self.sequence_len:
            x = x[:, x.size(1) - self.sequence_len :, :]
        return self.projection(x).squeeze(1)


class StandardProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2), nn.SiLU(), nn.Linear(input_dim * 2, output_dim)
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.norm(x)  # Input x shape: [batch_size, sequence_len, input_dim]
        x_pooled = torch.mean(x, dim=1)  # Output shape: [batch_size, input_dim]
        output = self.projection(x_pooled)  # Output shape: [batch_size, output_dim]
        return output


class SimpleEncoder(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.model = nn.Sequential(
            # Input: B, 3, 64, 256
            nn.Conv2d(3, n_embd // 4, kernel_size=3, stride=2, padding=1),  # B, n_embd/4, 32, 128
            nn.GroupNorm(32, n_embd // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_embd // 4, n_embd // 2, kernel_size=3, stride=2, padding=1),  # B, n_embd/2, 16, 64
            nn.GroupNorm(32, n_embd // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(n_embd // 2, n_embd, kernel_size=3, stride=2, padding=1),  # B, n_embd, 8, 32
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class DecoderHead(nn.Module):
    def __init__(self, in_channels, out_channels=4, mid_channels_factor=2):
        super().__init__()
        mid_channels = in_channels // mid_channels_factor
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features
        self.layers = layers if layers else [3, 8, 15, 22]  # Default layers to use for perceptual loss
        self.vgg = nn.Sequential(*[vgg[i] for i in range(max(self.layers) + 1)])
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, img1, img2, mask=None):
        # Rescale images from [-1, 1] to [0, 1] for VGG
        img1 = (img1 + 1.0) / 2.0
        img2 = (img2 + 1.0) / 2.0
        features1 = self.extract_features(img1)
        features2 = self.extract_features(img2)
        loss = 0
        for f1, f2 in zip(features1, features2):
            if mask is not None:
                # Downsample mask to the size of the feature map
                downsampled_mask = F.interpolate(mask, size=f1.shape[2:], mode="bilinear", align_corners=False)
                # Create weight map (e.g., text weight=2, bg weight=1)
                weight_map = 1.0 + downsampled_mask
                # Calculate weighted MSE on features
                layer_loss = (((f1 - f2) ** 2) * weight_map).mean()
                loss += layer_loss
            else:
                # Original unweighted loss
                loss += F.mse_loss(f1, f2)
        return loss

    def extract_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if i in self.layers:
                features.append(x)
        return features
