import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math

# hyperparameters
img_size       = 512     # height and width of each image

# CNN values
patch_size     = 32      # each patch is patch_size x patch_size pixels
stride         = patch_size // 2  # stride for overlapping patches
padding        = patch_size // 4  # padding for overlapping patches

# transformer values
n_embd         = 512     # transformer embedding dimension
n_head         = 8       # number of attention heads
n_layer        = 6       # number of transformer blocks
dropout        = 0.1     # dropout rate
batch_size     = 16      # images per training batch
max_iters      = 2000    # total training steps
eval_interval  = 100     # how often to calculate loss
learning_rate  = 3e-4    # learning rate (AdamW optimizer)
dataset_size   = 2000    # number of synthetic images to generate
save_path      = 'models/depth_model_overlapping.pth'
device         = 'cuda' if torch.cuda.is_available() else 'cpu'

# calcuate number of patches per image based on the patch size, stride, and padding
def calc_n_patches(img_size, patch_size, stride, padding):
    size   = (img_size - patch_size + 2 * padding) // stride + 1
    return size * size

torch.manual_seed(1337)

torch.backends.cudnn.enabled   = False
torch.backends.cudnn.benchmark = False

# Generates random synthetic scenes on-the-fly.
# Each scene is an RGB image with coloured shapes drawn at known
# depths.  The ground-truth is a single-channel depth map where
# 1.0 = closest and 0.0 = furthest away.

# use data generator from other file, but with the new constructor that takes img_size as an argument
from data_generator import SyntheticDepthDataset

# split into train / val
full_dataset = SyntheticDepthDataset(length=dataset_size, img_size=img_size)
n_val        = max(1, dataset_size // 10)
n_train      = dataset_size - n_val
train_ds, val_ds = torch.utils.data.random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(0)
)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)


# Split image into flat sequence of patch tokens

class PatchEmbed(nn.Module):
    """
    Cuts the image into non-overlapping patches and projects each patch
    into an n_embd-dimensional vector.
    (B, 3, H, W)  →  (B, n_patches, n_embd)
    """

    def __init__(self):
        super().__init__()
        # one conv with kernel=stride=patch_size gives exactly one vector per patch
        # self.proj = nn.Conv2d(3, n_embd, kernel_size=patch_size, stride=patch_size)
        self.proj = nn.Conv2d(3, n_embd, kernel_size=patch_size, stride=patch_size//2, padding=patch_size//4)

    def forward(self, x):
        x = self.proj(x)                    # (B, n_embd, h, w)  where h=H/patch_size
        B, D, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)    # (B, h*w, n_embd)
        return x, h, w


# One self-attention head

class Head(nn.Module):
    """
    Single head of self-attention (no causal mask - every patch
    can attend to every other patch, since this is vision not text).
    """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        
        k = self.key(x)    # project each token to a key vector (B, T, head_size)
        q = self.query(x)  # project each token to a query vector (B, T, head_size)

        # attention scores, scaled to produce more stable gradients for softmax
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.drop(wei)
        v   = self.value(x)    # (B, T, head_size)
        out = wei @ v          # (B, T, head_size)
        return out


# Multi-head attention

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention running in parallel. """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(head_size * num_heads, n_embd)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.drop(self.proj(out))
        return out


# Feed-forward block

class Encoder(nn.Module):
    """ Simple two-layer MLP (Multi Layer Perceptron) applied to each token independently. """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # embed to embed * 4 size
            nn.Linear(n_embd, 4 * n_embd),
            # GELU activation function is a smoother version of ReLU
            nn.GELU(),
            # back to embed size
            nn.Linear(4 * n_embd, n_embd),
            # dropout
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# One full Transformer block

class Block(nn.Module):
    """
    Transformer block: self-attention (communication between patches)
    followed by feed-forward (independent per-patch computation).
    Uses pre-norm (LayerNorm before each sub-layer).
    """

    def __init__(self):
        super().__init__()
        head_size       = n_embd // n_head
        self.attention  = MultiHeadAttention(n_head, head_size)
        self.encoder    = Encoder()
        self.ln1        = nn.LayerNorm(n_embd)
        self.ln2        = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))     # residual + attention
        x = x + self.encoder(self.ln2(x))   # residual + MLP
        return x


# CNN decoder: upsample patch grid back to full image resolution

class ConvDecoder(nn.Module):
    """
    Takes the (B, n_embd, h, w) feature map produced by the transformer
    and upsamples it back to (B, 1, H, W) using transposed convolutions.
    Each stage doubles the spatial size.
    """

    def __init__(self, h):
        super().__init__()
        n_stages = int(math.log2(img_size // h))  # how many doublings needed

        channels = [n_embd] + [max(16, n_embd // (2**i)) for i in range(1, n_stages + 1)]
        # e.g. [768, 384, 192, 96, 48]

        layers = []
        for i in range(n_stages):
            layers += [
                nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2),
                nn.BatchNorm2d(channels[i+1]),
                nn.ReLU(True),
            ]
        layers += [nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1), nn.Sigmoid()]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



# Full model


class DepthTransformer(nn.Module):
    """
    Full 2D-to-3D model.

    Pipeline:
      image input
      patch tokens
      positional embeddings
      N transformer blocks  
      reshape to 2-D grid
      CNN decoder
      depth map
    """

    def __init__(self):
        super().__init__()

        n_patches = calc_n_patches(img_size, patch_size, stride, padding)

        self.patch_embed = PatchEmbed() # Patch embedder module

        # position embeddings, initialized as random normal
        self.pos_embed   = nn.Parameter(torch.randn(1, n_patches, n_embd) * 0.02) 

        # initialize stack of transformer blocks depending on n_layer
        self.blocks      = nn.Sequential(*[Block() for _ in range(n_layer)])

        # final layer norm before the CNN decoder
        self.ln_f        = nn.LayerNorm(n_embd)

        # decoder module to upsample back to image resolution
        h                = int(math.sqrt(n_patches))  # spatial size of feature map
        self.decoder     = ConvDecoder(h)

        # initialise weights with a normal distribution, and zero the biases
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialise Linear and Embedding layers with a normal distribution, and zero the biases
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        # x       : (B, 3, H, W)
        # targets : (B, 1, H, W)  depth ground-truth for loss calculation

        tokens, h, w = self.patch_embed(x)                      # (B, N, n_embd)
        tokens = tokens + self.pos_embed                        # add position info
        tokens = self.blocks(tokens)                            # transformer blocks
        tokens = self.ln_f(tokens)                              # final norm

        # reshape token sequence back into a spatial grid for the CNN decoder
        B, N, D = tokens.shape
        feat     = tokens.transpose(1, 2).reshape(B, D, h, w)  # (B, n_embd, h, w)
        depth    = self.decoder(feat)                           # (B, 1, H, W)

        loss = None
        if targets is not None:
            loss = depth_loss(depth, targets)

        return depth, loss

    def predict(self, x):
        """ Convenience wrapper: run forward without targets. """
        depth, _ = self(x, targets=None)
        return depth



# Loss function
# L1 loss + gradient loss (encourages sharp depth edges)


def gradient_loss(pred, gt):
    def grad(t):
        dy = t[:, :, 1:, :] - t[:, :, :-1, :]   # vertical differences
        dx = t[:, :, :, 1:] - t[:, :, :, :-1]   # horizontal differences
        return dy, dx
    pred_dy, pred_dx = grad(pred)
    gt_dy,   gt_dx   = grad(gt)
    return (pred_dy - gt_dy).abs().mean() + (pred_dx - gt_dx).abs().mean()

def depth_loss(pred, gt, alpha=0.5):
    return F.l1_loss(pred, gt) + alpha * gradient_loss(pred, gt)



# Loss estimation helper (Karpathy's estimate_loss)

@torch.no_grad()
def estimate_loss():
    out = {}
    # disables dropout and switches BatchNorm to use running stats rather than batch stats
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = []
        for rgb, depth_gt in loader:
            rgb, depth_gt = rgb.to(device), depth_gt.to(device)
            _, loss = model(rgb, depth_gt)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# Instantiate model and optimizer

if __name__ == '__main__':
    model     = DepthTransformer()
    model     = model.to(device)

    # AdamW optimizer is a version of Adam with better regularization properties
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters, eta_min=1e-6)

    print(sum(p.numel() for p in model.parameters()), 'parameters')


    # Training loop

    # cycle the dataloader infinitely to use a step-based loop
    def infinite(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite(train_loader)

    best_val = float('inf')

    for step in range(max_iters):

        # evaluate on train and val every eval_interval steps
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            if losses['val'] < best_val:
                best_val = losses['val']
                torch.save(model, save_path)
                print(f"  -> saved best model to {save_path}")

        # sample a batch and take one gradient step
        rgb, depth_gt = next(data_iter)
        rgb, depth_gt = rgb.to(device), depth_gt.to(device)

        # compute loss
        _, loss = model(rgb, depth_gt)

        # clear old gradients from the last step
        optimizer.zero_grad(set_to_none=True)

        # backpropagate and update weights
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print(f"\nTraining complete. Best val loss: {best_val:.4f}")
    print(f"Model saved to: {save_path}")
