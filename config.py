RESOLS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

### Dataloader
N_WORKERS = 4
AUTOCAST = True
# RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 14, 512: 6, 1024: 3} # In the paper
RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 9, 256: 6, 512: 6, 1024: 3} # In my case

### Loss
LAMBDA = 10
LOSS_EPS = 0.001

### Adam optimizer
LR = 0.001
BETA1 = 0
BETA2 = 0.99
ADAM_EPS = 1e-8

# Directory
# DATA_DIR = "/Users/jongbeomkim/Documents/datasets/celebahq/"
DATA_DIR = "/home/ubuntu/project/celebahq/celeba_hq"

N_IMG_STEPS = 1000
N_CKPT_STEPS = 4000

# "We start with 4×4 resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."
RESOL_N_IMAGES = {
    4: 200_000, 8: 200_000, 16: 400_000, 32: 400_000, 64: 800_000, 128: 800_000, 256: 800_000
}
# RESOL_N_IMAGES = {4: 200_000, 8: 200_000, 16: 400_000, 32: 400_000, 64: 800_000, 128: 1_600_000}

### Resume
CKPT_PATH = "/home/ubuntu/project/pggan_from_scratch/pretrained/128×128_184000.pth"
STEP = 0
TRANS_PHASE = True
# RESOL_IDX = 6
RESOL_IDX = 0
