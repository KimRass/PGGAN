RESOLS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

### Data
DATA_DIR = "/home/ubuntu/project/cv/celebahq/celeba_hq" # DO NOT MODIFY

### Dataloader
N_WORKERS = 4 # DO NOT MODIFY
AUTOCAST = True # DO NOT MODIFY
# RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 14, 512: 6, 1024: 3} # In the paper
RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 9, 256: 3, 512: 3, 1024: 3} # In my case

### Loss
LAMBDA = 10 # DO NOT MODIFY
LOSS_EPS = 0.001 # DO NOT MODIFY

### Adam optimizer
LR = 0.001 # DO NOT MODIFY
BETA1 = 0 # DO NOT MODIFY
BETA2 = 0.99 # DO NOT MODIFY
ADAM_EPS = 1e-8 # DO NOT MODIFY

### Training
N_PRINT_STEPS = 1000
N_CKPT_STEPS = 4000
# "We start with 4×4 resolution and train the networks until we have shown the discriminator
# 800k real images in total. We then alternate between two phases: fade in the first 3-layer block
# during the next 800k images, stabilize the networks for 800k images, fade in the next 3-layer block
# during 800k images, etc."
RESOL_N_IMAGES = {
    4: 200_000,
    8: 200_000,
    16: 400_000,
    32: 400_000,
    64: 800_000,
    128: 800_000,
    256: 800_000,
    512: 800_000,
    1024: 800_000,
}
# RESOL_N_IMAGES = {4: 200_000, 8: 200_000, 16: 400_000, 32: 400_000, 64: 800_000, 128: 1_600_000}

### Checkpoint
CKPT_PATH = "/home/ubuntu/project/cv/pggan_from_scratch/checkpoints/256×256to512×512_60000.pth"
STEP = None
TRANS_PHASE = None
RESOL_IDX = None
