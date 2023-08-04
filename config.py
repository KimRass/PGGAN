### Directory
DATA_DIR = "/home/ubuntu/project/celebahq/celeba_hq"

N_IMG_STEPS = 1000
N_CKPT_STEPS = 4000

# RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 14, 512: 6, 1024: 3} # In the paper
# RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 16, 256: 9, 512: 6, 1024: 3} # In my case
RESOL_BATCH_SIZE = {4: 16, 8: 16, 16: 16, 32: 16, 64: 16, 128: 9, 256: 9, 512: 6, 1024: 3} # In my case

RESOL_N_IMAGES = {4: 200_000, 8: 200_000, 16: 400_000, 32: 400_000, 64: 800_000, 128: 1_600_000}

### Loss
LAMBDA = 10
LOSS_EPS = 0.001
RESOLS = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
N_WORKERS = 4
AUTOCAST = True

### Adam optimizer
LR = 0.001
BETA1 = 0
BETA2 = 0.99
ADAM_EPS = 1e-8

### Resume