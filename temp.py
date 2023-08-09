import torch
import torch.nn as nn
from torch.optim import Adam
from torch.cuda.amp import GradScaler
from collections import OrderedDict
from pathlib import Path

import config
from model import Generator, Discriminator
from utils import (
    save_checkpoint,
)


def _get_state_dict(
    state_dict
):
    new_state_dict = OrderedDict()
    for old_key in list(state_dict.keys()):
        if old_key and old_key.startswith("module."):
            new_key = old_key[len("module."):]
        else:
            new_key = old_key
        new_state_dict[new_key] = state_dict[old_key]
    return new_state_dict


disc = Discriminator()
gen = Generator()

disc_optim = Adam(
    params=disc.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.ADAM_EPS
)
gen_optim = Adam(
    params=gen.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2), eps=config.ADAM_EPS
)

disc_scaler = GradScaler()
gen_scaler = GradScaler()

DEVICE = torch.device("cuda")
if config.CKPT_PATH is not None:
    ckpt = torch.load(config.CKPT_PATH, map_location=DEVICE)
    # disc.load_state_dict(ckpt["D"])
    # gen.load_state_dict(ckpt["G"])
    disc.load_state_dict(_get_state_dict(ckpt["D"]))
    gen.load_state_dict(_get_state_dict(ckpt["G"]))
    disc_optim.load_state_dict(ckpt["D_optimizer"])
    gen_optim.load_state_dict(ckpt["G_optimizer"])

step = config.STEP if config.STEP is not None else ckpt["step"]
trans_phase = config.TRANS_PHASE if config.TRANS_PHASE is not None else ckpt["transition_phase"]
resol_idx = config.RESOL_IDX if config.RESOL_IDX is not None else ckpt["resolution_index"]

save_path=config.CKPT_PATH
Path(save_path).parent.mkdir(parents=True, exist_ok=True)

disc = nn.DataParallel(disc)
gen = nn.DataParallel(gen)
ckpt = {
    "resolution_index": resol_idx,
    "step": step,
    "transition_phase": trans_phase,
    "D_optimizer": disc_optim.state_dict(),
    "G_optimizer": gen_optim.state_dict(),
    "D_scaler": disc_scaler.state_dict(),
    "G_scaler": gen_scaler.state_dict(),
}
ckpt["D"] = disc.module.state_dict()
ckpt["G"] = gen.module.state_dict()
print(ckpt["D"].keys())
# ckpt["D"] = disc.state_dict()
# ckpt["G"] = gen.state_dict()

torch.save(ckpt, str(save_path))