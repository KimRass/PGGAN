# References:
    # https://engineer-mole.tistory.com/m/52

import torch


# "We use the improved Wasserstein loss."
def get_gradient_penalty(disc, img_size, alpha, real_image, fake_image):
    eps = torch.rand((real_image.size(0), 1, 1, 1), device=real_image.device)
    inter_image = eps * real_image + (1 - eps) * fake_image
    inter_image.requires_grad = True
    inter_pred = disc(inter_image, img_size=img_size, alpha=alpha)

    real_label = torch.ones_like(inter_pred, device=inter_pred.device)
    grad = torch.autograd.grad(
        outputs=inter_pred, inputs=inter_image, grad_outputs=real_label, create_graph=True, retain_graph=True,
    )[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp
