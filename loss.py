# References:
    # https://engineer-mole.tistory.com/m/52
    # https://arxiv.org/pdf/1704.00028.pdf

import torch


# "We use the improved Wasserstein loss."
def get_gradient_penalty(disc, resol, alpha, real_image, fake_image):
    eps = torch.rand(1, device=real_image.device)
    avg_image = eps * real_image + (1 - eps) * fake_image
    avg_image.requires_grad = True
    avg_pred = disc(avg_image, resol=resol, alpha=alpha)

    real_label = torch.ones_like(avg_pred, device=avg_pred.device)
    grad = torch.autograd.grad(
        outputs=avg_pred, inputs=avg_image, grad_outputs=real_label, create_graph=True, retain_graph=True
    )[0]
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp
