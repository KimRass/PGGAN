# References:
    # https://engineer-mole.tistory.com/m/52
    # https://arxiv.org/pdf/1704.00028.pdf

import torch

# "We use the improved Wasserstein loss."
def gp(disc, real_image, fake_image):
    real_image = torch.randn((4, 3, 512, 512))
    fake_image = torch.randn((4, 3, 512, 512))

    b, _, _, _ = real_image.shape
    eps = torch.rand(1, device=real_image.device)
    avg_image = eps * real_image + (1 - eps) * fake_image
    avg_pred = disc(avg_image)

    real_label = torch.ones(size=(b, 1), device=real_image.device)
    grad, _ = torch.autograd.grad(
        outputs=avg_pred, inputs=avg_image, grad_outputs=real_label, create_graph=True, retain_graph=True
    )
    grad = grad.view(grad.size(0), -1)
    gp = ((grad.norm(2, dim=1) - 1) ** 2).mean()
    return gp