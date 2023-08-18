# Paper Reading
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}}\Big[(\Vert \nabla_{\hat{x}}D(\hat{x}) \Vert_{2} - 1)^{2}\Big]$$

# Training
- Number of training images (including duplicates): 800,000 for each resolution

| Resolution | Batch size | Time | Number of steps | Total time |
| - | - | - | - | - |
| 4×4 | 16 |  | 50,000 |  |
| 4×4 to 8×8 | 16 |  | 50,000 |  |
| 8×8 | 16 |  | 50,000 |  |
| 8×8 to 16×16 | 16 |  | 50,000 |  |
| 16×16 | 16 |  | 50,000 |  |
| 16×16 to 32×32 | 16 |  | 50,000 |  |
| 32×32 | 16 |  | 50,000 |  |
| 32×32 to 64×64 | 16 |  | 50,000 |  |
| 64×64 | 16 |  | 50,000 |  |
| 64×64 to 128×128 | 9 |  | 88888 |  |
| 128×128 | 9 |  | 88,888 |
| 128×128 to 256×256 | 3 |  / 1,000 steps | 266,666 |
| 256×256 | 3 |  / 1,000 steps | 266,666 |
| 256×256 to 512×512 | 3 | 0:14:58 / 1,000 steps | 266,666 | 66:30:00 |
| 512×512 | 3 | 0:15:58 / 1,000 steps | 266,666 | 66:30:00 |
| 512×512 to 1,024×1,024 | - | - | - | - |
| 1,024×1,024 | - | - | - |

# Research
## 23.08.03
- 이전까지는 Celeb-A HQ의 Training set에 대해 Mean과 Std를 계산해서 이걸 가지고 Normalize했습니다.
    - `T.Normalize(mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269))`
- 논문에서는 "We represent training and generated images in $[-1, 1]$."라는 표현이 있는 것을 알고 있었으나 이를 간과하고 있었습니다.
- 다음과 같이 코드를 수정했습니다.
    - `T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`
## 23.08.17
- 해상도 1,024×1,024로 진입하면서 처음으로 다음 에러가 발생하는 것을 `torch.autograd.set_detect_anomaly(True)`을 통해 확인했습니다.
```
/home/ubuntu/.local/lib/python3.8/site-packages/torch/autograd/__init__.py:200: UserWarning: Error detected in DivBackward0. Traceback of forward call that caused the error:
 (Triggered internally at ../torch/csrc/autograd/python_anomaly_mode.cpp:114.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/home/ubuntu/.local/lib/python3.8/site-packages/torch/autograd/__init__.py:200: UserWarning: 

Previous calculation was induced by StdBackward0. Traceback of forward call that induced the previous calculation:
  File "train.py", line 172, in <module>
    gp = get_gradient_penalty(
  File "/home/ubuntu/project/cv/pggan_from_scratch/loss.py", line 13, in get_gradient_penalty
    avg_pred = disc(avg_image, img_size=img_size, alpha=alpha)
  File "/home/ubuntu/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/project/cv/pggan_from_scratch/model.py", line 181, in forward
    x = eval(f"""self.block{d}""")(x)
  File "/home/ubuntu/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/ubuntu/project/cv/pggan_from_scratch/model.py", line 105, in forward
    x = self.add_minibatch_std(x)
  File "/home/ubuntu/project/cv/pggan_from_scratch/model.py", line 97, in add_minibatch_std
    feat_map = x.std(dim=0, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)
 (Triggered internally at ../torch/csrc/autograd/python_anomaly_mode.cpp:121.)
  Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
Traceback (most recent call last):
  File "train.py", line 191, in <module>
    disc_loss.backward()
  File "/home/ubuntu/.local/lib/python3.8/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/home/ubuntu/.local/lib/python3.8/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: Function 'DivBackward0' returned nan values in its 0th output.
```
- `# feat_map = x.std(dim=0, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)`에서 발생한 것으로 생각돼 `feat_map = x.std(dim=0, correction=0, keepdim=True).mean(dim=(1, 2, 3), keepdim=True)`으로 수정해봤지만 에러가 여전히 발생하는 것을 확인했습니다.
