# Paper Reading
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}}\Big[(\Vert \nabla_{\hat{x}}D(\hat{x}) \Vert_{2} - 1)^{2}\Big]$$

# Training
| Resolution | Elapsed time spent training 1,000 steps | Steps | Total elapsed time | Accumulated elapsed time |
| - | - | - | - | - |
| 4×4 | 0:00:48 | 50 |  |  |
| 4×4 to 8×8 | 0:00:52 | 〃 |  |  |
| 8×8 | 〃 | 〃 | 〃 |  |
| 8×8 to 16×16 | 0:00:58 | 〃 |  |  |
| 16×16 | 〃 | 〃 | 〃 |  |
| 16×16 to 32×32 | 0:03:36 | 〃 |  |  |
| 32×32 | 〃 | 〃 | 〃 |  |
| 32×32 to 64×64 | 0:06:58 | 〃 |  |  |
| 64×64 | 〃 | 〃 | 〃 |  |
| 64×64 to 128×128 | 0:12:15 | 〃 |  |  |
| 128×128 | 0:12:15 | 〃 |
| 256×256 |  |
| 512×512 |  |
| 1,024×1,024 |  |

# Research
## 23.08.03
- 이전까지는 Celeb-A HQ의 Training set에 대해 Mean과 Std를 계산해서 이걸 가지고 Normalize했습니다.
    - `T.Normalize(mean=(0.517, 0.416, 0.363), std=(0.303, 0.275, 0.269))`
- 논문에서는 "We represent training and generated images in $[-1, 1]$."라는 표현이 있는 것을 알고 있었으나 이를 간과하고 있었습니다.
- 다음과 같이 코드를 수정했습니다.
    - `T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`
