# Paper Reading
- [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/pdf/1710.10196.pdf)
$$\mathcal{L}_{GP} = \lambda \mathbb{E}_{\hat{x} \sim \mathbb{P}_{\hat{x}}}\Big[(\Vert \nabla_{\hat{x}}D(\hat{x}) \Vert_{2} - 1)^{2}\Big]$$

# Training
| Resolution | Elapsed time spent training 1,000 steps | Steps | Total elapsed time | Accumulated elapsed time |
| - | - | - | - | - |
| 4×4 | 0:01:58 | 50 | 1:38:20 | 01:38:20 |
| 4×4 to 8×8 | 0:02:10 | 〃 | 1:48:20 | 03:26:40 |
| 8×8 | 〃 | 〃 | 〃 | 05:15:00 |
| 8×8 to 16×16 | 0:02:39 | 〃 | 2:12:30 | 07:27:30 |
| 16×16 | 〃 | 〃 | 〃 | 09:40:00 |
| 16×16 to 32×32 | 0:05:28 | 〃 | 4:33:20 | 14:13:20 |
| 32×32 | 〃 | 〃 | 〃 | 18:46:40 |
| 32×32 to 64×64 | 0:09:02 | 〃 | 7:31:20 | 26:18:00 |
| 64×64 | 〃 | 〃 | 〃 | 33:49:20 |
| 128×128 |  |
| 256×256 |  |
| 512×512 |  |
| 1,024×1,024 |  |
