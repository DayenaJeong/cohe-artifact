# Third-party attribution and redistribution scope

This is an attribution record, not a legal opinion. Users must check the
current upstream terms before obtaining or using any upstream model or dataset.

| Upstream asset | Official source | Use in this addendum | Release scope |
|---|---|---|---|
| CIFAR-100 | [CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) | training-order alignment | no raw dataset |
| DDPM CIFAR-10 | [google/ddpm-cifar10-32 model card](https://huggingface.co/google/ddpm-cifar10-32) | scalar proxy provenance | derived scalar values only |
| SD-VAE | [stabilityai/sd-vae-ft-mse model card](https://huggingface.co/stabilityai/sd-vae-ft-mse) | scalar proxy provenance | derived scalar values only |
| Stable Diffusion v1.5 | [stable-diffusion-v1-5 model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) | scalar proxy provenance | derived scalar values only |
| Emu3 Vision Tokenizer | [BAAI/Emu3-VisionTokenizer model card](https://huggingface.co/BAAI/Emu3-VisionTokenizer) | scalar proxy provenance | derived scalar values only |
| DINOv2 | [DINOv2 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | isolation-score provenance | derived scalar values only |
| ConvAE | locally trained research model | scalar proxy provenance | checkpoint excluded; derived scalar values only |

The DDPM card identifies Apache-2.0, the SD-VAE card identifies MIT, the
SD-v1.5 card identifies CreativeML OpenRAIL-M, the Emu3 tokenizer card
identifies Apache-2.0 with custom-code requirements, and DINOv2 publishes an
Apache-2.0 license. These notices do not grant rights to redistribute upstream
weights or data. ImageNet-derived arrays are intentionally excluded because
their scalar-only redistribution status remains unresolved for this release.
