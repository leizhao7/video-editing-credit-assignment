# Third-party notices

The root [MIT License](LICENSE) applies to project-authored code. It does not replace the licenses or copyright notices of third-party material.

| Component | Included material | License |
| :--- | :--- | :--- |
| OpenCode | Upstream notices, package metadata, and lockfile under [`training/third_party/OpenCode/`](training/third_party/OpenCode/). | [MIT](training/third_party/OpenCode/LICENSE), Copyright (c) 2025 opencode. |
| SkyRL | Upstream license, changes under [`training/patches/skyrl/`](training/patches/skyrl/), and SkyRL source-replacement fragments in the [OPSD installer](opsd/examples/train/answer_self_distillation/install.py). | [Apache-2.0](training/third_party/SkyRL/LICENSE). |

Preserve these upstream licenses and any accompanying notices when redistributing their material. Recorded dependency identities are listed in [`configs/dependency_pins.json`](configs/dependency_pins.json). External software dependencies and model weights remain subject to their own licenses; their inclusion as dependencies does not place them under this project's MIT license.

## Paper figure and media

[`docs/assets/paper-teaser.png`](docs/assets/paper-teaser.png) reproduces a paper figure containing third-party film thumbnails. The paper figure is excluded from the code license. No rights to the underlying films, trailers, audio, or other external media are granted by this repository's MIT license. The dataset and original media are not distributed.
