# Stage2 attribution and distribution

This directory retains complete upstream license texts. Code under LIDF's
license is not covered by the permissive licenses of other dependencies.

| Component | Pinned source | License |
|---|---|---|
| LIDF / Implicit Depth Completion | NVlabs/implicit_depth, 4dc85bbddfc96af60006d6006be0c74478fe9204 | NVIDIA Source Code License: noncommercial research/evaluation restriction; retain the full license and attribution |
| SPVNAS | mit-han-lab/spvnas, 5c53224ca685844eb27caedf22817a89233ac0ce | MIT |
| TorchSparse | mit-han-lab/torchsparse, 385f5ce8718fcae93540511b7f5832f4e71fd835 | MIT |
| sparsehash | sparsehash/sparsehash, f93c0c69e959c1c77611d0ba8d107aa971338811 | BSD3clause |

The LIDF-SPVNAS integration references qucheng-wuan/LIDF-SPVNAS commit
`a0b2771385bc802e2fbd52da2c4407ae86ac7e70`. Model numerical files are preserved
from the documented trained experiment; provenance/stage2_source_origin.json
records their exact original hashes. CUDA kernel source is included and compiled
locally. TorchSparse/sparsehash are installed from pinned upstream source.

The tiny SPVNAS net.config is redistributed with upstream attribution because
specialized topology construction requires it. Official pretrained SPVNAS20G
and ResNet34 ImageNet files are downloaded on explicit request and verified by
SHA256; they are not required for a complete trained inference bundle. Their
original distribution URLs are recorded in scripts/fetch_stage2_initialization.py.
The downloader does not relicense upstream weights or data.

Historical vendor notices are retained for provenance. Some describe earlier
intermediate experiments (e.g. no ImageNet or20/20/20) and are not the protocol
of this release. The actual final Stage2 uses verified ImageNet ResNet34,
pretrained SPVNAS20G,80epochs and20/30/30 curriculum. The authoritative portable
protocol is configs/stage2_train.json and docs/STAGE2_RUNTIME.md.
