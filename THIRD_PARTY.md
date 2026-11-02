# Third-party sources and model files

This integration retains the licenses of its upstream components. No single
permissive license is granted for the entire combined model or source tree.
The following locations contain the applicable original texts and notices.

| Component | Included license / attribution |
|---|---|
| LIDF / Implicit Depth Completion | `third_party/lidf/LICENSE`, `third_party/STAGE2_NOTICE.md`, `src/v2_lidf/vendor/lidf/NOTICE.md` |
| SPVNAS | `third_party/spvnas/LICENSE`, `third_party/STAGE2_NOTICE.md` |
| TorchSparse | `third_party/torchsparse/LICENSE`, `third_party/STAGE2_NOTICE.md` |
| sparsehash | `third_party/sparsehash/LICENSE`, `third_party/STAGE2_NOTICE.md` |
| MoGe, including its DINOv2 components | `third_party/moge/LICENSE` and individual file headers |
| DINOv3 | `third_party/dinov3/LICENSE.md` |
| TransNormal | `third_party/transnormal/LICENSE`, `third_party/transnormal/LICENSE-APACHE-2.0.txt` and individual file headers |
| TOM-Net | `third_party/stage1_support/licenses/TOM-Net-LICENSE` |
| ClearGrasp preprocessing helper | `third_party/cleargrasp/LICENSE`, `third_party/cleargrasp/NOTICE.md` |

LIDF retains NVIDIA's noncommercial research/evaluation terms. TransNormal's
main license is CC BY-NC 4.0; the separate Apache text applies to the components
identified by its source notices. DINOv3 retains its custom license. Model
weights retain their upstream terms even when fetched separately.

Official model files are fetched from the pinned revisions in
`provenance/release_assets.json`; they are not rehosted in ordinary Git history.
The project release includes the project-trained assets and calibration file
listed in that manifest. Dataset files are not distributed.

`provenance/stage1_source_origins.json` records local source origins and hashes;
`provenance/stage2_source_origin.json` records the trained Stage2 source origins.
Historical paths in these provenance records are evidence, not runtime inputs.
