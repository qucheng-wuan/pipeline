# Model assets

Large model files are excluded from Git. The release asset manifest records
their relative paths, byte sizes, SHA-256 checksums, and original sources.
User-trained weights are distributed through the private repository release.
Official third-party weights retain their upstream licenses and pinned sources.

Inference uses RGB and calibrated camera intrinsics. No sensor depth or GT mask
is supplied at deployment.
