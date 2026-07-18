# third_party/

**CropFormer** (the RGB frame segmenter) is not vendored here — it depends on
detectron2 + Mask2Former and has its own license. Clone it into this folder:

```bash
git clone https://github.com/qqlu/Entity.git third_party/CropFormer
```

Then follow its build instructions (detectron2 + Mask2Former + compiled ops) and
download the `CropFormer_swin_tiny_3x.pth` checkpoint into
`models/rgb_models/reference/` (see the main [README](../README.md#models)).

`paths.py` looks for the repo at `third_party/CropFormer` by default.

> SAM2 and HQ-SAM are installed as pip packages (from their source repos), not
> vendored here — see the main README.
