# Source drawings

The organic edit reads the user's existing Generative_Art vectors from the
checkout supplied with `--art-root`. The artwork stays in its existing repository.
These are the botanical/decayed-barn materials used for the earlier kaleidoscope
direction. No copies of the source artwork are bundled in HydraPy.

Pinned source revision: `181c161072f7737e1a61e6797a7e89d13cc6c386`.

- `studies/wisconsin-flora/output/leadplant.svg`: [Wisconsin flora / leadplant](https://github.com/JosiahAlanLee/Generative_Art/blob/181c161072f7737e1a61e6797a7e89d13cc6c386/studies/wisconsin-flora/output/leadplant.svg), Git blob `06edd97041a2fdb50b9270247bb636f7aa22388b`.
- `studies/farm-remnants/output/barn-22-d72.svg`: [Farm remnants / barn seed 22, decay .72](https://github.com/JosiahAlanLee/Generative_Art/blob/181c161072f7737e1a61e6797a7e89d13cc6c386/studies/farm-remnants/output/barn-22-d72.svg), Git blob `89dd85f7ca3613d3da5a9abae7bf178f0faf8f47`.

The renderer caches white-background rasterizations using Inkscape or CairoSVG,
then samples inverted luminance as a linework matte. Opposing rotations, moving
crop coordinates and independently crossfading mirror counts supply motion.
The vectors themselves are not modified or traced from a flattened preview.
The render sidecar records the SHA-256 of each input drawing.
