"""
Tile-scoring models behind one interface. Consumes images, emits raw scores.

``scorer.TileScorer`` is the interface tagging depends on. ``patchcore``
implements it with anomalib's PatchCore; ``tiling`` decodes frames and cuts
them into tiles; ``device`` picks the torch device. torch and anomalib are
imported lazily inside functions so ``import tagurit`` stays cheap and works
without the ``model`` extra.
"""
