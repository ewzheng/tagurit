"""
Tile-scoring models behind one interface. Consumes images, emits raw scores.

``scorer.TileScorer`` is the interface tagging depends on, and
``scorer.WindowedTileScorer`` its form for scorers that remember recent
frames. ``patchcore`` implements it with anomalib's PatchCore, and ``clip``
with query-only CLIP plus ``novelty``'s window of recent tiles; ``tiling``
decodes frames and cuts them into tiles; ``device`` picks the torch device. torch and anomalib are
imported lazily inside functions so ``import tagurit`` stays cheap and works
without the ``model`` extra.
"""
