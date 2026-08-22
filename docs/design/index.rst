How it works
============

Generating a chart is four separate problems, and they are solved separately because they fail
separately. Recovering the beat grid is a signal-processing problem. Deciding *where* steps go and
*which panels* they use is what the models do. Deciding what a chart may physically ask of a player
is neither, and is applied afterwards as rules measured off the corpus.

.. toctree::
   :maxdepth: 2

   timing
   features
   model
   decoding
   formats
   training
