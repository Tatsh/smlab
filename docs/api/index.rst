API reference
=============

``smlab`` re-exports the parsing, timing, playability and writing API at the package root, so
``from smlab import load_simfile, write_song`` works, and importing it costs nothing: none of those
pull in torch. Generation lives in :py:mod:`smlab.chart.gen`, which does.

Each name is documented once, under the module that defines it, rather than again under every
package that re-exports it.

.. toctree::
   :maxdepth: 2

   audio
   chart
   constraints
   corpus
   models
   timing
   train
   writer
   misc
