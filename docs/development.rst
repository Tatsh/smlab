Development
===========

.. code-block:: shell

   uv sync --group dev --extra stems

.. code-block:: shell

   uv run ruff format . && uv run ruff check . && uv run mypy smlab && uv run pytest

Retraining
----------

Only needed to change the models. Separating the corpus into stems dominates the wall clock;
training the chart model itself takes about 85 minutes on a consumer GPU, and the offset model
about 70 seconds.

.. code-block:: shell

   smlab scan ~/.project-outfox/Songs -w 16    # index the corpus and its timing
   smlab stems                                 # separate and build the training features
   smlab vocab -c cache/stems                  # collect the note-row patterns
   smlab train                                 # the chart model

The offset model reads its own, much smaller cache of onset envelopes, and needs no separation:

.. code-block:: shell

   smlab envelopes && smlab train-offset

Pass ``-c checkpoints`` to ``generate`` to use locally trained models instead of downloaded ones,
and ``smlab publish`` to upload them.

Accuracy, measured on held-out songs
------------------------------------

None of these numbers come from the training split. The offset figures cover the songs whose tempo
was recovered correctly, since a wrong tempo makes the offset meaningless.

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Task
     - Result
     - Previous heuristic
   * - Tempo within 0.5 BPM
     - 87.7%
     - 78.3%
   * - Offset within 30 ms
     - 58.1%
     - 46.2%
   * - Offset, median error
     - 20.5 ms
     - 45.5 ms
   * - Offset a clean half-beat out
     - 2.2%
     - 12.9%
   * - **Tempo and offset both right**
     - **50.9% of all songs**
     - —

The median offset error of 20.5 ms is one phase bin, so it is limited by the model's resolution
rather than its judgement.
