Installation
============

.. code-block:: shell

   pip install smlab

The chart and offset weights are **not** bundled: together they run to over 150 MB. They are
downloaded on first use and cached. Set ``SMLAB_WEIGHTS_REPO`` to fetch them from somewhere other
than the default, ``SMLAB_WEIGHTS_REVISION`` to pin a revision, or pass ``-c`` to point at a local
``checkpoints/`` directory.

Everything that needs no model works without them: parsing, timing estimation, playability
analysis, and chart drawing.

torch comes from PyPI, whose Linux wheels already carry CUDA, so a machine with an NVIDIA driver
uses the GPU without any further configuration. To build against something else — ROCm, a different
CUDA release, or CPU only — install that variant over the top:

.. code-block:: shell

   uv pip install --torch-backend=rocm6.4 torch

Retraining
----------

Separating the corpus into stems is by far the slow part, and it pulls in a second copy of torch's
ecosystem, so it is an optional dependency:

.. code-block:: shell

   pip install 'smlab[stems]'
