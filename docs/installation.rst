Installation
============

.. code-block:: shell

   pip install smlab

The chart and offset weights are **not** bundled: together they run to over 150 MB. They are
looked for on the machine first and downloaded only if they are not there, so a system package can
install them and nothing is ever fetched.

Everything that needs no model works without them: parsing, timing estimation, playability
analysis, and chart drawing.

Where the weights are looked for
--------------------------------

``smlab weights`` prints the list for the machine it runs on. In order:

#. The directory passed to ``-c``.
#. ``$SMLAB_WEIGHTS_DIR``.
#. ``checkpoints/`` under the working directory, which is where training writes.
#. ``~/.local/share/smlab/``, which is also where a download is kept.
#. ``share/smlab/`` under the installation prefix, for a virtual environment.
#. ``/usr/local/share/smlab/`` and ``/usr/share/smlab/``, following ``XDG_DATA_DIRS``.

A file found earlier wins, so locally trained weights are never fought over by a download, and a
user's own copy takes precedence over one installed for the whole machine.

Packaging the weights
---------------------

A distribution that wants ``smlab`` to work offline installs ``chart.pt`` and ``offset.pt`` into
``/usr/share/smlab/``. Every release carries them as assets, so the tag to fetch is the version
being packaged, and their digests are bundled in the wheel as ``smlab/assets/weights.sha256``, which
:command:`sha256sum -c` reads directly.

.. code-block:: shell

   https://github.com/Tatsh/smlab/releases/download/v0.0.1/chart.pt
   https://github.com/Tatsh/smlab/releases/download/v0.0.1/offset.pt

Downloading is checked against those digests, and a download that does not match is discarded
rather than loaded.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Variable
     - Meaning
   * - ``SMLAB_WEIGHTS_DIR``
     - Directory searched before any of the standard ones.
   * - ``SMLAB_WEIGHTS_REPO``
     - ``owner/name`` of the GitHub repository to download from.
   * - ``SMLAB_WEIGHTS_RELEASE``
     - Release tag to download from, which pins the weights.
   * - ``SMLAB_WEIGHTS_URL``
     - Base URL of a mirror, which the file name is appended to.

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
