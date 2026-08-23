smlab
=====

.. only:: html

   .. include:: badges.rst

   Generate StepMania ``dance-single`` charts from audio, using models trained on a corpus of
   human-authored simfiles.

   This is not a random step generator. The audio is separated into drums, bass, other and vocals,
   and the chart follows what is actually playing: one model decides where the steps go and a
   second decides which panels they use, both reading the same encoding of the music. Everything
   either model cannot be trusted with — how a chart is spelled, and what two feet can physically
   do — is applied afterwards, as rules measured off the corpus.

.. toctree::
   :maxdepth: 2

   cli

.. only:: html

   .. toctree::
      :caption: User guide
      :maxdepth: 2

      installation
      usage

   .. toctree::
      :maxdepth: 2

      design/index

   .. toctree::
      :maxdepth: 2

      api/index

   .. toctree::
      :caption: Reference
      :maxdepth: 2

      development

   Indices and tables
   ------------------

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`
