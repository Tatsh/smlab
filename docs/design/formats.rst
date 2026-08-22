Simfile formats
===============

Three formats can be written, and they are not equivalent.

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Format
     - Notes
   * - ``.ssc``
     - The default. Per-chart tags, so the groove radar and the chart hash are written.
   * - ``.sm``
     - No per-chart tags. Otherwise the same note data.
   * - ``.dwi``
     - Cannot spell a mine, a lift, or a roll. Mines and lifts are dropped and a roll becomes an
       ordinary freeze.

Note data
---------

Measures are emitted at the coarsest subdivision that still represents every row they contain,
which is what hand-written simfiles do and what makes note colours read correctly in game.

DWI is different in kind. A character advances the beat rather than naming a row: a bare character
is an eighth of a measure, and bracket groups switch to a finer step, so a measure carries only as
many characters as its own resolution needs. Panels are packed into a single character, one per
pair, and a chord of three or more needs an angle-bracket group.

A DWI freeze is written where it starts as ``step!panel`` and has **no tail of its own**. The next
step on that panel ends it and is swallowed by the reader. Verified by rendering 1286 corpus charts
and reading them back: 1284 return exactly what went in, eight of the remaining ten differ only by
their mines and lifts, and two carry 64th notes, which the note grid cannot hold in the first
place.

The groove radar
----------------

``#RADARVALUES`` is computed rather than zeroed, following ``NoteDataUtil::CalculateRadarValues``:
five rates and nine counts, written twice because the tag carries one set per player.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Value
     - Definition
   * - Stream
     - ``notes / seconds / 7``
   * - Voltage
     - ``(peak notes in any 8 beats / 8) * (last beat / seconds) / 10``
   * - Air
     - ``jumps / seconds``
   * - Freeze
     - ``holds / seconds``
   * - Chaos
     - ``rows finer than an eighth / seconds * 0.5``

``seconds`` is the **decoded** length of the audio, which is what StepMania divides by. The length a
container advertises is no substitute: for one MP3 checked here it was a quarter of a second out.
Counting hands additionally needs to know how long each freeze runs, so tails are paired to their
heads first — assuming a fixed length reported fourteen hands in a chart that has none.

The chart hash
--------------

``#CHARTHASH`` is a Project OutFox addition. It is the MD5, as lower-case hexadecimal, of the
``#NOTES`` value **exactly as the MSD parser hands it over**: comments removed but their lines left
behind empty, carriage returns gone because the file is read as text, and nothing at all trimmed —
the newline after the tag's own colon is part of the digest.

Two consequences follow. The hash is a property of the file's formatting rather than of the steps,
and it is computed over the text as loaded rather than over anything re-rendered.

Older OutFox builds wrote a decimal integer in this tag instead, of which ``2147483647`` is the
no-value sentinel, so a non-hex value in the wild is not a corrupt digest.
