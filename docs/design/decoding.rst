Decoding
========

Neither head knows anything about feet. The placement head ranks slots and the selection head ranks
patterns, and left alone they will pin a foot for two bars, retap a panel in 89 milliseconds, or
spend a whole chart on the two middle panels. Everything here is a rule measured off the corpus and
applied at decode time, separately from the model that proposes the notes.

Choosing the slots
------------------

How many slots to keep is decided separately from which ones, because the placement head ranks well
but is not calibrated. The count comes from the note rate the rating implies.

Which ones is three passes:

#. **Silence.** Slots whose loudest mixture band sits below −70 dB are put out of reach outright.
   Silence cannot be read off the placement scores, because those are a model output: over dead air
   the network still returns a number, and a whole song of them can be flat enough that trailing
   silence never looks unusual.
#. **A seed per measure.** The strongest beat of each measure that plays at all is taken first.
   Ranking a subdivision family across the whole song otherwise lets a loud passage outbid a quiet
   one: a song whose second half calms down keeps its peaks but loses on the average, so the
   quarter budget drains into the louder half. Which measures play is decided locally; how busy
   each one gets is still decided globally.
#. **Coarse before fine.** Quarters are laid down across the whole song before any eighth, and
   eighths before any sixteenth, in the proportions the corpus uses at that note rate. Ranking every
   subdivision together lets sixteenths win wherever the audio is loud, flooding those bars and
   emptying the rest.

Finally the rests are tidied. Taking the highest scoring slots leaves a hole wherever the score
dips, and a hole has no reason to line up with anything, whereas charts rest for a phrase: over 5893
corpus rests, 28.7 per cent begin on a downbeat and 29.3 per cent end on a bar line, against 9.5 per
cent each for holes left by score alone. Emptying the measures that were barely used and giving
their notes to measures already playing turns ragged holes into whole-measure rests without changing
the note count.

The rulebook
------------

Every candidate pattern passes a mask before it can be sampled. The rules, and what each is
measured against:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Rule
     - Basis
   * - Hold consistency
     - A head with no tail makes its panel read as occupied for the rest of the song and stops the
       file loading at all.
   * - Limb count
     - Two for ``feet``, four for ``hands``, unlimited for ``keyboard``.
   * - Jack limit
     - 150 ms. Over 1931 feet-style charts and 550 thousand same-panel intervals, the first
       percentile is 150 ms and only 0.4 per cent fall under 130.
   * - Fast jumps
     - 130 ms, from 44 thousand corpus jumps, of which 3.5 per cent follow a shorter gap.
   * - Freeze ration
     - 4 per cent of notes. Real charts sit at 3.9 per cent at the median; unrationed the model
       reaches for them about eight times that often.
   * - Jump ration
     - 8 per cent of rows for a pad, 14 for a keyboard. The corpus jump rate is flat from rating
       five through seventeen.
   * - Crossovers
     - See below.
   * - Panel balance
     - See below.

When every rule bites at once something still has to be placed, so they are surrendered in order of
how much they matter: crossover balance first, then the freeze and mine rations, then the rule
against fast jumps. Hold consistency and the jack limit are **never** surrendered, because a chart
that breaks the first will not load and one that breaks the second cannot be danced.

Crossovers
----------

Inside a run the limbs alternate, so the panel sequence decides whether a step lands crossed: left,
down, right, up puts the left foot on the right panel, and its reverse — up, left, down, right —
puts the right foot on the left one.

Two things are rationed, and both are tighter for a sixteenth stream: the share of the chart that
crosses at all, and how many crossed steps may follow one another. A share alone bounds the total
while letting them arrive in a clump, and the clump is what hurts: across 7285 crossed stretches in
500 corpus charts, 83 per cent are a single step and 97 per cent are one or two.

The default share is 4 per cent, half the corpus tenth percentile, because the generator streams
almost entirely in sixteenths where crossovers are worst. Sixteenths get a sixth of that again and
may never carry two in a row. ``--crossovers 0`` bars them outright at any speed.

Panel balance
-------------

Real charts spread their notes evenly over the four panels: across the corpus the median shares are
24.6, 26.5, 24.5 and 24.5 per cent. The model does not — left to itself it produces 20.6, 32.5,
28.7 and 18.1, favouring the two middle panels in every chart measured.

Mirroring the training data **cannot** reach this. The four reflections of a pad form the Klein
four-group, and every one of them maps the outer pair onto the outer pair and the middle pair onto
the middle pair, so no amount of averaging over them moves weight between those pairs. Reaching it
would need a quarter turn, and a quarter turn is not a symmetry of play: left and right are
naturally one foot each while up and down are shared.

So it is corrected at decode time, by nudging each pattern by how over-used its panels are so far.
The record decays with a half-life of about sixteen notes, so a single measure is still free to sit
on one arrow while the song-level totals stay even.
