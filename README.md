# smlab

<!-- WISWA-GENERATED-README:START -->

[![Python versions](https://img.shields.io/pypi/pyversions/smlab.svg?color=blue&logo=python&logoColor=white)](https://www.python.org/)
[![PyPI - Version](https://img.shields.io/pypi/v/smlab)](https://pypi.org/project/smlab/)
[![GitHub tag (with filter)](https://img.shields.io/github/v/tag/Tatsh/smlab)](https://github.com/Tatsh/smlab/tags)
[![License](https://img.shields.io/github/license/Tatsh/smlab)](https://github.com/Tatsh/smlab/blob/master/LICENSE.txt)
[![GitHub commits since latest release (by SemVer including pre-releases)](https://img.shields.io/github/commits-since/Tatsh/smlab/v0.0.0/master)](https://github.com/Tatsh/smlab/compare/v0.0.0...master)
[![CodeQL](https://github.com/Tatsh/smlab/actions/workflows/codeql.yml/badge.svg)](https://github.com/Tatsh/smlab/actions/workflows/codeql.yml)
[![QA](https://github.com/Tatsh/smlab/actions/workflows/qa.yml/badge.svg)](https://github.com/Tatsh/smlab/actions/workflows/qa.yml)
[![Tests](https://github.com/Tatsh/smlab/actions/workflows/tests.yml/badge.svg)](https://github.com/Tatsh/smlab/actions/workflows/tests.yml)
[![Coverage Status](https://coveralls.io/repos/github/Tatsh/smlab/badge.svg?branch=master)](https://coveralls.io/github/Tatsh/smlab?branch=master)
[![Dependabot](https://img.shields.io/badge/Dependabot-enabled-blue?logo=dependabot)](https://github.com/dependabot)
[![Documentation Status](https://readthedocs.org/projects/smlab/badge/?version=latest)](https://smlab.readthedocs.org/?badge=latest)
[![mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![uv](https://img.shields.io/badge/uv-261230?logo=astral)](https://docs.astral.sh/uv/)
[![pytest](https://img.shields.io/badge/pytest-zz?logo=Pytest&labelColor=black&color=black)](https://docs.pytest.org/en/stable/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://static.pepy.tech/badge/smlab/month)](https://pepy.tech/project/smlab)
[![Stargazers](https://img.shields.io/github/stars/Tatsh/smlab?logo=github&style=flat)](https://github.com/Tatsh/smlab/stargazers)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Tatsh/smlab/master.svg)](https://results.pre-commit.ci/latest/github/Tatsh/smlab/master)
[![Prettier](https://img.shields.io/badge/Prettier-black?logo=prettier)](https://prettier.io/)

[![@Tatsh](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpublic.api.bsky.app%2Fxrpc%2Fapp.bsky.actor.getProfile%2F%3Factor=did%3Aplc%3Auq42idtvuccnmtl57nsucz72&query=%24.followersCount&label=Follow+%40Tatsh&logo=bluesky&style=social)](https://bsky.app/profile/Tatsh.bsky.social)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-Tatsh-black?logo=buymeacoffee)](https://buymeacoffee.com/Tatsh)
[![Libera.Chat](https://img.shields.io/badge/Libera.Chat-Tatsh-black?logo=liberadotchat)](irc://irc.libera.chat/Tatsh)
[![Mastodon Follow](https://img.shields.io/mastodon/follow/109370961877277568?domain=hostux.social&style=social)](https://hostux.social/@Tatsh)
[![Patreon](https://img.shields.io/badge/Patreon-Tatsh2-F96854?logo=patreon)](https://www.patreon.com/Tatsh2)

<!-- WISWA-GENERATED-README:STOP -->

Generate StepMania `dance-single` (4-panel DDR) charts from an audio file, using models trained on a
corpus of human-authored simfiles.

This is not a random step generator. The audio is separated into drums, bass, other and vocals, and
the chart follows what is actually playing: a model decides where the steps go and a second decides
which panels they use, both reading the same encoding of the music.

```shell
smlab generate song.ogg -o /path/to/Pack -T "Song Title" -A "Artist"
```

That writes `/path/to/Pack/Song Title/` containing `Song Title.ssc` and a copy of the audio as
`Song Title.ogg`. `--format sm` and `--format dwi` write the older formats instead.

Only `.ssc` carries everything the generator works out. `.sm` has no per-chart tags, so the groove
radar and the chart hash go unwritten, and `.dwi` additionally cannot indicate a mine, a lift, or a
roll: mines and lifts are dropped and a roll becomes an ordinary freeze.

**The chart and offset weights are downloaded on first use**. Set `SMLAB_WEIGHTS_REPO` to fetch them
from somewhere other than the default, or pass `-c` to point at a local `checkpoints/` directory.
Everything that does not need a model works without them: parsing, timing estimation, playability
analysis, chart drawing.

## Automatically generated fields

| Field                         | How it is chosen                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `#BPMS`                       | Folded onset envelope over a 25 s excerpt. Override with `--bpm`, or `--bpm-multiplier 2` when detection lands an octave low.             |
| `#OFFSET`                     | A model that reads four frequency bands folded onto the bar and picks which of 96 positions holds the downbeat. Override with `--offset`. |
| `#TITLE`, `#ARTIST`, `#GENRE` | Read from the audio tags when not given. MP3, Ogg, FLAC, M4A, and Opus.                                                                   |
| `#SAMPLESTART`                | Predicted by a model that scores every measure and picks one.                                                                             |
| `#SAMPLELENGTH`               | Fixed at 15 s, which 83% of the corpus uses.                                                                                              |
| `#CREDIT`                     | The current username.                                                                                                                     |
| Steps                         | Placement and selection models, decoded under physical constraints.                                                                       |

## Accuracy

None of these numbers come from the training split. The offset figures cover the songs whose tempo
was recovered correctly.

| Task                            | Result                 | Previous heuristic |
| ------------------------------- | ---------------------- | ------------------ |
| Tempo within 0.5 BPM            | 87.7%                  | 78.3%              |
| Offset within 30 ms             | 58.1%                  | 46.2%              |
| Offset, median error            | 20.5 ms                | 45.5 ms            |
| Offset a clean half-beat out    | 2.2%                   | 12.9%              |
| **Tempo and offset both right** | **50.9% of all songs** | —                  |

The median offset error of 20.5 ms is one phase bin, so it is limited by the model's resolution
rather than its judgement.

**Check the detected timing before trusting a chart.** Roughly one song in eight gets the wrong
tempo, and for those the grid is wrong however good the offset is. The `confidence` figure printed
alongside the detection is a usable signal: values near 1.0 mean the winning tempo barely beat the
runner-up.

Two knobs exist for the common failures:

- `--bpm-multiplier 2` when the tempo comes out halved, `0.5` when doubled.
- `--latency 0.060` when every chart is consistently early or late on your setup, since playback
  latency is a property of the machine rather than the song. `SMLAB_LATENCY` sets it once.

## Difficulty and rating scales

A rating means nothing without knowing its scale, so `--scale` picks one: `10` for classic DDR,
`15` for In The Groove, `20` for X-era and later (the default). The same number differs sharply
across them — a nine is 4.33 notes per second in ITG, 4.10 on the classic scale, and 3.12 on the
modern one.

```shell
smlab generate song.ogg -D Hard -D "Challenge:16" --scale 20
```

Each difficulty takes an optional rating after a colon; `-m` sets one for any that do not. Because
the classic scale saturates — corpus charts labelled ten run anywhere from 3.5 to 6.9 notes per
second — `--nps` bypasses the rating and states the note rate outright.

## Keyboard charts versus pad charts

`--style` decides what the generator is allowed to write, and the same analysis is available for
existing files through `smlab analyze`.

- `feet` — danceable with two feet. Nothing needs more than two panels at once, no panel repeats
  faster than a foot can retap, and the note rate stays within human stamina.
- `hands` — additionally allows the three and four panel chords common in In The Groove.
- `keyboard` — no physical limits, and a separate density scale, because a keyboard chart at a given
  rating runs roughly twice as dense as a pad chart carrying the same number.

The classifier decides by three separate measures: whether a two-foot assignment exists at all
(a dynamic program over foot positions), whether any row needs more than two panels, and whether the
note rate over a one-second and a ten-second window exceeds what a dancer sustains. Validated
against the corpus, it rates the keyboard megapacks 26.2% keyboard-only against 1.5% for arcade
rips, and finds hand-chords in 10.9% of In The Groove charts against 0.0% for arcade.

## Chart image

```shell
smlab generate song.ogg --image      # while generating
smlab image "Song/Song.sm"           # from an existing simfile
```

Pictures go in `.images/` inside the song folder, one per chart. PNG by default, `--svg` for vector.

## Timing conventions

Read out of the StepMania source rather than assumed, because a sign error is silent: the chart
still loads and plays, merely off-beat forever.

| Tag                      | Meaning                                                     |
| ------------------------ | ----------------------------------------------------------- |
| `#OFFSET` (`.sm`/`.ssc`) | Beat 0 occurs at `-OFFSET` seconds into the audio.          |
| `#GAP` (`.dwi`)          | Whole milliseconds until beat 0, so `OFFSET = -GAP / 1000`. |

Tempo changes are not detected, but they can be stated. `smlab drift song.mp3` measures how far the
tempo wanders across a song and prints where a marker belongs; `--warp SECONDS:BPM` on `generate`
places one, repeatably. That is the Ableton warp workflow: the tool shows the drift, you place the
markers. Freezes are not inserted.

## Retraining

Only needed to change the models. Separating the corpus into stems is by far the slow part; training
the chart model itself takes about 85 minutes on a consumer GPU, and the offset model about 70
seconds.

Separation is an optional dependency, because it pulls in a second copy of torch's ecosystem:

```shell
pip install 'smlab[stems]'
```

```shell
smlab scan ~/.project-outfox/Songs -w 16    # index the corpus and its timing
smlab stems                                 # separate and build the training features
smlab vocab -c cache/stems                  # collect the note-row patterns
smlab train                                 # the chart model
```

The offset model reads its own smaller cache of onset envelopes and needs no separation:

```shell
smlab envelopes && smlab train-offset
```

Pass `-c checkpoints` to `generate` to use locally trained models instead of downloaded ones, and
`smlab publish` to upload them.

## Development

```shell
uv sync --all-groups --all-extras
```

After making changes:

```shell
uv run ruff format . && uv run ruff check . && uv run mypy smlab && uv run pytest
```

torch is taken from PyPI, whose Linux wheels already carry CUDA, so a machine with an NVIDIA driver
uses the GPU without any extra configuration. To build against something else — ROCm, a different
CUDA release, or CPU only - install that variant over the top:

```shell
uv pip install --torch-backend=rocm6.4 torch
```
