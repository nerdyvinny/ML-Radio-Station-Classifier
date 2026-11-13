# ML Radio Station Classifier

I wanted to know which FM channels around me actually have a station on them,
without sitting there turning a dial. This project uses an RTL-SDR dongle to
hop across the FM band (87.9 to 107.9 MHz, 101 channels) and a small
convolutional net to sort each channel into **empty**, **weak station** or
**strong station**. A local dashboard shows the whole band as one strip, with a
table of frequency, signal strength, model confidence and station name.

## Setup

You need an RTL-SDR dongle and Python 3.

```
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe server.py
```

Open http://127.0.0.1:8750 and press **Start sweep**. A full sweep takes about
33 seconds. The trained model (`data/model.pt`) is included, so the dashboard
works right away.

## How it works

1. **Sweep** (`sweep.py`). Tune to a channel, listen for 0.2 s, move on. Each
   listen is turned into a small picture: 128 frequency bins across, 16 time
   slices down, brightness = dB above that channel's own noise floor. The
   picture captures the *shape* of the signal rather than how loud it is.
2. **Labels** (`labels.py`). The real answer comes from listening:
   - **strong** = clean, listenable audio
   - **weak** = something is there but it's hissy or keeps dropping out
   - **empty** = static

   To save time, a slow pass (10 s per channel) makes a first guess: station
   text decodes -> strong, stereo pilot only -> weak, neither -> empty. Every
   "station" guess gets flagged for a listen in the dashboard, and so does every
   "empty" channel where the FCC list shows a big station (10 kW+) within 75 km.
3. **Model** (`model.py`). A 3-layer CNN looks at the picture and outputs a
   probability for each class. It's always scored against a plain
   "strength above X dB" rule, so it's obvious whether the net is earning its keep.
4. **Dashboard** (`server.py`, `dashboard.html`). Live strip and table, a detail
   panel showing the exact picture the net sees, a 5 s audio clip, buttons to
   overrule a label, and a Retrain button.

Station names come from the FCC list for each frequency (`fcc.py`, 100 km
around Orlando). A name is shown as heard only if the station's RDS text
actually decoded; otherwise it's marked "listed, not heard".

## Other commands

```
.\.venv\Scripts\python.exe fcc.py             # refresh the FCC list
.\.venv\Scripts\python.exe sweep.py 12        # record 12 sweeps for training
.\.venv\Scripts\python.exe labels.py listen   # slow 10 s per channel pass (~25 min)
.\.venv\Scripts\python.exe train.py           # train and print the score
```

The raw sweeps aren't in the repo (about 59 MB), so record your own with
`sweep.py` before retraining. My hand-checked labels are in `data/labels.json`.

## Results

13 sweeps, 1,313 pictures, every channel rated by ear: 84 empty, 5 weak,
12 strong. Scored with 5-fold cross-validation, always on channels the net
never trained on (`data/report.json`).

| | CNN | Strength rule |
|---|---|---|
| Balanced accuracy | **79.1%** | 65.0% |
| Empty pictures called a station | 73 of 1,092 | 443 of 1,092 |
| Strong pictures found | 131 of 156 | 130 of 156 |
| Weak pictures found | 39 of 65 | 34 of 65 |

Almost all of the net's advantage comes from channels right next to a loud
station. Those read 6 to 12 dB above the floor, so the strength rule calls them
weak stations. The net notices the energy is sitting at one edge of the picture
instead of the middle and correctly calls them empty.

## Limitations

- **Weak is barely learnable.** There are only 5 weak channels. The net finds
  39 of 65 weak pictures, and its false alarms are mostly 94.3, 96.9, 99.1 and
  107.5, which are really empty: faint bumps 1 to 3 dB above the floor.
- **One place, one evening.** Every picture came from the same antenna spot in
  about an hour. I haven't tested another day or location yet.
- **Dashboard confidence is optimistic.** The model on screen was trained on all
  101 channels, including the ones it's showing you.
- **The slow listening pass is noisy.** The same station gave 54 good RDS blocks
  on one listen and 437 on the next. Decoding in four separate chunks fixed the
  labels, but I never pinned down whether the cause was dropped samples or fading.

## Two things that tripped me up

- **Ghost stations.** Sampling narrow (250 kHz) made every powerful station
  show up again exactly 1.0 MHz away: 100.1, 102.1, 106.1, 91.3 and 91.7 all
  looked like strong stations and weren't. Each hop now samples 2.3 MHz wide
  and keeps the middle in software.
- **The FCC list isn't ground truth.** It has a licensed station on 76 of the
  101 frequencies within 100 km, mostly tiny translators you can't hear here.
