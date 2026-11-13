# hops the dongle across the FM band, one channel at a time,
# and turns what it hears on each channel into a small picture for the model

import ctypes
import os
import time

import numpy as np

import radio

# We listen WIDE (2.3 MHz) and keep only the middle 256 kHz in software.
# Listening narrow (250 kHz) made strong stations show up again as ghosts
# exactly 1.0 MHz away from where they really are. At 2.304 MHz the nearest
# fold-back point is 2.304 MHz away, which falls between two channels.
SAMPLE_RATE = 2304000
WIDE_FFT = 1152           # 2304000 / 1152 = 2 kHz per bin
NARROW_RATE = 256000      # what is left after keeping the middle 128 bins
GAIN = 19.7               # same gain rf-decoder uses
DWELL_SECONDS = 0.2       # how long we listen on each channel

FIRST_MHZ = 87.9
STEP_MHZ = 0.2
NUM_CHANNELS = 101        # 87.9, 88.1, ... 107.9

FFT_SIZE = 128            # picture is 128 frequency bins wide
TIME_ROWS = 16            # and 16 time slices tall
PICTURE_RANGE_DB = 40     # anything more than 40 dB over the background is drawn the same

SMALL_BLOCK = 65536       # bytes per read while hopping (about 14 ms)
SETTLE_BLOCKS = 2         # throw these away after a retune so the tuner can settle

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SWEEP_FOLDER = os.path.join(DATA_FOLDER, "sweeps")


def channel_list():
    return [round(FIRST_MHZ + STEP_MHZ * k, 1) for k in range(NUM_CHANNELS)]


def freq_key(freq_mhz):
    # the text used as a key in the json files, like "88.3"
    return f"{freq_mhz:.1f}"


def open_dongle():
    return radio.open_radio(round(FIRST_MHZ * 1e6), SAMPLE_RATE, GAIN)


def tune(lib, dev, freq_mhz):
    if lib.rtlsdr_set_center_freq(dev, round(freq_mhz * 1e6)) != 0:
        raise RuntimeError("couldn't tune to " + str(freq_mhz))
    lib.rtlsdr_set_tuner_gain(dev, round(GAIN * 10))


def read_fast(lib, dev, num_samples):
    # like radio.read_raw but with small blocks so a hop is quick
    lib.rtlsdr_reset_buffer(dev)
    buf = (ctypes.c_uint8 * SMALL_BLOCK)()
    n_read = ctypes.c_int()
    chunks = []
    total = 0
    skipped = 0
    while total < num_samples * 2:
        if lib.rtlsdr_read_sync(dev, buf, SMALL_BLOCK, ctypes.byref(n_read)) != 0:
            raise RuntimeError("reading from the dongle failed (unplugged?)")
        if skipped < SETTLE_BLOCKS:
            skipped += 1
            continue
        chunks.append(bytes(buf)[:n_read.value])
        total += n_read.value
    return b"".join(chunks)[:num_samples * 2]


def hop(lib, dev, freq_mhz):
    # tune to one channel and return what we heard as complex samples
    tune(lib, dev, freq_mhz)
    raw = read_fast(lib, dev, int(DWELL_SECONDS * SAMPLE_RATE))
    return radio.bytes_to_iq(raw)


def power_frames(iq):
    # chop the samples into short pieces and get the spectrum of each one,
    # then keep only the middle FFT_SIZE bins (the channel we are tuned to).
    # result has one row per piece and FFT_SIZE columns, low frequency on the left
    count = len(iq) // WIDE_FFT
    frames = iq[:count * WIDE_FFT].reshape(count, WIDE_FFT) * np.hanning(WIDE_FFT)
    spectrum = np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)
    middle = WIDE_FFT // 2
    spectrum = spectrum[:, middle - FFT_SIZE // 2:middle + FFT_SIZE // 2]
    return np.abs(spectrum) ** 2


def to_narrow(iq):
    # throw away everything except the middle 256 kHz, so the result is the
    # tuned channel on its own at NARROW_RATE samples per second.
    # used for the long listens where we decode the station text
    keep = len(iq) * NARROW_RATE // SAMPLE_RATE
    spectrum = np.fft.fftshift(np.fft.fft(iq))
    middle = len(iq) // 2
    spectrum = spectrum[middle - keep // 2:middle + keep // 2]
    return np.fft.ifft(np.fft.ifftshift(spectrum)).astype(np.complex64)


def make_picture(iq):
    # the picture the neural net looks at.
    # brightness = how many dB that spot is above this channel's own quiet level,
    # so the picture shows the SHAPE of the signal and not how loud the channel is
    power = power_frames(iq)
    rows_per_slice = len(power) // TIME_ROWS
    power = power[:rows_per_slice * TIME_ROWS].reshape(TIME_ROWS, rows_per_slice, FFT_SIZE).mean(axis=1)
    db = 10 * np.log10(power + 1e-12)
    db = db - np.percentile(db, 10)
    return (np.clip(db, 0, PICTURE_RANGE_DB) / PICTURE_RANGE_DB).astype(np.float32)


def channel_power_db(iq):
    # average power inside the 150 kHz a station is allowed to use
    power = power_frames(iq).mean(axis=0)
    freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, 1 / NARROW_RATE))
    inside = np.abs(freqs) < 75000
    return float(10 * np.log10(power[inside].mean() + 1e-12))


def above_floor(powers_db):
    # the empty-channel level is not the same across the band (it climbs about
    # 10 dB from 88 to 108), so compare each channel to the quiet ones near it.
    # floor = the 20th percentile of the channels within 2 MHz
    powers = np.asarray(powers_db, dtype=float)
    result = np.zeros(len(powers))
    for i in range(len(powers)):
        nearby = powers[max(0, i - 10):i + 11]
        nearby = nearby[~np.isnan(nearby)]
        if np.isnan(powers[i]) or len(nearby) < 5:
            result[i] = np.nan
        else:
            result[i] = powers[i] - np.percentile(nearby, 20)
    return result


def run_sweep(lib, dev, on_channel=None, should_stop=None):
    # one pass over all 101 channels
    freqs = channel_list()
    pictures = np.zeros((NUM_CHANNELS, TIME_ROWS, FFT_SIZE), dtype=np.float32)
    powers = np.full(NUM_CHANNELS, np.nan)
    for i, freq in enumerate(freqs):
        if should_stop is not None and should_stop():
            return None
        iq = hop(lib, dev, freq)
        pictures[i] = make_picture(iq)
        powers[i] = channel_power_db(iq)
        if on_channel is not None:
            on_channel(i, freq, pictures[i], powers[i])
    return {"pictures": pictures, "power_db": powers, "above_floor": above_floor(powers)}


def save_sweep(sweep):
    os.makedirs(SWEEP_FOLDER, exist_ok=True)
    name = os.path.join(SWEEP_FOLDER, "sweep_" + time.strftime("%Y%m%d_%H%M%S") + ".npz")
    np.savez_compressed(name, pictures=sweep["pictures"], power_db=sweep["power_db"],
                        above_floor=sweep["above_floor"])
    return name


def load_sweeps():
    if not os.path.isdir(SWEEP_FOLDER):
        return []
    sweeps = []
    for name in sorted(os.listdir(SWEEP_FOLDER)):
        if name.endswith(".npz"):
            data = np.load(os.path.join(SWEEP_FOLDER, name))
            sweeps.append({"name": name, "pictures": data["pictures"],
                           "power_db": data["power_db"], "above_floor": data["above_floor"]})
    return sweeps


if __name__ == "__main__":
    # python sweep.py 10   -> record 10 sweeps for training
    import sys
    how_many = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    lib, dev = open_dongle()
    try:
        for n in range(how_many):
            start = time.time()
            sweep = run_sweep(lib, dev)
            print("sweep", n + 1, "of", how_many, "took", round(time.time() - start, 1),
                  "s, saved", os.path.basename(save_sweep(sweep)))
    finally:
        radio.close_radio(lib, dev)
