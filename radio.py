# talks to the rtl sdr dongle

import ctypes
import os

import numpy as np

DLL_FOLDER = os.environ.get("RTLSDR_DIR", os.path.join(os.path.expanduser("~"), "tools", "rtl-sdr", "x64"))
BLOCK_SIZE = 16 * 16384


def load_library():
    os.add_dll_directory(DLL_FOLDER)
    return ctypes.CDLL(os.path.join(DLL_FOLDER, "rtlsdr.dll"))


def open_radio(freq_hz, sample_rate, gain_db):
    lib = load_library()
    if lib.rtlsdr_get_device_count() == 0:
        raise RuntimeError("no RTL-SDR dongle found, is it plugged in?")

    dev = ctypes.c_void_p()
    if lib.rtlsdr_open(ctypes.byref(dev), 0) != 0:
        raise RuntimeError("couldn't open the dongle (close SDR# or anything else using it)")

    if lib.rtlsdr_set_sample_rate(dev, sample_rate) != 0:
        raise RuntimeError("couldn't set the sample rate")
    if lib.rtlsdr_set_center_freq(dev, freq_hz) != 0:
        raise RuntimeError("couldn't tune to " + str(freq_hz) + " Hz")
    if lib.rtlsdr_get_direct_sampling(dev) != 0:
        raise RuntimeError("tuner is still bypassed (direct sampling) after tuning")

    # gain has to be set after tuning because tuning resets the tuner
    if hasattr(lib, "rtlsdr_set_bias_tee"):
        lib.rtlsdr_set_bias_tee(dev, 0)
    lib.rtlsdr_set_tuner_gain_mode(dev, 1)
    lib.rtlsdr_set_tuner_gain(dev, round(gain_db * 10))
    return lib, dev


def read_raw(lib, dev, num_samples):
    # clear old data and skip the first block
    lib.rtlsdr_reset_buffer(dev)
    buf = (ctypes.c_uint8 * BLOCK_SIZE)()
    n_read = ctypes.c_int()
    chunks = []
    total = 0
    first = True
    while total < num_samples * 2:
        if lib.rtlsdr_read_sync(dev, buf, BLOCK_SIZE, ctypes.byref(n_read)) != 0:
            raise RuntimeError("reading from the dongle failed (unplugged?)")
        if first:
            first = False
            continue
        chunks.append(bytes(buf)[:n_read.value])
        total += n_read.value
    return b"".join(chunks)[:num_samples * 2]


def close_radio(lib, dev):
    lib.rtlsdr_close(dev)


def bytes_to_iq(raw):
    # turn the raw bytes into complex numbers
    data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    data = (data - 127.5) / 127.5
    return data[0::2] + 1j * data[1::2]
