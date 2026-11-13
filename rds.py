# decodes rds data from an fm station

from datetime import datetime, timedelta

import numpy as np

PILOT = 19000
RDS_CARRIER = 57000
RDS_WIDTH = 2400
CHECK_POLY = 0b10110111001
OFFSETS = {"A": 0x0FC, "B": 0x198, "C": 0x168, "C'": 0x350, "D": 0x1B4}

PROGRAM_TYPES = [
    "None", "News", "Information", "Sports", "Talk", "Rock", "Classic Rock", "Adult Hits",
    "Soft Rock", "Top 40", "Country", "Oldies", "Soft", "Nostalgia", "Jazz", "Classical",
    "Rhythm and Blues", "Soft R&B", "Language", "Religious Music", "Religious Talk",
    "Personality", "Public", "College", "Spanish Talk", "Spanish Music", "Hip Hop",
    "Unassigned", "Unassigned", "Weather", "Emergency Test", "Emergency",
]


def bandpass(signal, sample_rate, low_hz, high_hz):
    # float64 or the pilot drifts
    # pad to a power of 2 so the fft is fast
    size = 1
    while size < len(signal):
        size = size * 2
    spectrum = np.fft.fft(np.asarray(signal, dtype=np.float64), size)
    freqs = np.fft.fftfreq(size, 1 / sample_rate)
    spectrum[(freqs < low_hz) | (freqs > high_hz)] = 0
    return 2 * np.fft.ifft(spectrum)[:len(signal)]


def get_bits(baseband, sample_rate):
    pilot = bandpass(baseband, sample_rate, PILOT - 15, PILOT + 15)
    nearby = bandpass(baseband, sample_rate, PILOT + 200, PILOT + 230)
    if np.median(np.abs(pilot)) < 4 * np.median(np.abs(nearby)):
        raise ValueError("no 19 kHz pilot tone, this only works on stereo stations")
    pilot = pilot / np.abs(pilot)

    # the rds carrier is the pilot times 3
    rds = bandpass(baseband, sample_rate, RDS_CARRIER - RDS_WIDTH, RDS_CARRIER + RDS_WIDTH)
    rds = rds * np.conj(pilot ** 3)
    angle = np.angle(np.sum(rds ** 2)) / 2
    rds = np.real(rds * np.exp(-1j * angle))

    # the bit clock is the pilot divided by 16
    cycles = np.unwrap(np.angle(pilot)) / (2 * np.pi)
    a = len(cycles) // 10
    b = len(cycles) * 9 // 10
    cycles_per_sample = (cycles[b] - cycles[a]) / (b - a)
    samples_per_bit = 16 / cycles_per_sample

    # try 32 starting points and keep the strongest
    running_sum = np.concatenate(([0.0], np.cumsum(rds)))
    best_score = -1
    best_values = None
    for offset in np.linspace(0, samples_per_bit, 32, endpoint=False):
        starts = np.arange(offset, len(rds) - samples_per_bit, samples_per_bit)
        start = starts.astype(int)
        middle = (starts + samples_per_bit / 2).astype(int)
        end = (starts + samples_per_bit).astype(int)
        first_half = running_sum[middle] - running_sum[start]
        second_half = running_sum[end] - running_sum[middle]
        values = first_half - second_half
        score = np.mean(np.abs(values))
        if score > best_score:
            best_score = score
            best_values = values

    # a 1 means the sign changed
    sent = best_values > 0
    bits = []
    for i in range(1, len(sent)):
        if sent[i] != sent[i - 1]:
            bits.append(1)
        else:
            bits.append(0)
    return bits


def calc_check(data):
    reg = data << 10
    for i in range(25, 9, -1):
        if (reg >> i) & 1:
            reg = reg ^ (CHECK_POLY << (i - 10))
    return reg


def block_ok(word, offset):
    data = word >> 10
    check = word & 0x3FF
    return (calc_check(data) ^ offset) == check


def read_block(word, names):
    for name in names:
        if block_ok(word, OFFSETS[name]):
            return word >> 10

    # try fixing 1 bad bit or 2 next to each other
    for name in names:
        for i in range(26):
            if block_ok(word ^ (1 << i), OFFSETS[name]):
                return (word ^ (1 << i)) >> 10
        for i in range(25):
            if block_ok(word ^ (3 << i), OFFSETS[name]):
                return (word ^ (3 << i)) >> 10
    return None


def get_word(bits, i):
    word = 0
    for bit in bits[i:i + 26]:
        word = (word << 1) | bit
    return word


def get_groups(bits):
    # find a good A block followed by a good B block
    start = None
    for i in range(len(bits) - 52):
        if block_ok(get_word(bits, i), OFFSETS["A"]) and block_ok(get_word(bits, i + 26), OFFSETS["B"]):
            start = i
            break
    if start is None:
        return [], 0, 0

    groups = []
    good = 0
    total = 0
    i = start
    while i + 104 <= len(bits):
        group = []
        for j, names in enumerate([["A"], ["B"], ["C", "C'"], ["D"]]):
            data = read_block(get_word(bits, i + 26 * j), names)
            total += 1
            if data is not None:
                good += 1
            group.append(data)
        groups.append(group)
        i += 104
    return groups, good, total


def to_char(code):
    if 32 <= code < 127:
        return chr(code)
    return "?"


def call_letters(pi):
    # us call letters counted in base 26
    if pi >> 12 == 0xA:
        if (pi >> 8) & 0xF == 0xF:
            pi = (pi & 0xFF) << 8
        else:
            pi = (((pi >> 8) & 0xF) << 12) | (pi & 0xFF)
    if 0x1000 <= pi <= 0x54A7:
        first = "K"
        n = pi - 0x1000
    elif 0x54A8 <= pi <= 0x994F:
        first = "W"
        n = pi - 0x54A8
    else:
        return None
    return first + chr(65 + n // 676) + chr(65 + (n // 26) % 26) + chr(65 + n % 26)


def program_type(pty):
    if pty is None:
        return "unknown"
    return PROGRAM_TYPES[pty]


def decode_clock(b, c, d):
    # date is days since nov 17 1858
    mjd = ((b & 3) << 15) | (c >> 1)
    hour = ((c & 1) << 4) | (d >> 12)
    minute = (d >> 6) & 63
    if hour > 23 or minute > 59 or mjd < 50000:
        return None
    offset = (d & 31) / 2
    if (d >> 5) & 1:
        offset = -offset
    utc = datetime(1858, 11, 17) + timedelta(days=mjd, hours=hour, minutes=minute)
    return utc + timedelta(hours=offset), offset


def most_common(counts):
    if not counts:
        return None
    return max(counts, key=counts.get)


def decode_groups(groups):
    station = {"pi": None, "pty": None, "names": [], "texts": [], "clock": None}
    pi_counts = {}
    pty_counts = {}
    name = ["?"] * 8
    name_parts = set()
    text = [None] * 64
    text_flag = None

    for a, b, c, d in groups:
        if a is not None:
            pi_counts[a] = pi_counts.get(a, 0) + 1
        if b is None:
            continue
        group_type = b >> 12
        version_b = (b >> 11) & 1
        pty = (b >> 5) & 31
        pty_counts[pty] = pty_counts.get(pty, 0) + 1

        if group_type == 0 and d is not None:
            # station name
            part = b & 3
            name[part * 2] = to_char(d >> 8)
            name[part * 2 + 1] = to_char(d & 255)
            name_parts.add(part)
            if len(name_parts) == 4:
                station["names"].append("".join(name))
                name_parts = set()

        elif group_type == 2:
            # radio text
            flag = (b >> 4) & 1
            if flag != text_flag:
                text = [None] * 64
                text_flag = flag
            part = b & 15
            if version_b == 0 and c is not None and d is not None:
                text[part * 4:part * 4 + 4] = [c >> 8, c & 255, d >> 8, d & 255]
            elif version_b == 1 and d is not None:
                text[part * 2:part * 2 + 2] = [d >> 8, d & 255]
            end = 64 if version_b == 0 else 32
            if 13 in text:
                end = text.index(13)
            if None not in text[:end]:
                full_text = "".join(to_char(x) for x in text[:end]).strip()
                if full_text:
                    station["texts"].append(full_text)
                text = [None] * 64

        elif group_type == 4 and version_b == 0 and c is not None and d is not None:
            # clock
            clock = decode_clock(b, c, d)
            if clock is not None:
                station["clock"] = clock

    station["pi"] = most_common(pi_counts)
    station["pty"] = most_common(pty_counts)
    return station


def decode_station(baseband, sample_rate):
    bits = get_bits(baseband, sample_rate)
    groups, good, total = get_groups(bits)
    station = decode_groups(groups)
    station["good_blocks"] = good
    station["total_blocks"] = total
    return station


def newest_song(texts):
    # only count a text if it showed up twice
    for text in reversed(texts):
        if texts.count(text) >= 2:
            return text
    return None
