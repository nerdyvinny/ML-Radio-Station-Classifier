# works out the right answer for each channel: strong, weak or empty
#
#   python labels.py listen   -> sit on each channel for 10 s and collect evidence (about 25 min)
#   python labels.py          -> rebuild labels.json from the evidence already collected
#
# The rule (the user's choice, 2026-11-12): YOUR EAR decides.
#   strong = a station you can listen to with clean audio
#   weak   = a station is there but it is hissy or breaks up
#   empty  = only static
# The computer can't hear, so what it measures is only a first GUESS:
#   station text (RDS) decodes, id read 5+ times -> guess strong
#   19 kHz stereo pilot tone but no text          -> guess weak
#   neither                                        -> guess empty
# Every guess that says "a station is here" is flagged for you to confirm, and
# so is every "empty" channel where the FCC list has a big station nearby
# (it might be a mono or hissy station the pilot test can't see).
# A flag does not change the label or keep the channel out of training, only
# your click does. The FCC list never decides a label: it lists something on
# 76 of the 101 frequencies, so "listed" does not mean "receivable here".

import json
import os
import sys
import wave

import numpy as np

import fcc
import fm
import radio
import rds
import sweep

LISTEN_SECONDS = 10
AUDIO_SECONDS = 5
PIECES = 4                # the listen is decoded in 4 separate pieces
MIN_ID_COUNT = 5          # "decoded" = the same station id read at least this many times
REVIEW_DB = 6             # an unexplained signal this far above the floor gets reviewed
BIG_STATION_KW = 10       # an "empty" channel with a full-power station this big
BIG_STATION_KM = 75       # and this close on the FCC list gets a listen too

CLASSES = ["empty", "weak", "strong"]

EVIDENCE_FILE = os.path.join(sweep.DATA_FOLDER, "evidence.json")
LABELS_FILE = os.path.join(sweep.DATA_FOLDER, "labels.json")
AUDIO_FOLDER = os.path.join(sweep.DATA_FOLDER, "audio")


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


def save_audio(filename, audio, rate):
    peak = np.max(np.abs(audio))
    if peak == 0:
        peak = 1
    samples = (audio / peak * 0.9 * 32767).astype(np.int16)
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(samples.tobytes())


def best_name(station):
    # the 8 letter name the station sends, if we caught a clean copy of it
    clean = [n.strip() for n in station["names"] if "?" not in n and n.strip()]
    if not clean:
        return None
    return max(set(clean), key=clean.count)


def decode_in_pieces(baseband):
    # the decoder in rds.py locks on once and never recovers, so one hiccup in
    # the radio stream ruins everything after it. the same loud station gave
    # 54 good blocks on one listen and 437 on the next. so we cut the listen
    # into pieces, decode each one on its own, and count how many times the
    # station id came through in total
    found = {"pilot": False, "decoded": False, "pi": None, "pi_count": 0, "call": None,
             "name": None, "good_blocks": 0, "total_blocks": 0}
    piece_length = len(baseband) // PIECES
    id_counts = {}
    names = []
    for n in range(PIECES):
        piece = baseband[n * piece_length:(n + 1) * piece_length]
        try:
            bits = rds.get_bits(piece, sweep.NARROW_RATE)
        except ValueError:
            continue    # rds.py raises this when there is no pilot tone
        found["pilot"] = True
        groups, good, total = rds.get_groups(bits)
        found["good_blocks"] += good
        found["total_blocks"] += total
        for group in groups:
            if group[0] is not None:
                id_counts[group[0]] = id_counts.get(group[0], 0) + 1
        names += rds.decode_groups(groups)["names"]
    if id_counts:
        best = max(id_counts, key=id_counts.get)
        found["pi_count"] = id_counts[best]
        if id_counts[best] >= MIN_ID_COUNT:
            found["decoded"] = True
            found["pi"] = f"{best:04X}"
            found["call"] = rds.call_letters(best)
            found["name"] = best_name({"names": names})
    return found


def listen_to_channel(lib, dev, freq_mhz):
    # one long listen. returns a dict of what we could prove about this channel
    sweep.tune(lib, dev, freq_mhz)
    raw = radio.read_raw(lib, dev, LISTEN_SECONDS * sweep.SAMPLE_RATE)
    narrow = sweep.to_narrow(radio.bytes_to_iq(raw))
    baseband = fm.demodulate(narrow, sweep.NARROW_RATE)

    found = decode_in_pieces(baseband)

    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    step = sweep.NARROW_RATE // fm.AUDIO_RATE
    audio = fm.get_audio(baseband[:AUDIO_SECONDS * sweep.NARROW_RATE], sweep.NARROW_RATE)
    save_audio(os.path.join(AUDIO_FOLDER, sweep.freq_key(freq_mhz) + ".wav"), audio,
               sweep.NARROW_RATE // step)
    return found


def listen_to_all(only=None):
    evidence = load_json(EVIDENCE_FILE)
    lib, dev = sweep.open_dongle()
    try:
        for freq in sweep.channel_list():
            if only and sweep.freq_key(freq) not in only:
                continue
            found = listen_to_channel(lib, dev, freq)
            evidence[sweep.freq_key(freq)] = found
            save_json(EVIDENCE_FILE, evidence)
            print(sweep.freq_key(freq), "pilot" if found["pilot"] else "-----",
                  "text" if found["decoded"] else "----",
                  "id read", found["pi_count"], "times,",
                  found["good_blocks"], "/", found["total_blocks"], "blocks",
                  found["call"] or "", found["name"] or "", flush=True)
    finally:
        radio.close_radio(lib, dev)


def average_above_floor():
    # how far above the local floor each channel sits, averaged over the saved sweeps
    sweeps = sweep.load_sweeps()
    if not sweeps:
        return [float("nan")] * sweep.NUM_CHANNELS
    return list(np.nanmean([s["above_floor"] for s in sweeps], axis=0))


def louder_neighbour_has_same_station(i, keys, evidence, strength):
    # if the channel next door decoded the same station id and is louder,
    # then what we decoded here is that station spilling over, not a second one
    mine = evidence.get(keys[i], {})
    for j in (i - 1, i + 1):
        if 0 <= j < len(keys):
            other = evidence.get(keys[j], {})
            if other.get("decoded") and other.get("pi") == mine.get("pi") and strength[j] > strength[i]:
                return keys[j]
    return None


def strong_neighbour(i, keys, evidence):
    for j in (i - 1, i + 1):
        if 0 <= j < len(keys) and evidence.get(keys[j], {}).get("decoded"):
            return keys[j]
    return None


def big_listed_station(stations):
    # the nearest full-power station on the FCC list that could plausibly be heard
    for station in stations:
        if station["service"] == "FM" and station["erp_kw"] >= BIG_STATION_KW and station["km"] <= BIG_STATION_KM:
            return f"{station['call']} ({station['erp_kw']:g} kW, {station['km']:.0f} km)"
    return None


def build_labels():
    listed = fcc.by_freq(fcc.load())
    evidence = load_json(EVIDENCE_FILE)
    old = load_json(LABELS_FILE)
    strength = average_above_floor()
    keys = [sweep.freq_key(f) for f in sweep.channel_list()]
    labels = {}
    for i, key in enumerate(keys):
        if old.get(key, {}).get("source") == "you":
            labels[key] = old[key]      # never overwrite an answer the user gave
            continue
        found = evidence.get(key)
        if found is None:
            continue
        loud = strength[i] >= REVIEW_DB
        entry = {"label": "empty", "source": "auto", "needs_review": False, "why": ""}
        if found["decoded"]:
            spill_from = louder_neighbour_has_same_station(i, keys, evidence, strength)
            if spill_from is None:
                entry["label"] = "strong"
                entry["needs_review"] = True
                entry["why"] = "guess: station text decoded. Does it sound clean?"
            else:
                entry["needs_review"] = True
                entry["why"] = "decoded the same station as " + spill_from + ", probably spill-over"
        elif found["pilot"]:
            entry["label"] = "weak"
            entry["needs_review"] = True
            entry["why"] = "guess: stereo pilot tone but no text. Hissy or clean?"
        elif loud:
            # no pilot and no text means empty under the rule, however loud.
            # next to a strong station it is that station spilling over.
            # anywhere else it is unexplained, so it is flagged for a listen
            spill_from = strong_neighbour(i, keys, evidence)
            if spill_from is not None:
                entry["why"] = "no pilot, no text: spill-over from " + spill_from
            else:
                entry["needs_review"] = True
                entry["why"] = "signal with no pilot and no text and no strong station next door"
        else:
            entry["why"] = "no pilot, no text, nothing above the floor"
        if entry["label"] == "empty" and not entry["needs_review"]:
            big = big_listed_station(listed.get(key, []))
            if big is not None:
                entry["needs_review"] = True
                entry["why"] += ". But " + big + " is listed nearby, any station in the clip?"
        labels[key] = entry
    save_json(LABELS_FILE, labels)
    return labels


def station_names():
    # name to show for each channel: what we decoded off the air, or else the
    # nearest station on the FCC list (marked as coming from the list)
    evidence = load_json(EVIDENCE_FILE)
    listed = fcc.by_freq(fcc.load())
    names = {}
    for freq in sweep.channel_list():
        key = sweep.freq_key(freq)
        found = evidence.get(key, {})
        if found.get("decoded"):
            # call letters worked out from the station id are only trusted when
            # the FCC list agrees. lots of stations send an id that does not
            # follow the call-letter scheme (104.1 came out as "KCZY", it is WTKS)
            # the 8 letter text most stations send is a scrolling song title or
            # slogan ("MORGAN", "Nirvana"), not a name, so the call sign goes first
            listed_calls = [s["call"] for s in listed.get(key, [])]
            agreed = [c for c in listed_calls if c.split("-")[0] == found.get("call")]
            if agreed:
                name = agreed[0]
            elif listed_calls:
                name = listed_calls[0]
            else:
                name = "id " + found["pi"]
            if found.get("name"):
                name += ', sending "' + found["name"] + '"'
            names[key] = {"name": name, "source": "air"}
        elif key in listed:
            nearest = listed[key][0]
            names[key] = {"name": nearest["call"] + " " + nearest["city"], "source": "list",
                          "km": nearest["km"]}
        else:
            names[key] = {"name": "", "source": "none"}
    return names


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "listen":
        # python labels.py listen            -> all 101 channels
        # python labels.py listen 92.3 96.5  -> just these
        listen_to_all(sys.argv[2:])
    result = build_labels()
    for name in CLASSES:
        print(name, sum(1 for e in result.values() if e["label"] == name))
    print("need review", sum(1 for e in result.values() if e["needs_review"]))
