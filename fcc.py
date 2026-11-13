# downloads the official FCC list of FM stations near the UCF campus area
# run it once:  python fcc.py

import json
import os
import urllib.request

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FCC_FILE = os.path.join(DATA_FOLDER, "fcc.json")

# campus area, not an address. 28 36 09 N, 81 12 00 W
SEARCH_KM = 100
URL = (
    "https://transition.fcc.gov/fcc-bin/fmq?state=&call=&city=&arn=&serv=&vac="
    "&freq=87.9&fre2=107.9&facid=&asrn=&class=&list=4"
    "&dist=" + str(SEARCH_KM) +
    "&dlat2=28&mlat2=36&slat2=9&NS=N&dlon2=81&mlon2=12&slon2=0&EW=W&size=9"
)

# FM = full power, FL = low power, FX = translator, FB = booster
# FS (backup transmitters) is skipped because the main one is already listed
REAL_SERVICES = ["FM", "FL", "FX", "FB"]


def parse_line(line):
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 30:
        return None
    service = parts[3]
    status = parts[9]
    # LIC means licensed and on the air. CP is only a construction permit
    if service not in REAL_SERVICES or status != "LIC":
        return None
    try:
        freq = round(float(parts[2].replace("MHz", "")), 1)
        km = float(parts[28].replace("km", ""))
    except ValueError:
        return None
    try:
        erp_kw = float(parts[14].replace("kW", "").strip())
    except ValueError:
        erp_kw = 0.0    # the list shows "-" for some stations
    return {"freq": freq, "call": parts[1], "city": parts[10].title(), "state": parts[11],
            "service": service, "erp_kw": erp_kw, "km": km}


def download():
    with urllib.request.urlopen(URL, timeout=60) as response:
        text = response.read().decode("latin-1")
    stations = []
    seen = set()
    for line in text.splitlines():
        station = parse_line(line)
        if station is None:
            continue
        key = (station["freq"], station["call"])
        if key in seen:
            continue
        seen.add(key)
        stations.append(station)
    stations.sort(key=lambda s: (s["freq"], s["km"]))
    return stations


def load():
    if not os.path.exists(FCC_FILE):
        return []
    with open(FCC_FILE) as f:
        return json.load(f)


def by_freq(stations):
    # frequency text like "88.3" -> list of stations, nearest first
    table = {}
    for station in stations:
        table.setdefault(f"{station['freq']:.1f}", []).append(station)
    return table


if __name__ == "__main__":
    os.makedirs(DATA_FOLDER, exist_ok=True)
    found = download()
    with open(FCC_FILE, "w") as f:
        json.dump(found, f, indent=1)
    print(len(found), "licensed stations within", SEARCH_KM, "km on",
          len(by_freq(found)), "different frequencies")
