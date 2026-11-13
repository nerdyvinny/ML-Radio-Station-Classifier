# the neural net, the simple strength rule it is compared against,
# and the test that tells us whether either of them actually works

import json
import os

import numpy as np
import torch
import torch.nn as nn

import labels
import sweep

CLASSES = labels.CLASSES          # ["empty", "weak", "strong"]
MODEL_FILE = os.path.join(sweep.DATA_FOLDER, "model.pt")
REPORT_FILE = os.path.join(sweep.DATA_FOLDER, "report.json")

EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.002
FOLDS = 5


def build_net():
    # input is one picture: 1 x 16 time rows x 128 frequency bins.
    # three small layers of filters, each followed by shrinking the picture.
    # the last pooling keeps 8 positions left-to-right, because WHERE the energy
    # sits matters: a station is in the middle, spill-over is at one edge
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1, 2)),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 8)),
        nn.Flatten(),
        nn.Linear(32 * 8, len(CLASSES)),
    )


def load_dataset():
    # every saved sweep gives one picture per channel. the label belongs to the
    # channel, so every picture of 104.1 gets 104.1's label.
    # only channels with no label at all are left out
    answers = labels.load_json(labels.LABELS_FILE)
    keys = [sweep.freq_key(f) for f in sweep.channel_list()]
    pictures, targets, channels, strengths = [], [], [], []
    skipped = []
    for i, key in enumerate(keys):
        entry = answers.get(key)
        if entry is None:
            skipped.append(key)
    for one in sweep.load_sweeps():
        for i, key in enumerate(keys):
            if key in skipped or np.isnan(one["above_floor"][i]):
                continue
            pictures.append(one["pictures"][i])
            targets.append(CLASSES.index(answers[key]["label"]))
            channels.append(i)
            strengths.append(one["above_floor"][i])
    return {"pictures": np.array(pictures, dtype=np.float32), "targets": np.array(targets),
            "channels": np.array(channels), "strengths": np.array(strengths), "skipped": skipped}


def train_net(pictures, targets, seed=0):
    torch.manual_seed(seed)
    net = build_net()
    x = torch.tensor(pictures).unsqueeze(1)
    y = torch.tensor(targets, dtype=torch.long)

    # most channels are empty. without weights the net could score 80% by
    # always saying "empty", so mistakes on rare classes cost more
    counts = np.bincount(targets, minlength=len(CLASSES)).astype(float)
    weights = torch.tensor(len(targets) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)
    loss_function = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    history = []
    for epoch in range(EPOCHS):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            xb = x[batch]
            # mirror half the pictures left-to-right so the net can't learn
            # "spill-over only ever comes from the right"
            flip = torch.rand(len(batch)) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            optimizer.zero_grad()
            loss = loss_function(net(xb), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        history.append(total / len(x))
    net.eval()
    return net, history


def predict(net, pictures):
    # returns one row per picture: chance of empty, weak, strong (adds up to 1)
    with torch.no_grad():
        x = torch.tensor(np.asarray(pictures, dtype=np.float32)).unsqueeze(1)
        return torch.softmax(net(x), dim=1).numpy()


def balanced_accuracy(targets, guesses):
    # average of "how many of the real empties did we get", same for weak and
    # strong. plain accuracy would hide failures on the rare classes
    scores = []
    for c in range(len(CLASSES)):
        mine = targets == c
        if mine.sum() > 0:
            scores.append(float((guesses[mine] == c).mean()))
    return float(np.mean(scores))


def apply_strength_rule(strengths, cuts):
    # the dumb method: below the first cut = empty, above the second = strong
    return (strengths >= cuts[0]).astype(int) + (strengths >= cuts[1]).astype(int)


def fit_strength_rule(strengths, targets):
    # try every pair of cut-offs in half-dB steps and keep the best pair
    best_score, best_cuts = -1, (3.0, 12.0)
    steps = np.arange(0, 30.5, 0.5)
    for low in steps:
        for high in steps[steps > low]:
            score = balanced_accuracy(targets, apply_strength_rule(strengths, (low, high)))
            if score > best_score:
                best_score, best_cuts = score, (float(low), float(high))
    return best_cuts


def assign_folds(channels, targets, seed=0):
    # split CHANNELS (not pictures) into 5 groups, spreading each class evenly.
    # a channel's pictures all look alike, so if the same channel were in both
    # training and test the net could just recognise it instead of learning
    rng = np.random.default_rng(seed)
    fold_of_channel = {}
    for c in range(len(CLASSES)):
        mine = np.unique(channels[targets == c])
        rng.shuffle(mine)
        for n, channel in enumerate(mine):
            fold_of_channel[int(channel)] = n % FOLDS
    return np.array([fold_of_channel[int(ch)] for ch in channels])


def confusion(targets, guesses):
    table = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for t, g in zip(targets, guesses):
        table[t, g] += 1
    return table.tolist()


def cross_validate(data):
    # train on 4 groups of channels, test on the 5th, five times over.
    # every picture ends up being judged by a net that never saw its channel
    folds = assign_folds(data["channels"], data["targets"])
    net_guess = np.zeros(len(folds), dtype=int)
    rule_guess = np.zeros(len(folds), dtype=int)
    for fold in range(FOLDS):
        test = folds == fold
        train = ~test
        net, _ = train_net(data["pictures"][train], data["targets"][train], seed=fold)
        net_guess[test] = predict(net, data["pictures"][test]).argmax(axis=1)
        cuts = fit_strength_rule(data["strengths"][train], data["targets"][train])
        rule_guess[test] = apply_strength_rule(data["strengths"][test], cuts)

    # which channels did the net get wrong most of the time?
    keys = [sweep.freq_key(f) for f in sweep.channel_list()]
    missed = []
    for channel in np.unique(data["channels"]):
        mine = data["channels"] == channel
        share = float((net_guess[mine] == data["targets"][mine]).mean())
        if share < 0.5:
            missed.append({"freq": keys[channel], "truth": CLASSES[data["targets"][mine][0]],
                           "said": CLASSES[np.bincount(net_guess[mine]).argmax()]})
    return {
        "net": {"balanced_accuracy": balanced_accuracy(data["targets"], net_guess),
                "confusion": confusion(data["targets"], net_guess)},
        "rule": {"balanced_accuracy": balanced_accuracy(data["targets"], rule_guess),
                 "confusion": confusion(data["targets"], rule_guess)},
        "missed_channels": missed,
    }


def train_and_report():
    data = load_dataset()
    if len(data["targets"]) == 0:
        raise RuntimeError("no training data yet: record some sweeps and run the listening pass first")
    channels_per_class = [int(len(np.unique(data["channels"][data["targets"] == c])))
                          for c in range(len(CLASSES))]
    if min(channels_per_class) < 2:
        raise RuntimeError("need at least 2 channels of every class to test honestly, have "
                           + str(dict(zip(CLASSES, channels_per_class))))
    report = cross_validate(data)
    report["pictures"] = int(len(data["targets"]))
    report["sweeps"] = len(sweep.load_sweeps())
    report["channels_per_class"] = dict(zip(CLASSES, channels_per_class))
    report["left_out"] = data["skipped"]
    answers = labels.load_json(labels.LABELS_FILE)
    report["flagged_for_review"] = [k for k in answers if answers[k]["needs_review"]]
    report["strength_cuts"] = list(fit_strength_rule(data["strengths"], data["targets"]))

    # the net the dashboard uses is trained on everything we have
    net, history = train_net(data["pictures"], data["targets"])
    report["loss_first_epoch"] = history[0]
    report["loss_last_epoch"] = history[-1]
    torch.save(net.state_dict(), MODEL_FILE)
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=1)
    return report


def load_net():
    if not os.path.exists(MODEL_FILE):
        return None
    net = build_net()
    net.load_state_dict(torch.load(MODEL_FILE))
    net.eval()
    return net


def load_report():
    if not os.path.exists(REPORT_FILE):
        return None
    with open(REPORT_FILE) as f:
        return json.load(f)
