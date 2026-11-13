# trains the net and prints how it did next to the simple strength rule
#   python train.py

import model


def print_table(name, result):
    print()
    print(name, "- balanced accuracy", round(100 * result["balanced_accuracy"], 1), "%")
    print("   rows = the truth, columns = what it said")
    print("            " + "".join(c.rjust(8) for c in model.CLASSES))
    for c, row in zip(model.CLASSES, result["confusion"]):
        print("   " + c.ljust(9) + "".join(str(n).rjust(8) for n in row))


if __name__ == "__main__":
    report = model.train_and_report()
    print(report["pictures"], "pictures from", report["sweeps"], "sweeps")
    print("channels per class:", report["channels_per_class"])
    print("flagged for a listen:", ", ".join(report["flagged_for_review"]) or "none")
    print_table("NEURAL NET", report["net"])
    print_table("STRENGTH RULE", report["rule"])
    print()
    print("channels the net usually gets wrong:")
    for miss in report["missed_channels"]:
        print("  ", miss["freq"], "is", miss["truth"], "but it says", miss["said"])
    if not report["missed_channels"]:
        print("   none")
    print()
    print("training loss went from", round(report["loss_first_epoch"], 3),
          "to", round(report["loss_last_epoch"], 3))
