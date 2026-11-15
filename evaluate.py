import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, classification_report
)


sign_list = [
    "20km/h", "30km/h", "50km/h", "60km/h",
    "70km/h", "80km/h", "End 80", "100km/h",
    "120km/h", "No passing", "No pass trucks",
    "Priority next", "Priority road", "Yield",
    "Stop", "No vehicles", "No trucks", "No entry",
    "Caution", "Left curve", "Right curve",
    "Double curve", "Bumpy", "Slippery",
    "Narrows right", "Road work", "Signals",
    "Pedestrians", "Children", "Bicycles",
    "Ice snow", "Animals", "End limits",
    "Turn right", "Turn left", "Straight",
    "Straight right", "Straight left",
    "Keep right", "Keep left", "Roundabout",
    "End no pass", "End no pass trucks"
]


def test_model(model, xtest, ytest):
    batch_size = 64
    all_preds = []
    for i in range(0, len(xtest), batch_size):
        batch = xtest[i:i+batch_size]
        preds = model(batch, training=False).numpy()
        all_preds.append(preds)
    pred_probs = np.concatenate(all_preds, axis=0)

    y_pred = np.argmax(pred_probs, axis=1)
    y_true = np.argmax(ytest, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\ntest accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_true, y_pred,
          target_names=sign_list, zero_division=0))

    return y_true, y_pred, acc


def plot_history(h, name=""):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h.history["accuracy"], label="train")
    ax[0].plot(h.history["val_accuracy"], label="val")
    ax[0].set_title(name + " accuracy")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(h.history["loss"], label="train")
    ax[1].plot(h.history["val_loss"], label="val")
    ax[1].set_title(name + " loss")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(name + "_history.png", dpi=120)
    plt.show()
    print("saved", name + "_history.png")


def plot_cm(yt, yp, name=""):
    cm = confusion_matrix(yt, yp)
    plt.figure(figsize=(16, 13))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=sign_list,
                yticklabels=sign_list)
    plt.title(name + " confusion matrix")
    plt.ylabel("actual")
    plt.xlabel("predicted")
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    plt.savefig(name + "_cm.png", dpi=120)
    plt.show()


def show_preds(Xt, yt, yp, n=12):
    picks = np.random.choice(len(Xt), n, replace=False)
    fig, ax = plt.subplots(3, 4, figsize=(13, 9))
    ax = ax.ravel()
    for i, idx in enumerate(picks):
        ax[i].imshow(Xt[idx])
        actual = sign_list[yt[idx]]
        pred = sign_list[yp[idx]]
        col = "green" if yt[idx] == yp[idx] else "red"
        ax[i].set_title(f"A:{actual[:11]}\nP:{pred[:11]}",
                        fontsize=7, color=col)
        ax[i].axis("off")
    plt.suptitle("Sample Results", fontsize=11)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=120)
    plt.show()


def bar_chart(scores):
    plt.figure(figsize=(6, 4))
    names = list(scores.keys())
    vals = [v * 100 for v in scores.values()]
    bars = plt.bar(names, vals,
                   color=["#1565C0", "#BF360C"], width=0.4)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2,
                 b.get_height() - 2,
                 f"{v:.1f}%",
                 ha="center", color="white",
                 fontsize=12, fontweight="bold")
    plt.ylim(80, 100)
    plt.ylabel("test accuracy %")
    plt.title("model comparison")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=120)
    plt.show()

