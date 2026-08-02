"""
plot_curves.py

Plots training and validation curves.
"""

import matplotlib.pyplot as plt

# ==========================================
# Paste your training history here
# ==========================================

train_losses = [
    0.1365731375648017,
    0.09126774611035221,
    0.09739843734196799
]

val_losses = [
    0.06289957677852176,
    0.09409384624509959,
    0.05523003612183975
]

train_accuracies = [
    0.9463114138328241,
    0.9638904191916663,
    0.9617869484649672
]

val_accuracies = [
    0.9759519038076152,
    0.9599198396793587,
    0.9759519038076152
]

epochs = range(1, len(train_losses) + 1)

# ==========================================
# Loss Curve
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    epochs,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.savefig("outputs/loss_curve.png")
plt.show()

# ==========================================
# Accuracy Curve
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracies,
    marker="o",
    label="Train Accuracy"
)

plt.plot(
    epochs,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.title("Training & Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.savefig("outputs/accuracy_curve.png")
plt.show()

print("✅ Curves saved inside outputs/")