import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

with open("data.json", "r") as f:
    trainer_states = json.load(f)

log_history = trainer_states[-1]["log_history"]

training_data = []
eval_data = []

for entry in log_history:
    if "eval_loss" in entry:
        eval_data.append(
            {
                "step": entry["step"],
                "epoch": entry["epoch"],
                "eval_loss": entry["eval_loss"],
                "eval_wer": entry["eval_wer"],
            }
        )
    elif "loss" in entry:
        training_data.append(
            {
                "step": entry["step"],
                "epoch": entry["epoch"],
                "loss": entry["loss"],
                "learning_rate": entry["learning_rate"],
            }
        )

train_df = pd.DataFrame(training_data)
eval_df = pd.DataFrame(eval_data)

# Set a nice style for the plots
sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 12})

# Set up the figure layout (2x2 grid of plots)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Whisper-Small Malayalam Fine-tuning Metrics", fontsize=16)

# 1. Training Loss vs Epochs
axes[0, 0].plot(train_df["epoch"], train_df["loss"], "o-", color="#1f77b4", linewidth=2)
axes[0, 0].set_title("Training Loss vs Epochs")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylabel("Training Loss")
axes[0, 0].grid(True)

# 2. Validation Loss vs Epochs
axes[0, 1].plot(
    eval_df["epoch"], eval_df["eval_loss"], "o-", color="#ff7f0e", linewidth=2
)
axes[0, 1].set_title("Validation Loss vs Epochs")
axes[0, 1].set_xlabel("Epochs")
axes[0, 1].set_ylabel("Validation Loss")
axes[0, 1].grid(True)

# 3. WER vs Epochs
axes[1, 0].plot(
    eval_df["epoch"], eval_df["eval_wer"], "o-", color="#2ca02c", linewidth=2
)
axes[1, 0].set_title("Word Error Rate (WER) vs Epochs")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("WER (%)")
axes[1, 0].grid(True)

train_loss_at_eval = []
for eval_step in eval_df["step"]:
    closest_train_entry = train_df[train_df["step"] <= eval_step].iloc[-1]
    train_loss_at_eval.append(closest_train_entry["loss"])

axes[1, 1].plot(
    train_loss_at_eval, eval_df["eval_wer"], "o-", color="#d62728", linewidth=2
)
axes[1, 1].set_title("WER vs Training Loss")
axes[1, 1].set_xlabel("Training Loss")
axes[1, 1].set_ylabel("WER (%)")
axes[1, 1].grid(True)

best_wer_idx = eval_df["eval_wer"].idxmin()
best_wer = eval_df.loc[best_wer_idx, "eval_wer"]
best_epoch = eval_df.loc[best_wer_idx, "epoch"]
best_step = eval_df.loc[best_wer_idx, "step"]

for i in range(2):
    for j in range(2):
        if i == 1 and j == 0:
            axes[i, j].annotate(
                f"Best WER: {best_wer:.2f}%\nEpoch: {best_epoch:.2f}\nStep: {best_step}",
                xy=(best_epoch, best_wer),
                xytext=(best_epoch + 0.01, best_wer + 5),
                arrowprops=dict(facecolor="black", shrink=0.05, width=1.5),
            )

plt.figure(figsize=(14, 6))
plt.suptitle("Learning Rate Schedule and Training Progress", fontsize=16)

ax1 = plt.subplot(1, 1, 1)
ax1.plot(
    train_df["epoch"], train_df["learning_rate"], "o-", color="purple", linewidth=2
)
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Learning Rate", color="purple")
ax1.tick_params(axis="y", labelcolor="purple")
ax1.set_title("Learning Rate Schedule")
ax1.grid(True)

ax2 = ax1.twinx()
ax2.plot(train_df["epoch"], train_df["loss"], "o-", color="green", alpha=0.7)
ax2.set_ylabel("Training Loss", color="green")
ax2.tick_params(axis="y", labelcolor="green")

max_lr_idx = train_df["learning_rate"].idxmax()
max_lr = train_df.loc[max_lr_idx, "learning_rate"]
max_lr_epoch = train_df.loc[max_lr_idx, "epoch"]

plt.tight_layout()
fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.figure(figsize=(16, 8))
plt.suptitle("Whisper-Small Malayalam Fine-tuning Progress Summary", fontsize=16)
plt.subplot(1, 1, 1)
plt.plot(
    train_df["epoch"],
    train_df["loss"],
    "o-",
    color="#1f77b4",
    linewidth=2,
    label="Training Loss",
)
plt.plot(
    eval_df["epoch"],
    eval_df["eval_loss"],
    "s-",
    color="#ff7f0e",
    linewidth=2,
    label="Validation Loss",
)

ax3 = plt.gca()
ax4 = ax3.twinx()
ax4.plot(
    eval_df["epoch"],
    eval_df["eval_wer"],
    "^-",
    color="#2ca02c",
    linewidth=2,
    label="WER (%)",
)
ax4.set_ylabel("WER (%)", color="#2ca02c")
ax4.tick_params(axis="y", labelcolor="#2ca02c")

# Combine legends
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Combined Training Metrics")
plt.grid(True)

best_loss_idx = eval_df["eval_loss"].idxmin()
best_loss = eval_df.loc[best_loss_idx, "eval_loss"]
best_loss_epoch = eval_df.loc[best_loss_idx, "epoch"]

textstr = "\n".join(
    (
        f"Initial WER: {eval_df.iloc[0]['eval_wer']:.2f}%",
        f"Final WER: {eval_df.iloc[-1]['eval_wer']:.2f}%",
        f"Best WER: {best_wer:.2f}% (epoch {best_epoch:.2f})",
        f"Best Val Loss: {best_loss:.4f} (epoch {best_loss_epoch:.2f})",
        f"WER Reduction: {eval_df.iloc[0]['eval_wer'] - eval_df.iloc[-1]['eval_wer']:.2f}%",
        f"Training Steps: {train_df['step'].max()}",
    )
)

props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
plt.gcf().text(0.15, 0.15, textstr, fontsize=12, verticalalignment="bottom", bbox=props)

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()

print(f"Final model performance after {train_df['epoch'].max():.3f} epochs:")
print(f"- Training Loss: {train_df['loss'].iloc[-1]:.4f}")
print(f"- Validation Loss: {eval_df['eval_loss'].iloc[-1]:.4f}")
print(f"- Word Error Rate: {eval_df['eval_wer'].iloc[-1]:.2f}%")
print(
    f"- WER Reduction: {eval_df['eval_wer'].iloc[0] - eval_df['eval_wer'].iloc[-1]:.2f}%"
)
