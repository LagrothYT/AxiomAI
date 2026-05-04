import math


class OverfitMonitor:
    def __init__(self, name, min_epochs=4, patience=2, min_delta=0.01, gap_threshold=0.35):
        self.name = name
        self.min_epochs = min_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.gap_threshold = gap_threshold
        self.history = []

    def update(self, epoch, train_loss, val_loss):
        if val_loss is None or not math.isfinite(float(val_loss)):
            return None

        train_loss = float(train_loss)
        val_loss = float(val_loss)
        self.history.append((int(epoch), train_loss, val_loss))

        if len(self.history) < self.min_epochs:
            return None

        current_gap = val_loss - train_loss
        previous_best_val = min(v for _, _, v in self.history[:-1])
        recent = self.history[-(self.patience + 1):]

        val_getting_worse = len(recent) == self.patience + 1 and all(
            recent[i][2] > recent[i - 1][2] + self.min_delta
            for i in range(1, len(recent))
        )
        train_getting_better = recent[-1][1] < recent[0][1] - self.min_delta

        if val_getting_worse and train_getting_better:
            return (
                f"  ⚠ Overfit Watch │ {self.name}: train loss is still falling, "
                f"but val loss rose for {self.patience} epochs. Consider stopping soon."
            )

        if current_gap >= self.gap_threshold and val_loss > previous_best_val + self.min_delta:
            return (
                f"  ⚠ Overfit Watch │ {self.name}: validation is drifting away from train "
                f"(gap {current_gap:.3f}). More epochs may memorize instead of generalize."
            )

        return None
