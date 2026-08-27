import torch

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        # Automatically handles raw tensors to prevent graph retention / memory leaks
        if isinstance(val, torch.Tensor):
            val = val.detach().item()
            
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} (avg: {avg" + self.fmt + "})"
        return fmtstr.format(name=self.name, val=self.val, avg=self.avg)


# --- Working Example ---
if __name__ == "__main__":
    # Create the meter
    loss_meter = AverageMeter("Loss", ":.4f")

    print("Simulating a training loop...")
    for step in range(1, 6):
        # Create a dummy tensor with gradients attached (simulating a model loss)
        simulated_loss = torch.tensor(2.0 / step, requires_grad=True)

        # Update the meter (it safely detaches under the hood)
        loss_meter.update(simulated_loss)

        print(loss_meter)
