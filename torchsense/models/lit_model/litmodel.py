import lightning as L
import torch
from torchmetrics.functional.classification.accuracy import accuracy


class LitModel(L.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx: int):
        x, y = batch

        logits = self(x)

        loss = self.loss_fn(logits, y)
        accuracy_train = accuracy(logits.argmax(-1), y, num_classes=10, task="multiclass", top_k=1)
        self.log("loss", loss, prog_bar=True)
        self.log("accuracy", accuracy_train, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-4)
        return [optim]

    def validation_step(self, *args, **kwargs):
        return self.training_step(*args, **kwargs)
