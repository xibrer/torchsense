from torchsense.models.lit_model import LitModel
import lightning as L


class Trainer:
    def __init__(self, model, max_epochs=5, accelerator="auto"):
        self.model = LitModel(model)
        self.max_epochs = max_epochs
        self.accelerator = accelerator

    def fit(self, train_loader, val_loader):
        trainer = L.Trainer(
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,
        )
        trainer.fit(self.model, train_loader, val_loader)
