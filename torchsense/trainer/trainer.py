from torchsense.models.lit_model import LitRegressModel,LitClassModel
import lightning as L


class Trainer:
    def __init__(self, model, task="r", max_epochs=5, accelerator="auto"):
        if task == "r":
            self.model = LitRegressModel(model)
        elif task == "c":
            self.model = LitClassModel(model)
        else:
            raise ValueError("task must be either 'r' or 'c'")
        self.max_epochs = max_epochs
        self.accelerator = accelerator

    def fit(self, train_loader, val_loader):
        trainer = L.Trainer(
            accelerator=self.accelerator,
            max_epochs=self.max_epochs,
        )
        trainer.fit(self.model, train_loader, val_loader)
