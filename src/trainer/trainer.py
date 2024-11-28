import copy
import os

import numpy as np
import torch
from torchmetrics import Accuracy
from tqdm import tqdm

from trainer.training_callbacks import TrainingLogger


class Trainer:
    def __init__(
        self,
        model,
        num_epochs: int,
        batch_size: int,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cpu_run: bool = False,
        num_workers: int = 12,
        optimizer=None,
        optimizer_params=None,
        lr: float = None,
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
        loss=torch.nn.CrossEntropyLoss(),
        early_stopping: bool = True,
        patience_train=3,
        save_every_n_epoch: int = None,
        save_dir: str = None,
        check_val_every_n_epoch: int = 1,
        console_log: bool = True,
        verbose: bool = False,
    ):
        if early_stopping:
            assert (
                patience_train < num_epochs
            ), "Patience for early stopping should be less than the number of epochs"
            assert patience_train > 0, "Patience for early stopping should be greater than 0"

        self.pytorch_model = model  # PyTorch model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device
        self.cpu_run = cpu_run
        self.num_workers = num_workers
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.lr = lr
        self.scheduler = scheduler
        self.loss = loss
        self.early_stopping = early_stopping
        self.patience_train = patience_train
        self.save_every_n_epoch = save_every_n_epoch
        self.save_dir = save_dir
        self.check_val_every_n_epoch = check_val_every_n_epoch

        log_file = os.path.join(self.save_dir, "training.log")
        self.logger = TrainingLogger(log_file=log_file, console_log=console_log)

    def save(self, path):
        full_folder_path = os.path.join(self.save_path, path)
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        filename = os.path.join(full_folder_path, "model.pth")
        model_scripted = torch.jit.script(self.pytorch_model.to("cpu"))
        model_scripted.save(filename)
        ckpt_filename = os.path.join(full_folder_path, "checkpoint.pth")
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
            },
            ckpt_filename,
        )

    def fit(self, train_dataloader, val_dataloader):
        """
        Args:
            train_dataloader (torch.utils.data.DataLoader): Training data loader
            val_dataloader (torch.utils.data.DataLoader): Validation data loader
        """
        self.logger.log(
            f"Number of trainable parameters: {sum(p.numel() for p in self.pytorch_model.parameters() if p.requires_grad)}"
        )
        self.logger.log(
            f"Total number of parameters: {sum(p.numel() for p in self.pytorch_model.parameters())}"
        )

        # Tracking metrics for logging
        self.train_metrics = {"losses": [], "accuracies": []}
        self.val_metrics = {"losses": [], "accuracies": []}

        # Metrics
        accuracy_fn = Accuracy(task="multiclass", num_classes=self.pytorch_model.num_classes).to(
            self.device
        )

        # Training parameters
        patience = 0
        device = self.device
        self.pytorch_model = self.pytorch_model.to(device)
        self.best_val_loss = np.inf
        num_steps_per_epoch = len(train_dataloader)
        total_steps = self.num_epochs * num_steps_per_epoch

        # Main progress bar for steps
        pbar = tqdm(total=total_steps, desc="Training Steps")

        for epoch in range(self.num_epochs):
            # Training phase
            self.pytorch_model.train()
            train_loss, train_accuracy = 0, 0

            for step, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()
                x, cat, y = batch
                x, cat, y = x.to(device), cat.to(device), y.to(device)

                pred = self.pytorch_model(x, cat)
                loss = self.loss(pred, y)

                loss.backward()
                self.optimizer.step()

                # Accumulate metrics
                train_loss += loss.item()
                train_accuracy += accuracy_fn(pred, y)

                # Update main progress bar
                pbar.set_description(
                    f"Epoch {epoch+1}/{self.num_epochs}, Step {step+1}/{num_steps_per_epoch}"
                )
                pbar.set_postfix(
                    {
                        "Train Loss": f"{loss.item():.4f}",
                        "Train Accuracy": f"{train_accuracy / (step + 1):.4f}",
                    }
                )
                pbar.update(1)

            # Average training metrics
            train_loss /= num_steps_per_epoch
            train_accuracy /= num_steps_per_epoch

            # Store metrics for logging
            self.train_metrics["losses"].append(train_loss)
            self.train_metrics["accuracies"].append(train_accuracy)

            self.logger.log(
                f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}"
            )

            # Validation phase
            if epoch % self.check_val_every_n_epoch == 0:
                self.pytorch_model.eval()
                val_loss, val_accuracy = 0, 0

                # Validation progress bar
                with tqdm(
                    total=len(val_dataloader),
                    desc=f"Validation (Epoch {epoch+1}/{self.num_epochs})",
                    leave=False,
                ) as val_pbar:
                    with torch.no_grad():
                        for batch in val_pbar:
                            x, cat, y = batch
                            x, cat, y = x.to(device), cat.to(device), y.to(device)

                            y_hat = self.pytorch_model(x, cat)
                            loss = self.loss(y_hat, y)

                            val_loss += loss.item()
                            val_accuracy += accuracy_fn(y_hat, y)

                            # Update validation progress bar
                            val_pbar.set_postfix(
                                {
                                    "Val Loss": f"{loss.item():.4f}",
                                    "Val Accuracy": f"{val_accuracy / (val_pbar.n + 1):.4f}",
                                }
                            )
                            val_pbar.update(1)

                # Average validation metrics
                val_loss /= len(val_dataloader)
                val_accuracy /= len(val_dataloader)

                # Store metrics for logging
                self.val_metrics["losses"].append(val_loss)
                self.val_metrics["accuracies"].append(val_accuracy)

                self.logger.log(
                    f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}, Val Accuracy = {val_accuracy:.4f}"
                )

                # Update main progress bar with validation info
                pbar.set_postfix(
                    {
                        "Train Loss": f"{train_loss:.4f}",
                        "Val Loss": f"{val_loss:.4f}",
                        "Train Accuracy": f"{train_accuracy:.4f}",
                        "Val Accuracy": f"{val_accuracy:.4f}",
                    }
                )

            # Model saving, learning rate scheduling, and early stopping logic
            if epoch % self.save_every_n_epoch == 0:
                torch.save(self.pytorch_model, self.save_path)

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            if self.early_stopping:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = copy.deepcopy(self.pytorch_model.to("cpu"))
                    patience = 0
                else:
                    patience += 1
                    if patience == self.patience_train:
                        self.logger.log(
                            f"Early stopping triggered at epoch {epoch+1}", level="error"
                        )
                        break

        pbar.close()
