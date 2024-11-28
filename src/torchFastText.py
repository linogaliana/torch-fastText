import logging
import os
import time

import numpy as np
import torch
from torch.optim import SGD, Adam

from checkers import check_X, check_Y
from dataset import FastTextModelDataset
from pytorch_model import FastTextModel
from tokenizer import NGramTokenizer
from trainer.trainer import Trainer

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


class torchFastText:
    def __init__(
        self,
        num_buckets,
        embedding_dim,
        min_count,
        min_n,
        max_n,
        len_word_ngrams,
        sparse,
    ):
        self.num_buckets = num_buckets
        self.embedding_dim = embedding_dim
        self.min_count = min_count
        self.min_n = min_n
        self.max_n = max_n
        self.len_word_ngrams = len_word_ngrams
        self.tokenizer = None
        self.pytorch_model = None
        self.sparse = sparse
        self.trained = False
        self.num_categorical_features = None
        self.num_classes = None
        self.training_config = None

    def build_tokenizer(self, training_text):
        self.tokenizer = NGramTokenizer(
            self.min_count,
            self.min_n,
            self.max_n,
            self.num_buckets,
            self.len_word_ngrams,
            training_text,
        )

    def build(self, X_train, y_train):
        training_text, categorical_variables, no_cat_var = check_X(X_train)
        y_train = check_Y(y_train)
        self.num_classes = len(np.unique(y_train))
        self.num_categorical_features = categorical_variables.shape[1]
        self.build_tokenizer(training_text)
        self.pytorch_model = FastTextModel(
            tokenizer=self.tokenizer,
            embedding_dim=self.embedding_dim,
            vocab_size=self.num_buckets + self.tokenizer.get_nwords() + 1,
            num_classes=self.num_classes,
            categorical_vocabulary_sizes=np.max(categorical_variables, axis=0) + 1,
            padding_idx=self.num_buckets + self.tokenizer.get_nwords(),
            sparse=self.sparse,
            direct_bagging=True,
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_epochs: int,
        batch_size: int,
        device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        cpu_run: bool = False,
        num_workers: int = 12,
        optimizer=None,
        optimizer_params=None,
        lr: float = None,
        scheduler=None,
        loss=torch.nn.CrossEntropyLoss(),
        early_stopping: bool = True,
        patience_train=3,
        patience_scheduler=3,
        save_every_n_epoch: int = None,
        save_dir: str = None,
        check_val_every_n_epoch: int = 1,
        verbose: bool = False,
    ):
        if save_dir is None:
            self.save_dir = "torch_fastText_{}/".format(time.strftime("%Y%m%d-%H%M"))
        else:
            self.save_dir = save_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            if verbose:
                logger.info(
                    f"Created directory {self.save_dir}, checkpoints and models will be saved there."
                )

        ##### Formatting exception handling #####
        assert isinstance(loss, torch.nn.Module), "loss must be a PyTorch loss function."
        assert optimizer is None or optimizer.__module__.startswith(
            "torch.optim"
        ), "optimizer must be a PyTorch optimizer."
        assert (
            scheduler is None or scheduler.__module__ == "torch.optim.lr_scheduler"
        ), "scheduler must be a PyTorch scheduler."

        # checking right format for inputs
        if verbose:
            logger.info("Checking inputs...")

        training_text, train_categorical_variables, train_no_cat_var = check_X(X_train)
        val_text, val_categorical_variables, val_no_cat_var = check_X(X_val)
        y_train = check_Y(y_train)
        y_val = check_Y(y_val)

        # some checks
        assert (
            train_no_cat_var == val_no_cat_var
        ), "X_train and X_val must have the same number of categorical variables."
        # shape
        assert (
            X_train.shape[0] == y_train.shape[0]
        ), "X_train and y_train must have the same first dimension (number of observations)."
        assert (
            X_train.shape[1] == X_val.shape[1]
        ), "X_train and X_val must have the same number of columns."

        self.no_cat_var = train_no_cat_var

        if verbose:
            logger.info("Inputs successfully checked. Starting the training process..")

        ######## Starting the training process ########

        # Device
        if cpu_run:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if verbose:
            logger.info(f"Running on: {self.device}")

        # Build tokenizer PyTorch model (using training text and categorical variables)
        if self.tokenizer is None or self.pytorch_model is None:
            if verbose:
                start = time.time()
                logger.info("Building the model...")
            self.build(X_train, y_train)  # We use all categorical variables not to miss any labels
            if verbose:
                end = time.time()
                logger.info("Model successfully built in {:.2f} seconds.".format(end - start))

        self.pytorch_model = self.pytorch_model.to(self.device)

        # Datasets and dataloaders
        train_dataset = FastTextModelDataset(
            categorical_variables=train_categorical_variables,
            texts=training_text,
            outputs=y_train,
            tokenizer=self.tokenizer,
        )
        val_dataset = FastTextModelDataset(
            categorical_variables=val_categorical_variables,
            texts=val_text,
            outputs=y_val,
            tokenizer=self.tokenizer,
        )

        train_dataloader = train_dataset.create_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        val_dataloader = val_dataset.create_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )

        # Optimizer, scheduler and loss
        if optimizer is None:
            assert lr is not None, "Please provide a learning rate."
            if not self.sparse:
                self.optimizer = Adam(self.pytorch_model.parameters(), lr=lr)
            else:
                self.optimizer = SGD(self.pytorch_model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=patience_scheduler, factor=0.5, mode="min"
            )
            if verbose:
                logger.info("Using ReduceLROnPlateau scheduler.")
        else:
            self.scheduler = scheduler
        self.loss = loss

        # Trainer
        self.trainer = Trainer(
            model=self.pytorch_model,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=self.device,
            cpu_run=cpu_run,
            num_workers=num_workers,
            optimizer=self.optimizer,
            optimizer_params=optimizer_params,
            lr=lr,
            scheduler=self.scheduler,
            loss=loss,
            patience_train=patience_train,
            save_every_n_epoch=save_every_n_epoch,
            save_dir=self.save_dir,
            check_val_every_n_epoch=check_val_every_n_epoch,
            console_log=True,
            verbose=verbose,
        )

        torch.cuda.empty_cache()
        torch.set_float32_matmul_precision("medium")

        if verbose:
            logger.info("Launching training...")
            start = time.time()
        self.trainer.fit(train_dataloader, val_dataloader)
        if verbose:
            end = time.time()
            logger.info("Training done in {:.2f} seconds.".format(end - start))

        self.trained = True

    def validate(self, X, Y, batch_size=256, num_workers=12):
        """
        Validates the model on the given data.

        Args:
            X (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            Y (np.ndarray): Array of shape (N,) with the labels.

        Returns:
            float: The validation loss.
        """

        if not self.trained:
            raise Exception("Model must be trained first.")

        # checking right format for inputs
        text, categorical_variables, no_cat_var = check_X(X)
        y = check_Y(Y)

        if categorical_variables.shape[1] != self.num_categorical_features:
            raise Exception(
                f"X must have the same number of categorical variables as the training data ({self.num_categorical_features})."
            )

        self.pytorch_model.to(X.device)

        dataset = FastTextModelDataset(
            categorical_variables=categorical_variables,
            texts=text,
            outputs=y,
            tokenizer=self.tokenizer,
        )
        dataloader = dataset.create_dataloader(batch_size=batch_size, num_workers=num_workers)

        return self.trainer.test(self.pytorch_model, test_dataloaders=dataloader, verbose=False)

    def predict(self, X, top_k=1):
        """
        Predicts the "top_k" classes of the input text.

        Args:
            X (np.ndarray): Array of shape (N,d) with the first column being the text and the rest being the categorical variables.
            top_k (int): Number of classes to predict (by order of confidence).

        Returns:
            np.ndarray: Array of shape (N,top_k)
        """

        if not self.trained:
            raise Exception("Model must be trained first.")

        # checking right format for inputs
        text, categorical_variables, no_cat_var = check_X(X)
        if categorical_variables.shape[1] != self.num_categorical_features:
            raise Exception(
                f"X must have the same number of categorical variables as the training data ({self.num_categorical_features})."
            )

        self.pytorch_model.to(X.device)

        return self.pytorch_model.predict(text, categorical_variables, top_k=top_k)

    def predict_and_explain(self, X, top_k=1):
        if not self.trained:
            raise Exception("Model must be trained first.")

        # checking right format for inputs
        text, categorical_variables, no_cat_var = check_X(X)

        if categorical_variables.shape[1] != self.num_categorical_features:
            raise Exception(
                f"X must have the same number of categorical variables as the training data ({self.num_categorical_features})."
            )

        self.pytorch_model.to(X.device)

        return self.pytorch_model.predict_and_explain(text, categorical_variables, top_k=top_k)

    def quantize():
        # TODO
        pass

    def dequantize():
        # TODO
        pass
