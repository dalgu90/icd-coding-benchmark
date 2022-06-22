import math
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.modules.embeddings import *
from src.modules.losses import *
from src.modules.metrics import *
from src.modules.optimizers import *
from src.modules.schedulers import *
from src.modules.tokenizers import *
from src.utils.checkpoint_savers import *
from src.utils.configuration import Config
from src.utils.file_loaders import save_json
from src.utils.graph_writers import *
from src.utils.mapper import ConfigMapper
from src.utils.misc import *
from src.utils.text_loggers import get_logger

logger = get_logger(__name__)


@ConfigMapper.map("trainers", "base_trainer")
class BaseTrainer:
    def __init__(self, config):
        cls_name = self.__class__.__name__
        logger.info(f"Initializing {cls_name}")
        logger.debug(f"Initializing {cls_name} with config: {config}")
        self.config = config

        # Loss function
        self.loss_fn = ConfigMapper.get_object(
            "losses",
            self.config.loss.name,
        )(self.config.loss.params)
        logger.debug(
            f"Created loss function {self.loss_fn.__class__.__name__} with "
            f"config: {self.config.loss.params}"
        )

        # Evaluation metrics
        self.eval_metrics = {}
        for config_dict in self.config.eval_metrics:
            metric_name = config_dict["name"]
            self.eval_metrics[metric_name] = load_metric(config_dict)

    def train(self, model, train_dataset, val_dataset=None):
        """Train the model"""
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Data loader
        train_loader_config = self.config.data_loader.as_dict()
        logger.debug(f"Creating train DataLoader: {train_loader_config}")
        if "collate_fn" in dir(train_dataset):
            train_loader_config["collate_fn"] = train_dataset.collate_fn
        train_loader = DataLoader(train_dataset, **train_loader_config)
        if val_dataset:
            # We force the val dataset not shuffled and fully used
            val_loader_config = self.config.data_loader.as_dict()
            val_loader_config["drop_last"] = False
            val_loader_config["shuffle"] = False
            logger.debug(f"Creating val DataLoader: {val_loader_config}")
            if "collate_fn" in dir(val_dataset):
                val_loader_config["collate_fn"] = val_dataset.collate_fn
            val_loader = DataLoader(val_dataset, **val_loader_config)
        batch_size = self.config.data_loader.batch_size
        if train_loader_config["drop_last"]:
            num_train_batch = math.floor(len(train_dataset) / batch_size)
        else:
            num_train_batch = math.ceil(len(train_dataset) / batch_size)

        # Optimizer & LR scheduler
        optimizer = ConfigMapper.get_object(
            "optimizers", self.config.optimizer.name
        )(model.parameters(), **self.config.optimizer.params.as_dict())
        logger.debug(
            f"Created optimizer {optimizer.__class__.__name__} with config: "
            f"{self.config.optimizer.params}"
        )
        scheduler = None
        if self.config.lr_scheduler is not None:
            scheduler = ConfigMapper.get_object(
                "schedulers", self.config.lr_scheduler.name
            )(optimizer, **self.config.lr_scheduler.params.as_dict())
            logger.debug(
                f"Created scheduler {scheduler.__class__.__name__} with "
                f"config: {self.config.lr_scheduler.params}"
            )

        # Add evaluation metrics for graph
        for config_dict in (
            self.config.graph.train.metric + self.config.graph.val.metric
        ):
            metric_name = config_dict["name"]
            if metric_name not in self.eval_metrics and metric_name != "loss":
                self.eval_metrics[metric_name] = load_metric(config_dict)
        train_metric_names = [
            metric["name"] for metric in self.config.graph.train.metric
        ]
        val_metric_names = [
            metric["name"] for metric in self.config.graph.val.metric
        ]

        # Stopping criterion: (metric, max/min, patience)
        max_epochs = int(self.config.max_epochs)
        stopping_criterion = None
        if self.config.stopping_criterion is not None:
            sc_config = self.config.stopping_criterion
            # Load metric
            sc_metric_config = sc_config.metric.as_dict()
            if sc_metric_config["name"] in self.eval_metrics:
                sc_metric = self.eval_metrics[sc_metric_config["name"]]
            else:
                sc_metric = load_metric(sc_metric_config)
            # Metric + max/min + patience
            stopping_criterion = (
                sc_metric,
                sc_config.desired,
                sc_config.patience,
            )
            best_stopping_val = float("inf")
            best_stopping_epoch = 0
            if sc_config.desired == "max":
                best_stopping_val *= -1.0

        if self.config.use_gpu:
            model.cuda()
            logger.info("Use GPU")

        # Checkpoint saver
        ckpt_saver = ConfigMapper.get_object(
            "checkpoint_savers", self.config.checkpoint_saver.name
        )(self.config.checkpoint_saver.params)

        # Load latest checkpoint
        latest_ckpt = ckpt_saver.get_latest_checkpoint()
        if latest_ckpt is not None:
            ckpt_epoch, ckpt_fname = latest_ckpt
            ckpt_saver.load_ckpt(
                model=model, optimizer=optimizer, ckpt_fname=ckpt_fname
            )
            logger.info(f"Checkpoint loaded from {ckpt_fname}")
            init_epoch = ckpt_epoch + 1
        else:
            init_epoch = 0
        global_step = (len(train_dataset) // batch_size) * init_epoch

        # Graph Writer (tensorboard)
        writer = ConfigMapper.get_object(
            "graph_writers", self.config.graph.writer.name
        )(self.config.graph.writer.params)

        # Train!
        for epoch in range(init_epoch, max_epochs):
            # Print training epoch
            logger.info(f"Epoch: {epoch}/{max_epochs}, Step {global_step:6}")

            model.train()

            # Train for one epoch
            pbar = tqdm(total=num_train_batch)
            pbar.set_description(f"Epoch {epoch}")
            for batch_train in train_loader:
                optimizer.zero_grad()

                batch_inputs, batch_labels = batch_train
                if self.config.use_gpu:
                    batch_inputs = batch_inputs.cuda()
                    batch_labels = batch_labels.cuda()

                # Initialise the hidden states.
                if self.config.initialise_hidden_states:
                    model.init_hidden(batch_inputs.size(0))

                batch_outputs = model(batch_inputs)
                batch_loss = self.loss_fn(
                    input=batch_outputs, target=batch_labels
                )
                if "regularizer" in dir(model):
                    batch_loss += model.regularizer(labels=batch_labels)

                batch_loss.backward()

                optimizer.step()
                if scheduler:
                    scheduler.step()

                # Write graph on proper steps (train)
                if (
                    self.config.graph.train.interval_unit == "step"
                    and global_step % self.config.graph.train.interval == 0
                ):
                    train_metric_vals = self._compute_metrics(
                        outputs=batch_outputs.detach().cpu(),
                        labels=batch_labels.cpu(),
                        metric_names=train_metric_names,
                    )
                    for metric_name, metric_val in train_metric_vals.items():
                        writer.write_scalar(
                            f"train/{metric_name}", metric_val, step=global_step
                        )

                pbar.set_postfix_str(f"Train Loss: {batch_loss.item():.6f}")
                pbar.update(1)

                global_step += 1
            pbar.close()

            # Evaluate on eval dataset
            if val_dataset:
                val_outputs, val_labels = self._forward_epoch(
                    model, dataloader=val_loader
                )
                val_loss = self.loss_fn(input=val_outputs, target=val_labels)
                val_labels_np = val_labels.numpy()
                val_prob = torch.sigmoid(val_outputs).numpy()
                val_pred = val_prob.round()
                logger.info("Evaluate on val dataset")
                for metric_config in self.config.eval_metrics:
                    metric_name = metric_config["name"]
                    metric_val = self.eval_metrics[metric_name](
                        y_true=val_labels_np, y_pred=val_pred, p_pred=val_prob
                    )
                    logger.info(f"{metric_name:>12}: {metric_val:6f}")

            # Plot graph on proper epochs (val)
            if (
                val_dataset
                and self.config.graph.val.interval_unit == "epoch"
                and epoch % self.config.graph.val.interval == 0
            ):
                val_metric_vals = self._compute_metrics(
                    outputs=val_outputs,
                    labels=val_labels,
                    metric_names=val_metric_names,
                )
                for metric_name, metric_val in val_metric_vals.items():
                    writer.write_scalar(
                        f"val/{metric_name}", metric_val, step=global_step
                    )

            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    # ReduceLROnPlateau uses validation loss
                    if val_dataset:
                        scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Update best stopping condition
            if val_dataset and stopping_criterion:
                metric, desired, patience = stopping_criterion
                stopping_val = metric(
                    y_true=val_labels, p_pred=val_outputs, y_pred=val_pred
                )
                if desired == "max" and stopping_val > best_stopping_val:
                    best_stopping_val = stopping_val
                    best_stopping_epoch = epoch
                if desired == "min" and stopping_val < best_stopping_val:
                    best_stopping_val = stopping_val
                    best_stopping_epoch = epoch

            # Checkpoint 1. Per interval epoch
            if ckpt_saver.check_interval(epoch):
                ckpt_fname = ckpt_saver.save_ckpt(
                    model=model, optimizer=optimizer, train_iter=epoch
                )
                logger.info(f"Checkpoint saved to {ckpt_fname}")

            # Checkpoint 2. Best val metric
            if val_dataset:
                metric_val, is_best = ckpt_saver.check_best(
                    y_true=val_labels, p_pred=val_prob, y_pred=val_pred
                )
                if is_best:
                    ckpt_fname = ckpt_saver.save_ckpt(
                        model=model,
                        optimizer=optimizer,
                        train_iter=epoch,
                        is_best=True,
                        metric_val=metric_val,
                    )
                    logger.info(
                        f"Checkpoint saved to {ckpt_fname} "
                        f"({ckpt_saver.config.metric.name}: "
                        f"{metric_val:.6f})"
                    )

            # Stop training if condition met
            if val_dataset and (epoch - best_stopping_epoch >= patience):
                break

        # Wrapping up
        # Save the last checkpoint, if not saved above
        if not ckpt_saver.check_interval(epoch):
            ckpt_fname = ckpt_saver.save_ckpt(
                model=model, optimizer=optimizer, train_iter=epoch
            )
            logger.info(f"Checkpoint saved to {ckpt_fname}")

        logger.info("Training completed")
        return

    def test(self, model, test_dataset):
        """Load the best or latest ckpt and evalutate on the given dataset."""
        # Load the best or the latest model
        ckpt_saver = ConfigMapper.get_object(
            "checkpoint_savers", self.config.checkpoint_saver.name
        )(self.config.checkpoint_saver.params)

        best_ckpt = ckpt_saver.get_best_checkpoint()
        if best_ckpt is not None:
            ckpt_fname = best_ckpt
        else:
            latest_ckpt = ckpt_saver.get_latest_checkpoint()
            if latest_ckpt is not None:
                ckpt_fname = latest_ckpt[1]
            else:
                logger.error("Cannot find a model checkpoint")
                return
        ckpt_saver.load_ckpt(model, ckpt_fname, optimizer=None)
        logger.info(f"Loaded checkpoint from {ckpt_fname}")

        if self.config.use_gpu:
            model.cuda()
            logger.info("Use GPU")

        # Evaluate on test dataset
        logger.info("Evaluating on test dataset")
        metric_vals = self.evaluate(model, test_dataset)

        # Print and save results
        for metric_name, metric_val in metric_vals.items():
            logger.info(f"{metric_name:>12}: {metric_val:6f}")

        result_fpath = os.path.join(self.config.output_dir, "test_result.json")
        logger.info(f"Saving result on {result_fpath}")
        save_json(metric_vals, result_fpath)

    def evaluate(self, model, dataset=None, dataloader=None):
        """Evaluate the model on the given dataset."""
        # Get preds and labels for the whole epoch
        epoch_outputs, epoch_labels = self._forward_epoch(
            model, dataset=dataset, dataloader=dataloader
        )

        # Evaluate the predictions using self.eval_metrics
        return self._compute_metrics(epoch_outputs, epoch_labels)

    def _compute_metrics(self, outputs, labels, metric_names=None):
        """
        Compute the metrics of given names. Inputs should be Torch tensors.
        """
        metric_vals = {}
        if metric_names is None:
            metric_names = self.eval_metrics.keys()

        labels_np = np.array(labels)
        probs = np.array(torch.sigmoid(outputs))
        preds = probs.round()
        for metric_name in metric_names:
            if metric_name == "loss":
                metric_vals["loss"] = self.loss_fn(input=outputs, target=labels)
            else:
                metric_vals[metric_name] = self.eval_metrics[metric_name](
                    y_true=labels_np, p_pred=probs, y_pred=preds
                )
        return metric_vals

    def _forward_epoch(self, model, dataset=None, dataloader=None):
        """Compute the forward pass on the given dataset."""
        assert dataset or dataloader

        # Dataloader
        if dataloader is None:
            # We force the data loader is not shuffled and fully checked
            data_config = self.config.data_loader.as_dict()
            data_config["drop_last"] = False
            data_config["shuffle"] = False
            logger.debug(f"Creating test DataLoader: {data_config}")
            if "collate_fn" in dir(dataset):
                data_config["collate_fn"] = dataset.collate_fn
            dataloader = DataLoader(dataset, **data_config)

        # Forward for the whole batch
        model.eval()
        epoch_outputs, epoch_labels = [], []
        with torch.no_grad():
            for i, (batch_inputs, batch_labels) in enumerate(dataloader):
                if self.config.use_gpu:
                    batch_inputs = batch_inputs.cuda()
                    batch_labels = batch_labels.cuda()

                # Initialise the hidden states.
                if self.config.initialise_hidden_states:
                    model.init_hidden(batch_inputs.size(0))

                batch_outputs = model(batch_inputs)
                epoch_labels.append(batch_labels.cpu())
                epoch_outputs.append(batch_outputs.cpu())

        # Concat
        epoch_labels = torch.cat(epoch_labels, 0)
        epoch_outputs = torch.cat(epoch_outputs, 0)

        return epoch_outputs, epoch_labels
