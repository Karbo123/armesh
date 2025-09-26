"""train a model"""

import json
import hydra
import shutil
import os, sys
import os.path as osp
from time import time
from datetime import datetime
from omegaconf import OmegaConf
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DeepSpeedStrategy

import wandb
import torchmetrics


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if True:
    FOLDER_PY = osp.abspath(
        osp.dirname(__file__)
    )  # the folder of experimental python files
    ROOT = osp.abspath(osp.join(FOLDER_PY, "../.."))  # the very root of this project

    # import everything
    sys.path.append(FOLDER_PY)
    sys.path.append(ROOT)


from exp_utils import *
from codebase.utils import (
    get_length_aware_sampler,
    safe_shutdown,
)


def read_manyline_jsonl(file_path, category=None):
    """read the dataset (jsonl with many lines)"""
    merged = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    if isinstance(category, str) and (category not in data["name"]):
                        continue
                    if callable(category) and (not category(data["name"])):
                        continue
                    merged[data["name"]] = data["op_lst"]
                except json.JSONDecodeError:
                    continue
    return merged


def to_category_id(category_english_name):
    """convert shapenet english name to id string
    https://gist.github.com/tejaskhot/15ae62827d6e43b91a4b0c5c850c168e
    """
    mapping = {
        "bench": "02828884",
        "lamp": "03636649",
        "chair": "03001627",
        "table": "04379243",
    }
    if (category_english_name is None) or category_english_name == "all":  # use all
        return None
    if category_english_name.isdigit():  # already a category id
        assert category_english_name in mapping.values()
        return category_english_name
    if "lambda" in category_english_name:
        return eval(category_english_name)
    return mapping[category_english_name]  # convert to a category id


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


@hydra.main(version_base=None, config_path=FOLDER_PY, config_name="cfg")
def main_fn(cfg):

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # some folder paths, and preparation
    #   note : this part should be the same for all kinds of experiments

    if True:

        if cfg.resume is None:
            EXP_NAME = osp.relpath(FOLDER_PY, ROOT).replace(
                "/", "-"
            )  # the experiment name
            EXP_NAME += "--" + datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        else:
            EXP_NAME = osp.normpath(cfg.resume).split(os.sep)[1]
            assert EXP_NAME.startswith("exp-")

        FOLDER_EXP_OUT = osp.join(ROOT, "out", EXP_NAME)  # the output folder

        # find a never used folder for code backup
        def create_backup_folder(folder_exp_out):
            idx = 0
            get_folder = lambda i: osp.join(
                folder_exp_out, f"backup_{i:03}"
            )  # save backup files here
            while osp.isdir(get_folder(idx)):
                idx += 1
            return get_folder(idx)

        # backup some files
        def backup_code_(
            folder_backup,  # save to here
            code_dirs=[FOLDER_PY, osp.join(ROOT, "codebase")],
            code_wrap_dirs=["expfiles", "codebase"],
            suffix_lst=[".py", ".yaml", ".cpp", ".h", ".hpp"],
        ):

            for code_wrap_dir, code_dir in zip(code_wrap_dirs, code_dirs):
                for root_dir, _, file_name_lst in os.walk(code_dir):
                    for file_name in file_name_lst:
                        if any(file_name.endswith(s) for s in suffix_lst):
                            src_path = osp.join(root_dir, file_name)
                            rel_path = osp.join(
                                code_wrap_dir, osp.relpath(src_path, code_dir)
                            )
                            dst_path = osp.join(folder_backup, rel_path)
                            os.makedirs(osp.dirname(dst_path), exist_ok=True)
                            try:
                                shutil.copy2(src_path, dst_path)
                            except Exception as e:
                                print(f"[backup] skipped {src_path}, due to {e}")
            print(f"we have backup all files to here: {folder_backup}")

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # the dataset

    from codebase.bpe import BPE

    bpe = BPE.load(osp.join(ROOT, cfg.folder_dataset, "vocab.json"))
    assert bpe.vocab_size == cfg.vocab_total - 2
    num_classes = bpe.vocab_size + 2  # add SOS, EOS
    id_SOS = num_classes - 2  # start
    id_EOS = num_classes - 1  # end

    class PlAllMeshData(pl.LightningDataModule):
        def __init__(self, trainer):
            super().__init__()
            self.trainer = trainer

            #
            bpe_file = osp.join(ROOT, cfg.folder_dataset, "op_tok_bpe.jsonl")

            bpe_tokens = read_manyline_jsonl(
                bpe_file, category=to_category_id(cfg.category)
            )
            bpe_tokens = [[i, k, v] for i, (k, v) in enumerate(bpe_tokens.items())]

            # differently augmented shape should index the same shape identity
            self.idx_lst = [x[0] // cfg.num_variants_per_shape for x in bpe_tokens]
            self.name_lst = [x[1] for x in bpe_tokens]
            self.bpet_lst = [
                torch.tensor([id_SOS] + sum(x[2], []) + [id_EOS]) for x in bpe_tokens
            ]
            self.lengths = [len(x) - 1 for x in self.bpet_lst]  # remove EOS

        def __len__(self):
            return len(self.name_lst)

        def __getitem__(self, idx):
            return dict(
                idx=self.idx_lst[idx],
                name=self.name_lst[idx],
                bpet=self.bpet_lst[idx],
            )

        def _collate_fn(self, batch):
            """gather data from multiple batches"""
            out = defaultdict(list)
            for b in batch:
                for k, v in b.items():
                    out[k].append(v)
            # # # # # # # # # # # # # # # # # # # # # # # #
            out["nums"] = torch.tensor(
                [len(x) - 1 for x in out["bpet"]], dtype=torch.long
            )  # remove either SOS/EOS
            out["bpet"] = torch.cat(out["bpet"], dim=0)
            out["idx"] = torch.tensor(out["idx"])
            return out

        def _kwargs_loader(self, shuffle):
            return dict(
                num_workers=cfg.num_workers,
                collate_fn=self._collate_fn,
                batch_sampler=get_length_aware_sampler(  # workload balance
                    self.lengths,
                    num_total_max=cfg.load_total_max,
                    num_rank=self.trainer.world_size,
                    local_rank=self.trainer.local_rank,
                    shuffle=shuffle,
                ),
            )

        def train_dataloader(self):
            return DataLoader(self, **self._kwargs_loader(True))

        def val_dataloader(self):
            return DataLoader(self, **self._kwargs_loader(False))

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model

    class Model(pl.LightningModule):
        def __init__(self, num_obj):
            super().__init__()
            self.model = CausalTransformerDecoder(
                d_model=cfg.d_model, n_layers=cfg.n_layers, num_classes=num_classes
            )
            self.embedding_self = nn.Embedding(num_classes, embedding_dim=cfg.d_model)
            self.embedding_cross = nn.Embedding(
                num_obj, embedding_dim=cfg.num_cross_tokens * cfg.d_model
            )
            self.empty_token = nn.Parameter(
                torch.randn(cfg.num_cross_tokens * cfg.d_model)
            )
            self.norm_cond = nn.RMSNorm(
                cfg.d_model
            )  # to regularize/limit the scale of shape latents
            self.train_acc = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.validation_outputs = {"always": [], "never": []}

        def _compute_logits(self, batch, mode_uncond="random"):
            assert mode_uncond in ("random", "always", "never")
            nums_x = batch["nums"]

            nums_y = torch.full(
                [len(nums_x)], fill_value=cfg.num_cross_tokens, device=self.device
            )

            # the conditioning
            seqs_in_y = self.embedding_cross(batch["idx"]).clone()
            if (
                mode_uncond == "random"
            ):  # randomly set the condition to zeros to sometimes achieve unconditioning
                msk_uncond = (
                    torch.rand(len(nums_y), device=self.device) < cfg.prob_uncond
                )
            elif (
                mode_uncond == "always"
            ):  # always set the condition to zeros to achieve unconditioning
                msk_uncond = torch.full(
                    [len(nums_y)], fill_value=True, dtype=torch.bool, device=self.device
                )
            elif mode_uncond == "never":
                msk_uncond = torch.full(
                    [len(nums_y)],
                    fill_value=False,
                    dtype=torch.bool,
                    device=self.device,
                )
            msk_uncond = msk_uncond.to(dtype=seqs_in_y.dtype)[:, None]
            seqs_in_y = self.empty_token[None, :] * msk_uncond + seqs_in_y * (
                1 - msk_uncond
            )
            seqs_in_y = self.norm_cond(seqs_in_y.reshape(-1, cfg.d_model))

            seqs_all = batch["bpet"]
            seqs_in = seqs_all[seqs_all != id_EOS]
            seqs_out_gt = seqs_all[seqs_all != id_SOS]

            seqs_out_pred = self.model(
                x=self.embedding_self(seqs_in),
                y=seqs_in_y,
                nums_x=nums_x,
                nums_y=nums_y,
            )
            return seqs_out_pred, seqs_out_gt

        def training_step(self, batch, _):
            seqs_out_pred, seqs_out_gt = self._compute_logits(
                batch, mode_uncond="random"
            )

            # compute loss
            loss = F.cross_entropy(seqs_out_pred, seqs_out_gt)
            self.log("train_loss", loss)

            # accuracy of the current batch
            self.train_acc(seqs_out_pred, seqs_out_gt)
            self.log("train_acc", self.train_acc, prog_bar=True)

            # record time being used
            elapsed_minutes = (time() - self.start_time) / 60
            self.log("time_used_minutes", elapsed_minutes)

            # the current batch size
            self.log("batch_size", len(batch["nums"]))

            return loss

        def validation_step(self, batch, _):
            """compute test accuracy for a batch"""
            for mode_uncond in ("always", "never"):
                seqs_out_pred, seqs_out_gt = self._compute_logits(
                    batch, mode_uncond=mode_uncond
                )
                num_correct = (
                    (torch.argmax(seqs_out_pred, dim=1) == seqs_out_gt).sum().item()
                )
                num_total = len(seqs_out_gt)
                self.validation_outputs[mode_uncond].append(
                    dict(num_correct=num_correct, num_total=num_total)
                )

        def on_validation_epoch_end(self):
            """compute overall test accuracy"""
            for mode_uncond in ("always", "never"):
                # compute the overall test accuracy
                sum_num_correct = sum(
                    [x["num_correct"] for x in self.validation_outputs[mode_uncond]]
                )
                sum_num_total = sum(
                    [x["num_total"] for x in self.validation_outputs[mode_uncond]]
                )
                self.validation_outputs[mode_uncond].clear()

                num_correct_tensor = torch.tensor(
                    [sum_num_correct], dtype=torch.long, device=self.device
                )
                num_total_tensor = torch.tensor(
                    [sum_num_total], dtype=torch.long, device=self.device
                )
                dist.all_reduce(num_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(num_total_tensor, op=dist.ReduceOp.SUM)

                # the overall accuracy
                sum_num_correct = num_correct_tensor.item()
                sum_num_total = num_total_tensor.item()
                accuracy = sum_num_correct / sum_num_total

                # log the overall test accuracy
                name_cond = "uncond" if mode_uncond == "always" else "cond"
                self.log(f"accuracy_{name_cond}_overall", accuracy, sync_dist=True)
                self.log(
                    f"accuracy_{name_cond}_overall_num_abs",
                    abs(sum_num_correct - sum_num_total),
                    sync_dist=True,
                )

        def configure_optimizers(self):
            return None  # we use deepspeed's optimizer

        def on_train_start(self):
            self.start_time = time()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    class SetEpochCallback(Callback):
        """set random seed of the sampler to the value of current epoch"""

        def on_train_epoch_start(self, trainer, pl_module):
            """the framework won't automatically set this for us
            we set this to ensure randomness in training
            """
            trainer.train_dataloader.batch_sampler.set_epoch(trainer.current_epoch)

    class BackupCodeCallback(Callback):
        def on_fit_start(self, trainer, pl_module):
            if trainer.is_global_zero:
                os.makedirs(trainer.default_root_dir, exist_ok=True)
                FOLDER_BACKUP = create_backup_folder(trainer.default_root_dir)
                backup_code_(FOLDER_BACKUP)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    # logger
    wandb_logger = WandbLogger(
        project="ARMesh",
        save_dir=FOLDER_EXP_OUT,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # callbacks of saving models and logging lr
    cb_checkpoint_best = ModelCheckpoint(
        monitor="accuracy_cond_overall",
        mode="max",
        save_top_k=1,
        filename="best",
        every_n_epochs=cfg.save_every_epochs,
        save_on_train_epoch_end=False,  # after validation
        verbose=True,
    )
    cb_checkpoint_latest = ModelCheckpoint(
        save_top_k=1,
        filename="latest",
        every_n_epochs=cfg.save_every_epochs,
        save_on_train_epoch_end=True,  # right after every some epochs
        verbose=True,
    )
    cb_lr_monitor = LearningRateMonitor(logging_interval="epoch")
    cb_set_sampler_epochs = SetEpochCallback()
    cb_save_backup_files = BackupCodeCallback()

    # trainer
    trainer = pl.Trainer(
        default_root_dir=FOLDER_EXP_OUT,
        #
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        #
        logger=wandb_logger,
        gradient_clip_val=1.0,
        use_distributed_sampler=False,  # we handle this manually
        callbacks=[
            cb_checkpoint_best,
            cb_checkpoint_latest,
            cb_lr_monitor,
            cb_set_sampler_epochs,
            cb_save_backup_files,
        ],
        #
        max_steps=cfg.steps_max,
        log_every_n_steps=10,
        check_val_every_n_epoch=cfg.test_every_epochs,
        #
        strategy=DeepSpeedStrategy(
            stage=0,  # 0 is sufficient for small model
            config={
                "train_micro_batch_size_per_gpu": 1,
                "optimizer": {
                    "type": "AdamW",
                    "params": {"lr": cfg.lr_minmax[1], "weight_decay": 0.00},
                },
                "scheduler": {
                    "type": "WarmupCosineLR",
                    "params": {
                        "total_num_steps": cfg.steps_max,
                        "warmup_num_steps": cfg.steps_warmup,
                        "warmup_type": "linear",
                        "cos_min_ratio": cfg.lr_minmax[0] / cfg.lr_minmax[1],  #
                    },
                },
            },
        ),
    )

    # data, and model
    datamodule = PlAllMeshData(trainer)
    num_obj = len(datamodule.lengths) // cfg.num_variants_per_shape
    model = Model(num_obj)

    # starting training
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=None if cfg.resume is None else osp.join(ROOT, cfg.resume),
    )

    safe_shutdown()
    wandb.finish()


if __name__ == "__main__":
    main_fn()
