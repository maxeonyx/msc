#!python

import warnings
from datetime import datetime
from pathlib import Path

import PIL

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import random_split, DataLoader

import torchvision
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import ignite
import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, global_step_from_engine
from ignite.metrics import Bleu
from ignite.utils import manual_seed, setup_logger

from datasets import load_dataset
from transformers import ViTConfig, ViTForMaskedImageModeling, ViTModel, ViTFeatureExtractor

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from dotmap import DotMap as dm

import hands_dataloader
import mnist_dataloader

warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise Exception("No CUDA")

cpu = torch.device("cpu")

def get_training_objects(num_epochs, steps_per_epoch):

    config = dm(
        seed=216,
    )

    dataset_config = dm(

    )

    manual_seed(config.seed)

    dataset = load_dataset("mnist")

    train_dataset = mnist_dataloader.InfiniteMaskedMNIST(dataset["train"])
    test_dataset = mnist_dataloader.dataset_to_pytorch(dataset["test"])

    train_dataloader = DataLoader(train_dataset, batch_size=8)
    test_dataloader = DataLoader(train_dataset, batch_size=8)


    mnist_configuration = ViTConfig(
        num_attention_heads=8,
        num_hidden_layers=4,
        hidden_size=256,
        intermediate_size=512,
        patch_size=1,
        num_channels=3,
        image_size=28,
        encoder_stride=1,
    )
    model = ViTForMaskedImageModeling(mnist_configuration)


    from transformers import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5)

    from ignite.contrib.handlers import PiecewiseLinear

    milestones_values = [
        (0, 5e-5),
        (num_epochs*steps_per_epoch, 0.0),
    ]
    lr_scheduler = PiecewiseLinear(
        optimizer, param_name="lr", milestones_values=milestones_values
    )
    model.to(device)

    def train_step(engine, batch):
        model.train()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss

    def evaluate_step(engine, batch):
        model.eval()

        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        return {'y_pred': predictions, 'y': batch["labels"]}



    trainer = Engine(train_step)

    trainer.add_event_handler(Events.ITERATION_STARTED, lr_scheduler)

    train_evaluator = Engine(evaluate_step)
    validation_evaluator = Engine(evaluate_step)

    from ignite.handlers import ModelCheckpoint

    checkpointer = ModelCheckpoint(dirname='models', filename_prefix='mnist-vit-mlm', n_saved=2, create_dir=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model})

    return dm(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        trainer=trainer,
        train_evaluator=train_evaluator,
        validation_evaluator=validation_evaluator,
        eval_to_image=eval_to_image,
    )

def eval_to_image(model, test_dataset):

    image = test_dataset[torch.randint(low=0, high=len(test_dataset), size=[])]
    image = image.unsqueeze(0)

    image_mask = torch.randint(low=0, high=2, size=[1, 1, 28, 28]).bool()
    idx_mask = image_mask.reshape([1, 784])

    batch = {
        'pixel_values': image,
        'bool_masked_pos': idx_mask,
    }

    model.eval()

    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits.to(cpu)
    predictions = logits.reshape(3, 28, 28).permute(1, 2, 0).byte()
    # print(predictions)

    image_mask1 = image_mask.reshape(28, 28, 1)
    image_mask3 = image_mask.reshape(28, 28).repeat(1, 1)
    image = image.reshape(3, 28, 28).permute(1, 2, 0).byte()
    # print(image.shape)
    out_image = image.masked_scatter(image_mask1, predictions)
    pink = torch.tensor([200, 10, 150], dtype=torch.uint8).reshape(1, 1, 3).repeat(28, 28, 1)
    # print(pink)
    pink_inp_image = pink.masked_scatter(~image_mask1, image[~image_mask3])

    images = [image, pink_inp_image, out_image]
    images = [to_pil_image(img.permute(2, 0, 1)).resize((200, 200), PIL.Image.NEAREST) for img in images]
    return images
