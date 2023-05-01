from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import trange
from tqdm import tqdm

from data import TranslationDataset, SpecialTokens
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import os

from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def train_epoch(model, train_dataloader, criterion, optimizer, device):
    # train the model for one epoch
    # you can obviously add new arguments or change the API if it does not suit you
    model.train()
    losses = 0
    progress_bar = tqdm(train_dataloader, leave=False)
    progress_bar.set_description("Train model")
    for idx, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)
        # bug-prone part starts here
        preds = model(src).logits
        loss = criterion(preds, tgt)
        # bug-prone part ends here
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses += loss.item()
        # scheduler.step()
        if idx % 10 == 0:
            progress_bar.set_postfix(loss=loss.item())
    return losses / len(train_dataloader)


@torch.inference_mode()
def evaluate(model, eval_dataloader, criterion, device):
    # compute the loss over the entire validation subset
    model.eval()
    losses = 0
    progress_bar = tqdm(eval_dataloader, leave=False)
    progress_bar.set_description("Evaluate model")
    correct = 0
    for idx, (src, tgt) in enumerate(progress_bar):
        src, tgt = src.to(device), tgt.to(device)
        # bug-prone part starts here
        preds = model(src).logits
        loss = criterion(preds, tgt)
        # bug-prone part ends here
        losses += loss.item()
        correct += torch.count_nonzero(preds == tgt)
        if idx % 10 == 0:
            progress_bar.set_postfix(loss=loss.item())
    return losses / len(eval_dataloader), correct / len(eval_dataloader)

def train_model(data_dir, num_epochs):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=4)
    model.to(device)

    if num_epochs == 0:
        model.load_state_dict(torch.load("checkpoint_best.pth", map_location=device))
        return model

    train_dataset = TranslationDataset(
        str(data_dir) + "/train.csv",
        tokenizer,
        max_len=128,  # might be enough at first
    )
    val_dataset = TranslationDataset(
        str(data_dir) + "/val.csv",
        tokenizer,
        max_len=128,
    )
    test_dataset = TranslationDataset(
        str(data_dir) + "/test.csv",
        tokenizer,
        max_len=128,
    )

    # create loss, optimizer, scheduler objects, dataloaders etc.
    # don't forget about collate_fn
    # if you intend to use AMP, you might need something else

    # for p in model.parameters():
    #     if p.dim() > 1:
    #         nn.init.xavier_uniform_(p)

    BATCH_SIZE = 16
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_dataset.collate_translation_data)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=val_dataset.collate_translation_data)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=val_dataset.collate_translation_data)

    criterion = nn.CrossEntropyLoss()

    STARTING_LEARNING_RATE = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=STARTING_LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=total_steps)

    min_val_loss = float("inf")

    wandb.init(project="Sujectivity detection in news articles")

    for epoch in trange(1, num_epochs + 1):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)

        wandb.log({
            "Train CrossEntropyLoss": train_loss,
            "Val Accuracy": val_accuracy,
            "Val CrossEntropyLoss":val_loss,
            "Learning rate": optimizer.param_groups[0]['lr'],
        })

        # might be useful to translate some sentences from validation to check your decoding implementation

        # also, save the best checkpoint somewhere around here
        if val_loss < min_val_loss:
            print(f"New best loss {val_loss} on epoch {epoch}! Saving checkpoint")
            torch.save(model.state_dict(), "checkpoint_best.pth")
            min_val_loss = val_loss

        # and the last one in case you need to recover
        # by the way, is this sufficient?
        torch.save(model.state_dict(), "checkpoint_last.pth")

    # load the best checkpoint
    model.load_state_dict(torch.load("checkpoint_best.pth", map_location=device))
    _, test_accuracy = evaluate(model, test_dataloader, criterion, device)
    print(f"Test accuracy on the best model: {test_accuracy}")

    return model

if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--data-dir", type=Path, help="Path to the directory containing processed data"
    )

    # argument groups are useful for separating semantically different parameters
    hparams_group = parser.add_argument_group("Training hyperparameters")
    hparams_group.add_argument(
        "--num-epochs", type=int, default=50, help="Number of training epochs"
    )

    args = parser.parse_args()

    train_model(args.data_dir, args.num_epochs)