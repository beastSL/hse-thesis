import torch.nn as nn
import torch
from data import MyDataset
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
import wandb
from model import SequenceClassificationModel

class Trainer:
    def __init__(
        self,
        device,
        tokenizer,
        dataset_path,
        batch_size,
        max_len,
        backbone_model,
        num_labels,
        task_type,
        optimizer_params,
        one_cycle_policy=False,
        num_shared_epochs=None,
        scheduler_params=None
    ):
        self.device = device
        
        self.data_dir_for_debug = dataset_path
        self.train_dataset = MyDataset(dataset_path + "/train.csv", tokenizer, max_len)
        self.val_dataset = MyDataset(dataset_path + "/val.csv", tokenizer, max_len)
        self.test_dataset = MyDataset(dataset_path + "/test.csv", tokenizer, max_len)
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_translation_data
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size,
            collate_fn=self.val_dataset.collate_translation_data
        )
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size,
            collate_fn=self.test_dataset.collate_translation_data
        )

        self.model = SequenceClassificationModel(backbone_model, num_labels).to(device)
        self.task_type = task_type
        self.criterion = nn.CrossEntropyLoss() if task_type == "classification" else nn.MSELoss()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        if not one_cycle_policy:
            self.scheduler = None
        else:
            total_steps = num_shared_epochs * len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                total_steps=total_steps,
                **scheduler_params
            )
        print(f"Initialized trainer for {dataset_path}")

    def train_epoch(self):
        self.model.train()
        losses = 0
        progress_bar = tqdm(self.train_dataloader, leave=False)
        progress_bar.set_description(f"Train model on {self.data_dir_for_debug}")
        for idx, (src, tgt) in enumerate(progress_bar):
            if self.task_type == "classification":
                tgt = tgt.type(torch.LongTensor)
            src, tgt = src.to(self.device), tgt.to(self.device)
            preds = self.model(src)
            loss = self.criterion(preds, tgt)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses += loss.item()
            if self.scheduler is not None:
                self.scheduler.step()
            if idx % 10 == 0:
                progress_bar.set_postfix(loss=loss.item())
        return losses / len(self.train_dataloader)

    @torch.inference_mode()
    def evaluate(self, val_or_test="val"):
        if val_or_test == "val":
            eval_dataloader = self.val_dataloader
            eval_dataset = self.val_dataset
        elif val_or_test == "test":
            eval_dataloader = self.test_dataloader
            eval_dataset = self.test_dataset
        else:
            raise "You should pass 'val' or 'test' into val_or_test"
        self.model.eval()
        losses = 0
        progress_bar = tqdm(eval_dataloader, leave=False)
        progress_bar.set_description(f"Evaluate model on {self.data_dir_for_debug}")
        correct = 0
        for idx, (src, tgt) in enumerate(progress_bar):
            if self.task_type == "classification":
                tgt = tgt.type(torch.LongTensor)
            src, tgt = src.to(self.device), tgt.to(self.device)
            preds = self.model(src)
            loss = self.criterion(preds, tgt)
            losses += loss.item()
            correct += torch.count_nonzero((torch.argmax(preds, dim=1) == tgt).type(torch.LongTensor))
            if idx % 10 == 0:
                progress_bar.set_postfix(loss=loss.item())
        return losses / len(eval_dataloader), correct / len(eval_dataset)

    def update_optimizer(self, optimizer_params, one_cycle_policy=False, num_epochs=None, scheduler_params=None):
        self.optimizer = torch.optim.Adam(self.model.parameters(), **optimizer_params)
        if one_cycle_policy:
            total_steps = num_epochs * len(self.train_dataloader)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                total_steps=total_steps,
                **scheduler_params
            )
        else:
            self.scheduler = None


class MultiTaskTrainer:
    def __init__(
        self,
        parsed_config,
        optimizer_params,
        batch_size,
        max_len,
        shared_one_cycle_policy=False,
        num_shared_epochs=5,
        scheduler_params=None
    ):
        super().__init__()

        torch.manual_seed(0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        huggingface_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        backbone_model = huggingface_model.bert
        huggingface_model.to(device)

        def init_trainer(config_entry):
            return Trainer(
                device,
                tokenizer,
                config_entry["data-dir"],
                batch_size,
                max_len,
                backbone_model,
                config_entry["num-labels"],
                config_entry["type"],
                optimizer_params,
                one_cycle_policy=shared_one_cycle_policy,
                num_shared_epochs=num_shared_epochs,
                scheduler_params=scheduler_params
            )

        self.num_shared_epochs = num_shared_epochs
        self.optimizer_params_for_update = optimizer_params
        self.main_trainer = init_trainer(parsed_config['main-task'])
        self.additional_trainers = [
            init_trainer(additional_config_entry) for additional_config_entry in parsed_config['additional-tasks']
        ]
        
    def train(
        self,
        num_epochs_main,
        pretrained_mtl=None,
        one_cycle_policy=False,
        scheduler_params=None
    ):  
        if self.num_shared_epochs > 0:
            wandb.init(project="Subjectivity-detection-in-news-articles")
            for epoch in trange(1, self.num_shared_epochs + 1):
                for additional_trainer in self.additional_trainers:
                    train_loss = additional_trainer.train_epoch()
                    wandb.log({
                        f"{additional_trainer.data_dir_for_debug}/Train CrossEntropyLoss": train_loss,
                        f"{additional_trainer.data_dir_for_debug}/Learning rate": 
                            additional_trainer.optimizer.param_groups[0]['lr']
                    })
                train_loss = self.main_trainer.train_epoch()
                val_loss, val_accuracy = self.main_trainer.evaluate("val")
                wandb.log({
                    f"{self.main_trainer.data_dir_for_debug}/Train CrossEntropyLoss": train_loss,
                    f"{self.main_trainer.data_dir_for_debug}/Val Accuracy": val_accuracy,
                    f"{self.main_trainer.data_dir_for_debug}/Val CrossEntropyLoss":val_loss,
                    f"{self.main_trainer.data_dir_for_debug}/Learning rate": 
                    self.main_trainer.optimizer.param_groups[0]['lr']}
                )
                torch.save(self.main_trainer.model.state_dict(), f"pretrain_epoch_{epoch}.pth")
            wandb.finish()

        wandb.init(project="Subjectivity-detection-in-news-articles")
        if pretrained_mtl is not None:
            self.main_trainer.model.load_state_dict(torch.load(pretrained_mtl, map_location=self.main_trainer.device))
            val_loss, val_accuracy = self.main_trainer.evaluate("val")
            wandb.log({
                "Val Accuracy": val_accuracy,
                "Val CrossEntropyLoss":val_loss,
                "Learning rate": self.main_trainer.optimizer.param_groups[0]['lr']
            })
            torch.save(self.main_trainer.model.state_dict(), "checkpoint_best.pth")
            min_val_loss = val_loss
        else:
            min_val_loss = float("inf")
        if one_cycle_policy:
            self.main_trainer.update_optimizer(
                self.optimizer_params_for_update,
                one_cycle_policy=True,
                num_epochs=num_epochs_main,
                scheduler_params=scheduler_params
            )
        for epoch in trange(1, num_epochs_main + 1):
            train_loss = self.main_trainer.train_epoch()
            val_loss, val_accuracy = self.main_trainer.evaluate("val")
            wandb.log({
                "Train CrossEntropyLoss": train_loss,
                "Val Accuracy": val_accuracy,
                "Val CrossEntropyLoss":val_loss,
                "Learning rate": self.main_trainer.optimizer.param_groups[0]['lr']
            })
            if val_loss < min_val_loss:
                print(f"New best loss {val_loss} on epoch {epoch}! Saving checkpoint")
                torch.save(self.main_trainer.model.state_dict(), "checkpoint_best.pth")
                min_val_loss = val_loss

            # and the last one in case you need to recover
            # by the way, is this sufficient?
            torch.save(self.main_trainer.model.state_dict(), "checkpoint_last.pth")

        # load the best checkpoint
        self.main_trainer.model.load_state_dict(torch.load("checkpoint_best.pth", map_location=self.main_trainer.device))
        _, test_accuracy = self.main_trainer.evaluate("test")
        print(f"Test accuracy on the best model: {test_accuracy}")

        return self.main_trainer.model
