from argparse import ArgumentParser
from pathlib import Path
from newsplease import NewsPlease
import nltk
from nltk.tokenize import sent_tokenize
from torch.utils.data import DataLoader
from data import MyDataset
import torch
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer
from model import SequenceClassificationModel
import os

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)


nltk.download('punkt')

@torch.inference_mode()
def predict(model, dataloader, device):
    model.eval()
    progress_bar = tqdm(dataloader, leave=False)
    progress_bar.set_description(f"Evaluating model")
    preds = []
    for src in progress_bar:
        src = src.to(device)
        preds.append(model(src))
    preds = torch.argmax(torch.vstack(preds).cpu(), dim=1).numpy()
    return preds

def save_html(rundir, title, image_url, main_text, preds):
    head = """<head>
<title>Green and Red Background Color Example</title>
<style>
    .green-background {
        background-color: rgb(200, 255, 200);
    }

    .red-background {
        background-color: rgb(255, 200, 200);
    }

    .yellow-background {
        background-color: rgb(255, 255, 150);
    }
</style>
</head>
""" 
    body = f"""<body>
<h1>{title}</h1>
<img src="{image_url}">
"""
    sentence_id = 0
    for paragraph in main_text:
        current_pred = -1
        body += "<p>\n"
        for sentence in paragraph:
            def add_sentence(body, current_pred, background):
                if current_pred != preds[sentence_id]:
                    body += f'<span class="{background}">'
                    current_pred = preds[sentence_id]
                body += sentence
                if sentence_id + 1 >= len(preds) or current_pred != preds[sentence_id + 1]:
                    body += '</span>'
                return body, current_pred
            if preds[sentence_id] == 0:
                body += sentence
            elif preds[sentence_id] == 1:
                body, current_pred = add_sentence(body, current_pred, "green-background")
            elif preds[sentence_id] == 2:
                body, current_pred = add_sentence(body, current_pred, "yellow-background")
            elif preds[sentence_id] == 3:
                body, current_pred = add_sentence(body, current_pred, "red-background")
            else:
                print(f"Bad prediction on sentence {sentence}, not in the correct range")
            body += ' '
            sentence_id += 1
        if current_pred > 0:
            body += "</span>"
        body += "\n</p>\n"
    body += "</body>"
    html = f"""<!DOCTYPE html>
<html>
{head}
{body}
</html>
"""
    with open(rundir / "index.html", "w") as fout:
        print(html, file=fout)
    

def label_webpage(url, rundir):
    article = NewsPlease.from_url(url)
    title = article.title.replace('‘', "'").replace('’', "'").replace("“", '"').replace("”", '"')
    main_text = article.maintext.replace('‘', "'").replace('’', "'").replace("“", '"').replace("”", '"')
    main_text = list(map(sent_tokenize, main_text.split('\n')))
    image_url = article.image_url
    flattened_main_text = sum(main_text, [])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = MyDataset(flattened_main_text, tokenizer, max_len=512)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=dataset.collate_translation_data
    )
    huggingface_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    backbone_model = huggingface_model.bert
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    huggingface_model.to(device)
    model = SequenceClassificationModel(backbone_model, 4).to(device)
    model.load_state_dict(torch.load("model/checkpoint_best.pth", map_location=device))
    preds = predict(model, dataloader, device)
    print(preds)
    save_html(rundir, title, image_url, main_text, preds)
    os.chdir(rundir)
    os.system('python3 -m http.server')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--url", type=str, help="URL of the page that you want to label"
    )
    parser.add_argument(
        "--rundir", type=Path, help="Directory from which to run a HTTP server"
    )
    args = parser.parse_args()

    label_webpage(args.url, args.rundir)