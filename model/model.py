import torch.nn as nn

class SequenceClassificationModel(nn.Module):
    def __init__(self, backbone_bertmodel, num_labels):
        super().__init__()
        self.bert = backbone_bertmodel
        self.head = nn.Sequential(nn.Dropout(p=0.1), nn.Linear(in_features=768, out_features=num_labels))

    def forward(self, input_ids):
        pooled_output = self.bert(input_ids).pooler_output
        return self.head(pooled_output)
