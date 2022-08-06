import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast

import json
import pandas as pd
import plotly.express as px

from tqdm import tqdm


# Torch dataset to call BERT later
class FakeNewsDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        tokens = tokenizer(X, max_length=512, truncation=True, padding=True, return_tensors='pt')
        self.input_ids = tokens.input_ids
        self.attention_masks = tokens.attention_mask
        self.labels = y
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]


if __name__ == '__main__':

    # We don't need NaNs in train titles
    data = pd.read_csv('./fake_news/train.csv').dropna(subset=['title'])

    BERT_MODEL = "bert-base-multilingual-uncased"
    BATCH_SIZE = 8
    VALID_FRACTION = 0.1
    N_EPOCHS = 3
    LEARNING_RATE = 1e-5

    device = 'cuda'
    device_ids = [0, 1]  # Available cuda devices for multiprocessing
    # Evening batch size per every node
    BATCH_SIZE = BATCH_SIZE * len(device_ids)

    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=2)
    
    # Multiprocessing wrapper
    model = nn.DataParallel(model, device_ids=[0, 1]).to(device)

    dataset = FakeNewsDataset(data.title.tolist(), data.label.tolist(), tokenizer)

    # Splits to spot overfitting easier
    train_dataset, valid_dataset = random_split(
        dataset,
        [len(dataset) - int(len(dataset) * VALID_FRACTION),
         int(len(dataset) * VALID_FRACTION)]
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Criterion is already embedded by transformers, so just optimizer here
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(N_EPOCHS):

        # Train loop
        pbar = tqdm(train_dataloader)
        model.train()
        total_loss = 0
        for i, (indices, att_masks, labels) in enumerate(pbar, start=1):
            indices, att_masks, labels = indices.to(device), att_masks.to(device), labels.to(device)
            optimizer.zero_grad()

            out = model(indices, attention_mask=att_masks, labels=labels)
            # Average loss accross devices
            loss = out.loss.mean()

            total_loss += loss.item()
            pbar.set_description(f'{epoch}/{N_EPOCHS}: Average train Cross-Entropy: {total_loss / i:.4f}', refresh=True)

            # Backprop
            loss.backward()
            optimizer.step()

        # Valid loop
        pbar = tqdm(valid_dataloader)
        model.eval()
        total_loss = 0
        for i, (indices, att_masks, labels) in enumerate(pbar, start=1): 
            indices, att_masks, labels = indices.to(device), att_masks.to(device), labels.to(device)
            out = model(indices, attention_mask=att_masks, labels=labels)
            # Average loss accross devices        
            total_loss += out.loss.mean().item()
            pbar.set_description(f'{epoch}/{N_EPOCHS}: Average valid Cross-Entropy: {total_loss / i:.4f}', refresh=True)
        
        # Checkpointing just in case
        torch.save(model.state_dict(), f'./checkpoints/{epoch}_model.pt')
    
    # Filling NaNs with blank spaces in hope BERT will understand (probably better do it rule-based)
    test_X = pd.read_csv('./fake_news/test.csv').title.fillna(' ').tolist()
    test_y = pd.read_csv('./fake_news/labels.csv').label.tolist()
    
    # Testing datasets
    test_dataset = FakeNewsDataset(test_X, test_y, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # We will iterate over batches and add predictions and labels here
    test_y = torch.Tensor()
    test_pred = torch.Tensor()

    pbar = tqdm(test_dataloader)
    model.eval()
    total_loss = 0
    for i, (indices, att_masks, labels) in enumerate(pbar, start=1): 
        indices, att_masks, labels = indices.to(device), att_masks.to(device), labels.to(device)
        out = model(indices, attention_mask=att_masks, labels=labels)
        
        # Just to get a sense of what was going on
        total_loss += out.loss.mean().item()
        pbar.set_description(f'Average test Cross-Entropy: {total_loss / i:.4f}', refresh=True)

        # Adding labels and predictions
        test_y = torch.cat([test_y, labels.cpu()])
        test_pred = torch.cat([test_pred, torch.argmax(out.logits, dim=1).cpu()])

    
    # Calculating metrics, finally
    test_y = test_y.numpy()
    test_pred = test_pred.numpy()

    print('Test Accuracy:', accuracy_score(test_y, test_pred))
    print('Test f1 score:', f1_score(test_y, test_pred))
    conf_matrix = confusion_matrix(test_y, test_pred) / len(test_y)

    # Show the confusion matrix
    px.imshow(conf_matrix, text_auto=True)

    # Saving results to access them without CLI
    results = {
        'test_accuracy': accuracy_score(test_y, test_pred),
        'test_f1': f1_score(test_y, test_pred),
        'test_confusion_matrix': conf_matrix.tolist()
    }

    with open('results_bert_title.json', 'w') as f:
        json.dump(results, f)
