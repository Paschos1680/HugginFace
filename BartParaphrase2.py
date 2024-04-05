from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Load tokenizer and dataset
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
dataset = load_dataset("glue", "mrpc")

def preprocess_function(examples):
    # Prepare inputs using the tokenizer
    inputs = tokenizer(["paraphrase: " + doc for doc in examples["sentence1"]], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    targets = tokenizer(examples["sentence2"], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs["decoder_input_ids"] = targets.input_ids
    return inputs

# Map the preprocessing function over the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids"])

# Split the dataset into training and validation sets
train_dataset = tokenized_datasets["train"]
val_dataset = tokenized_datasets["validation"]

# Define the model
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prepare DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(3):  # Number of epochs
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=decoder_input_ids)
        
        # Compute loss and perform a backward pass
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Update progress
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
model.save_pretrained("./final_bart_paraphrase_model")
tokenizer.save_pretrained("./final_bart_paraphrase_model")
