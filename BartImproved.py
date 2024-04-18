from transformers import BartForConditionalGeneration, BartTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, DatasetDict

# Function to load and preprocess datasets
def load_and_preprocess_datasets():
    # Load datasets
    paws = load_dataset("paws", "labeled_final", split='train')
    quora = load_dataset("quora", split='train')
    mrpc = load_dataset("glue", "mrpc", split='train').remove_columns(['label', 'idx'])  # Remove columns not used

    # Combine datasets into a single DatasetDict for easier handling
    combined_dataset = concatenate_datasets([paws, quora, mrpc])

    # Initialize tokenizer
    tokenizer = BartTokenizer.from_pretrained("eugenesiow/bart-paraphrase")

    # Define preprocessing function to handle both inputs and labels with padding and truncation
    def preprocess_function(examples):
        # Prepare inputs
        inputs = tokenizer(["paraphrase: " + (doc if doc is not None else "") for doc in examples['sentence1']],
                           max_length=128, padding='max_length', truncation=True)
        # Prepare labels
        labels = tokenizer([doc if doc is not None else "" for doc in examples['sentence2']],
                           max_length=128, padding='max_length', truncation=True)
        
        # Properly format labels to avoid tensor creation issues
        inputs['labels'] = labels['input_ids']
        return inputs

    # Apply preprocessing and convert to a format suitable for training
    tokenized_datasets = combined_dataset.map(preprocess_function, batched=True, remove_columns=combined_dataset.column_names)
    return tokenized_datasets, tokenizer

tokenized_datasets, tokenizer = load_and_preprocess_datasets()

# Load the BART model
model = BartForConditionalGeneration.from_pretrained("eugenesiow/bart-paraphrase")

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    logging_dir='./logs'
)

# Data collator for dynamic padding
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Custom trainer class to handle loss calculation
class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs["labels"]
        loss_fct = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# Initialize the trainer with the custom loss function
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./final_bart_paraphrase_model")
tokenizer.save_pretrained("./final_bart_paraphrase_model")
