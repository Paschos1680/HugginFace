import numpy as np
from datasets import load_dataset
import torch  # Ensure torch is imported
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset - using a small slice for quick evaluation
dataset = load_dataset("quora", split='train[:1%]')

# Define your model paths or Hugging Face model identifiers
model_paths = {
    "Custom Fine-Tuned BERT": r"c:\Users\Michalis\Desktop\ceid\HugginFace\final_bart_paraphrase_model",
    "Custom Fine-Tuned BART": r"c:\Users\Michalis\Desktop\ceid\HugginFace\results\checkpoint-1000",
    "sentence-transformers/paraphrase-xlm-r-multilingual-v1": None,
    "sentence-transformers/paraphrase-mpnet-base-v2": None,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": None,
}

# Ensure you're initializing RougeScorer correctly
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeLsum'], use_stemmer=True)

# Function to compute metrics
def compute_metrics(prediction, reference):
    rouge_scores = rouge_scorer.score(reference, prediction)
    bleu_score = sentence_bleu([reference.split()], prediction.split(), smoothing_function=SmoothingFunction().method1)
    return rouge_scores['rougeLsum'].fmeasure, bleu_score

# Evaluation loop
results = {}
for model_name, model_path in model_paths.items():
    if model_path is None:  # Model loaded from Hugging Face
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Assuming seq2seq for simplicity
    else:  # Model loaded from a local path
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    
    rouge_scores, bleu_scores = [], []
    for example in dataset:
        inputs = tokenizer(example['questions']['text'][0], return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], max_length=512, num_beams=5)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        reference = example['questions']['text'][1]
        rouge_score, bleu_score = compute_metrics(prediction, reference)
        rouge_scores.append(rouge_score)
        bleu_scores.append(bleu_score)

    # Aggregate and store results
    results[model_name] = {
        "ROUGE": np.mean(rouge_scores),
        "BLEU": np.mean(bleu_scores)
    }

# Visualization of results
def plot_results(metric):
    labels = list(results.keys())
    scores = [result[metric] for result in results.values()]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=labels, y=scores, palette="viridis")
    plt.title(f'{metric} Scores Across Models')
    plt.ylabel('Score')
    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.show()

plot_results("ROUGE")
plot_results("BLEU")




