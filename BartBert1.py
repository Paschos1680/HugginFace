from transformers import pipeline, AutoTokenizer, BartForConditionalGeneration, AutoModelForSequenceClassification
import torch
import matplotlib.pyplot as plt

bart_tokenizer = AutoTokenizer.from_pretrained("C:/Users/Michalis/Desktop/ceid/HugginFace/final_bart_paraphrase_model")
bart_model = BartForConditionalGeneration.from_pretrained("C:/Users/Michalis/Desktop/ceid/HugginFace/final_bart_paraphrase_model")
generator = pipeline('text-generation', model=bart_model, tokenizer=bart_tokenizer, framework="pt", num_beams=5, num_return_sequences=5)

bert_tokenizer = AutoTokenizer.from_pretrained("Prompsit/paraphrase-bert-en")
bert_model = AutoModelForSequenceClassification.from_pretrained("Prompsit/paraphrase-bert-en")

def generate_paraphrases(text, num_paraphrases=5):
    paraphrases = generator(text, max_length=60)
    return [para['generated_text'].strip() for para in paraphrases]

def evaluate_paraphrases(paraphrases, original_text):
    scores = []
    for paraphrase in paraphrases:
        inputs = bert_tokenizer.encode_plus(original_text, paraphrase, return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = bert_model(**inputs)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(outputs.logits)
        score = probs[:, 1].item()
        scores.append(score)
    return scores

original_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Lightning never strikes the same place twice.",
    "An apple a day keeps the doctor away.",
    "Where there is love there is life."
]

all_paraphrases = []
all_scores = []

for original_text in original_texts:
    paraphrases = generate_paraphrases(original_text)
    scores = evaluate_paraphrases(paraphrases, original_text)
    all_paraphrases.extend(paraphrases)
    all_scores.extend(scores)

plt.figure(figsize=(10, 5))
plt.barh(all_paraphrases, all_scores, color='skyblue')
plt.xlabel('Paraphrase Quality Score')
plt.title('Paraphrase Evaluation')
plt.gca().invert_yaxis()
plt.show()
