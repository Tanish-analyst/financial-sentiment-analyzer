from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_finetuned_model():
    model = AutoModelForSequenceClassification.from_pretrained("kkkkkjjjjjj/results")
    tokenizer = AutoTokenizer.from_pretrained("kkkkkjjjjjj/results")
    return tokenizer, model
