from transformers import BertTokenizer, BertForSequenceClassification

def load_finetuned_model():
    model = BertForSequenceClassification.from_pretrained("kkkkkjjjjjj/results")
    tokenizer = BertTokenizer.from_pretrained("kkkkkjjjjjj/results")
    return tokenizer, model
