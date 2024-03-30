from flask import Flask, request, jsonify
from transformers import BertTokenizerFast, EncoderDecoderModel
import torch

app = Flask(__name__)

# Load tokenizer and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizerFast.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization')
model = EncoderDecoderModel.from_pretrained('mrm8488/bert-small2bert-small-finetuned-cnn_daily_mail-summarization').to(device)

# Function to generate summary with minimum length constraint
def generate_summary_with_min_length(text, min_length=100):
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    # Set minimum length constraint
    min_length = max(min_length, 0)
    if min_length > 0:
        output = model.generate(input_ids, attention_mask=attention_mask, min_length=min_length)
    else:
        output = model.generate(input_ids, attention_mask=attention_mask)

    return tokenizer.decode(output[0], skip_special_tokens=True)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data['text']
    min_length = data.get('min_length', 100)

    summary = generate_summary_with_min_length(text, min_length=min_length)
    
    return jsonify({'summary': summary})

if __name__ == '__main__':
    app.run(debug=True)
