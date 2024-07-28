from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Load BERT tokenizer and model
bert_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"  
bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)

# Load BiLSTM model and pipeline
bilstm_model_path = "avichr/heBERT_sentiment_analysis"  
bilstm_pipeline = pipeline(
    "sentiment-analysis",
    model=bilstm_model_path,
    tokenizer=bilstm_model_path,
    return_all_scores=True
)

def analyze_sentiment(text, model_type):
    if model_type == 'bert':
        encoded_input = bert_tokenizer(text, return_tensors='pt')
        output = bert_model(**encoded_input)
        scores = output.logits.detach().numpy()[0]
        scores = softmax(scores)
        labels = ['negative', 'neutral', 'positive']
        scores_dict = {labels[i]: float(scores[i]) for i in range(len(labels))}
        return scores_dict
    
    elif model_type == 'bilstm':
        result = bilstm_pipeline(text)[0]
        scores_dict = {item['label'].lower(): item['score'] for item in result}
        return scores_dict

def create_pie_chart(scores):
    labels = list(scores.keys())
    values = list(scores.values())
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        textinfo='label+percent',
        insidetextorientation='horizontal',
        marker=dict(
            colors=['#F22587', '#FFD359', '#0AA836'],  # Custom colors for slices
            line=dict(color='#FFFFFF', width=2)  # Black border for slices
        ),
        textfont=dict(size=18, color='white')  # Increased font size and color inside slices
    )])
    fig.update_layout(
        title_text='Sentiment Distribution',  # Title of the chart
        title_font_size=20,  # Increased title font size
        autosize=False,
        width=600,  # Increased width of the chart
        height=500,  # Increased height of the chart
        margin=dict(l=60, r=20, t=60, b=20)  # Adjusted margins around the chart
    )
    chart_html = pio.to_html(fig, full_html=False)
    return chart_html

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    model_type = request.form['model_type']
    text = request.form['text']
    
    result = analyze_sentiment(text, model_type)
    pie_chart_html = create_pie_chart(result)
    
    return render_template('index.html', text=text, scores=result, pie_chart_html=pie_chart_html, model_type=model_type)

if __name__ == '__main__':
    app.run(debug=True)
