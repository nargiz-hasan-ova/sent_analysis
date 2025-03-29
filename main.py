# main.py

from flask import Flask, render_template, request
from app_utils import load_model_and_labels, get_prediction

app = Flask(__name__)

# Load the models once at app startup
sentiment_tokenizer, sentiment_model, sentiment_labels = load_model_and_labels("sentiment")
emotion_tokenizer, emotion_model, emotion_labels = load_model_and_labels("emotion")
hate_tokenizer, hate_model, hate_labels = load_model_and_labels("hate")

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment_result = None
    emotion_result = None
    hate_result = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form.get("user_text", "").strip()
        if user_text:
            # 1) Sentiment
            sentiment_result = get_prediction(sentiment_tokenizer, sentiment_model, sentiment_labels, user_text)
            # 2) Emotion
            emotion_result   = get_prediction(emotion_tokenizer, emotion_model, emotion_labels, user_text)
            # 3) Hate
            hate_result      = get_prediction(hate_tokenizer, hate_model, hate_labels, user_text)

    return render_template(
        "index.html",
        user_text=user_text,
        sentiment_result=sentiment_result,
        emotion_result=emotion_result,
        hate_result=hate_result
    )


if __name__ == "__main__":
    # Run in debug mode for development
    app.run(debug=True)
