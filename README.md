# Sentiment-Analysis-Model
Multi-Task NLP Web App
A simple web application that uses CardiffNLP’s Twitter-roBERTa models to perform sentiment analysis, emotion detection, and hate speech detection on user-input text. The project runs on Flask and displays model predictions in a clean, Bootstrap-styled interface.

Features
Sentiment Analysis: Classifies text into negative, neutral, or positive sentiment.

Emotion Detection: Identifies emotions such as anger, joy, optimism, and sadness.

Hate Speech Detection: Flags whether a message is hateful or not, based on TweetEval’s hate model.

Real-Time Predictions: Simply enter a text comment or tweet, and see all three results instantly.

Bootstrap UI: A responsive, polished front-end perfect for demos or portfolio showcases.

How It Works
Models: Powered by Hugging Face Transformers (specifically cardiffnlp/twitter-roberta-base-* variants).

Text Preprocessing: Tweets are normalized by replacing @user mentions and URLs with placeholders.

Flask Backend:

Loads three PyTorch models at startup (sentiment, emotion, hate).

Receives input text via a POST form.

Outputs label probabilities (sorted by descending confidence).

Front-End: Renders predictions using Bootstrap cards, lists, or progress bars.

Project Structure
graphql
Копировать
Редактировать
sentiment_emotion_app/
├── app_utils.py         # Helper functions for model loading and prediction
├── main.py              # Flask app entry point
├── requirements.txt     # Dependencies
├── templates/
│   └── index.html       # Main HTML page (Bootstrap form + results)
└── venv/                # Virtual environment (excluded via .gitignore)
Getting Started
Prerequisites
Python 3.7+

(Optional) virtualenv or another environment manager

Installation
Clone this repository:

bash
Копировать
Редактировать
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
Create a virtual environment (recommended):

bash
Копировать
Редактировать
python3 -m venv venv
source venv/bin/activate       # On macOS/Linux
# or: venv\Scripts\activate    # On Windows
Install dependencies:

bash
Копировать
Редактировать
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Running the App
Activate your virtual environment (if not already):

bash
Копировать
Редактировать
source venv/bin/activate
Start the Flask server:

bash
Копировать
Редактировать
python main.py
Open your browser at http://127.0.0.1:5000/.

Enter a text (e.g., “I love NLP!”) and click Analyze. You’ll see results for sentiment, emotion, and hate classification.

Example Usage
Input:

arduino
Копировать
Редактировать
"I hate rainy days."
Output (example probabilities):

yaml
Копировать
Редактировать
Sentiment: 
  - negative: 0.92
  - neutral:  0.07
  - positive: 0.01

Emotion: 
  - anger: 0.88
  - sadness: 0.11
  - joy: 0.01
  - optimism: 0.00

Hate Speech: 
  - not-hate: 0.98
  - hate: 0.02
Customization
Model Choice: Swap out the CardiffNLP model names (e.g., 'cardiffnlp/twitter-roberta-base-sentiment') with any other Hugging Face model.

UI/Styling: Modify index.html and integrate more Bootstrap components or custom CSS.

Deployment: Containerize with Docker, or deploy to Heroku/AWS/GCP if you want a production URL.

Contributing
Fork the repository.

Create a new branch.

Commit and push your changes.

Open a pull request.

All contributions are welcome!

License
This project is open-sourced under the MIT License.

