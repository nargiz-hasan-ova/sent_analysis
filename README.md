# Sentiment-Analysis-Model

# Multi-Task NLP Web App

A lightweight Flask application demonstrating **sentiment analysis**, **emotion detection**, and **hate speech detection** using **CardiffNLP’s Twitter-roBERTa** models. Simply enter text in the web interface, and the app returns label probabilities (e.g., negative/positive sentiment, anger/joy emotion, and hate/not-hate classification). This project highlights practical **NLP** deployment via **Hugging Face Transformers**, **PyTorch**, and a **Bootstrap**-styled frontend—ideal for showcasing data science and web development skills.



## Features
1. **Sentiment Analysis**: Classifies text into **negative**, **neutral**, or **positive** sentiment.  
2. **Emotion Detection**: Identifies emotions such as **anger**, **joy**, **optimism**, and **sadness**.  
3. **Hate Speech Detection**: Flags whether a message is hateful or not, based on TweetEval’s hate model.  
4. **Real-Time Predictions**: Simply enter a text comment or tweet, and see all three results instantly.  
5. **Bootstrap UI**: A responsive, polished front-end perfect for demos or portfolio showcases.

## How It Works
- **Models**: Powered by **Hugging Face Transformers** (specifically `cardiffnlp/twitter-roberta-base-*` variants).
- **Text Preprocessing**: Tweets are normalized by replacing `@user` mentions and URLs with placeholders.
- **Flask Backend**:
  1. Loads three **PyTorch** models at startup (sentiment, emotion, hate).
  2. Receives input text via a POST form.
  3. Outputs label probabilities (sorted by descending confidence).
- **Front-End**: Renders predictions using **Bootstrap** cards, lists, or progress bars.


│ └── index.html # Main HTML page (Bootstrap form + results) 
└── venv/ # Virtual environment (excluded via .gitignore)

