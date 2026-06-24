---
title: Pro Football AI System
emoji: ⚽
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---


⚽ Pro Football AI System

A machine learning football prediction engine that combines Elo rating dynamics, team form analysis, and probabilistic ML modeling to predict match outcomes.

🧠 What this system does

This app predicts:

Home Win probability
Draw probability
Away Win probability
Live team strength (Elo ratings)

It updates team strength dynamically as predictions are made, simulating real-world football rating systems.

⚙️ Core Features
📊 Hybrid AI Model

Combines:

Machine Learning model (scikit-learn)
Elo rating system (dynamic team strength)
Form-based statistical features

⚽ Team Intelligence

Each team is evaluated using:

Last 10 match performance
Goals scored and conceded
Match results (win/draw/loss)
Momentum trends

🧠 Dynamic Elo System
Every prediction updates team ratings
Strong teams become stronger over time
Weak teams lose rating dynamically
Simulates real football ranking systems
🌐 Live Web App

Built with Gradio and deployed on Hugging Face Spaces.

🏗️ Project Structure
SoccerMatchPredictor/
│
├── app.py                # Main Gradio application
├── features.py          # Team form feature engine
├── elo_engine.py        # Dynamic Elo system
├── model.pkl            # Trained ML model
├── matches.csv          # Historical match data
├── requirements.txt     # Dependencies
└── README.md            # Project documentation

🚀 How it works
Select Home and Away teams
System extracts:
Form stats
Elo ratings
ML model predicts probabilities
Elo system updates team strength
Results returned as percentages

📦 Installation (Local)
pip install -r requirements.txt

▶️ Run locally
python app.py

🌐 Deploy on Hugging Face
Push repo to GitHub
Create Hugging Face Space
Select Gradio
Link repository
Deploy automatically

📊 Example Output
Home Win: 62.4%
Draw: 21.3%
Away Win: 16.3%

Home Elo: 1582.4
Away Elo: 1491.2

🧠 Tech Stack
Python
Pandas
Scikit-learn
Joblib
Gradio