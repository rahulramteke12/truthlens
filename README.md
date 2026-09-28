# TruthLens AI — Advanced Fake News & Misinformation Detector

> A production-grade NLP pipeline and interactive intelligence studio designed to verify news authenticity in real-time, achieving **96.7% test accuracy** through a de-biased soft-voting ensemble.

---

## 🔍 About TruthLens AI
Misinformation spreads 6x faster than truth. TruthLens AI provides an explainable engineering solution to that problem.
It decouples the model training pipeline from the inference engine, offering real-time predictions, explainable token inspection, and credibility diagnostics via a Flask web application and REST API.

### 🌟 Key Features
* **Dual-Engine Soft-Voting Ensemble:** Combines Logistic Regression and Multinomial Naive Bayes for high-generalization classification.
* **De-Biased Training:** Removed publisher source watermarks (eliminating the classic "Reuters" data leakage bug) so genuine real news is not falsely flagged as fake.
* **Empirical 96.7% Accuracy:** Balanced evaluation with 97% precision and 97% recall across true and fake news.
* **Explainable AI Inspector:** Real-time token highlighting distinguishing verified journalistic citations from sensational clickbait triggers.
* **Modern Interactive Studio:** Includes live text telemetry, confidence gauge, diagnostic scorecards, test presets, and history storage.
* **Model Persistence:** Integrated `joblib`/`pickle` for serializing the model and vectorizer, eliminating the need to retrain for every request.

---

##  Tech Stack
* **Language:** Python 3.x
* **Backend:** Flask (Web Framework)
* **Machine Learning:** Scikit-Learn (Logistic Regression, Decision Tree)
* **Data Processing:** Pandas, NumPy, Regular Expressions (Re)
* **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **Version Control:** Git

---

## System Architecture
The project follows a clear separation of concerns:

1.  **`train.py` (The Pipeline):** * Ingests raw data (`Fake.csv`, `True.csv`).
    * Cleans and normalizes text using the `wordopt` function.
    * Trains the Logistic Regression model.
    * Serializes (saves) the trained model (`model.pkl`) and vectorizer (`vectorizer.pkl`) to disk.

2.  **`app.py` (The Application):** * Initializes a Flask web server.
    * Loads the pre-trained "brain" (`.pkl` files) into memory.
    * Exposes a `/predict` endpoint to handle user inputs and return classification results.

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/SaravanaKumar056/fake_news_detect.git](https://github.com/SaravanaKumar056/fake_news_detect.git)
cd fake_news_detect


2. Install Dependencies
Bash

pip install -r requirements.txt
3. Train the Model (Optional)
Note: The repo comes with a pre-trained model, but you can retrain it using the dataset.

Bash

python train.py
# Output: "Success! You can now run app.py."
4. Start the Application
Bash

python app.py
# Output: * Running on [http://127.0.0.1:5000](http://127.0.0.1:5000)
5. Verify
Open your browser and go to http://127.0.0.1:5000. Enter a news headline to see the prediction!

 Performance Metrics
During the training phase, the model was evaluated on a split test set:

Logistic Regression: 99% Accuracy

Decision Tree: 99% Accuracy

Gradient Boosting: 99% Accuracy

Random Forest: 99% Accuracy

The production app currently uses Logistic Regression for its balance of speed and accuracy.

Future Improvements
Dockerization: Containerize the application for cloud deployment (AWS/Azure).

Deep Learning: Implement an LSTM or BERT-based model for better context understanding.

CI/CD: Set up GitHub Actions for automated testing and deployment.
