# 🚢 Titanic Survival Prediction

This project is a **Streamlit web application** that predicts whether a Titanic passenger would survive based on personal information.

Users input age, gender, fare, class, and more — and the model estimates the probability of survival.

## 🧠 Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn** – Logistic Regression model
- **Joblib** – For saving/loading the model
- **Streamlit** – Web UI
- **Matplotlib & Seaborn** – For data visualization (EDA)

---

## 📦 Installation

To run this project locally:

```bash
# Clone the repository
git clone https://github.com/iceweasel13/titanic.git
cd titanic

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running the App

```bash
streamlit run app.py
```

The app will open in your browser automatically.

---

## 💡 Feature Engineering

The following features were added:

- `Title`: Extracted from passenger name, grouped rare titles under "Rare"
- `FamilySize`: SibSp + Parch + 1
- `IsAlone`: Binary feature if passenger is traveling alone
- `Fare`: Log-transformed using `log1p`
- `One-Hot Encoding`: Applied to categorical features (`Sex`, `Embarked`, `Pclass`, `Title`)

---

## 📊 Model Training

- Model: `LogisticRegression`
- Used inside a `Pipeline` with `StandardScaler`
- Saved as `model.pkl` after training

---

## 🌐 Live Demo

Try it online:

👉(https://iceweasel13-titanic-app-yfp76v.streamlit.app/)


---

## 📁 Project Structure

```
titanic/
├── app.py                # Streamlit app
├── model.pkl             # Trained ML model
├── requirements.txt      # Dependencies
├── README.md             # This file
└── data/
    └── train.csv         # Titanic dataset
```

---

## 📜 License

This project is intended for educational purposes only and not for commercial use.

---

## 🙋‍♂️ Contributing

Pull requests are welcome. Feel free to fork the repo and improve it!
