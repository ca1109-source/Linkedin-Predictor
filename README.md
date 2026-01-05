# The Profile Prophet 🔮

A mystical LinkedIn usage prediction web application that uses machine learning to predict whether someone is likely to be a LinkedIn user based on their demographic profile.

## 🎯 Project Overview

This interactive Streamlit application predicts LinkedIn usage probability using logistic regression. The app features a mystical, oracle-themed UI that makes data science predictions engaging and fun.

## ✨ Features

- **Interactive Prediction Interface**: User-friendly form to input demographic information
- **Real-time Probability Calculation**: Instant LinkedIn usage predictions
- **Comprehensive Visualizations**: Six interactive charts showing how each feature impacts predictions
- **Mystical Theme**: Engaging crystal ball interface with custom CSS styling

## 🛠️ Technologies Used

- **Python 3.x**
- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning (Logistic Regression)
- **Pandas & NumPy**: Data manipulation
- **Matplotlib**: Data visualization

## 📊 Model Performance

- **Accuracy**: 65.87%
- **Precision**: 51.94%
- **Recall**: 73.63%
- **F1 Score**: 60.91%

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
streamlit run profile_prophet_app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📈 How It Works

The model uses six demographic features to predict LinkedIn usage:
1. **Income Level** (1-9 scale)
2. **Education Level** (1-8 scale)
3. **Age**
4. **Parent Status**
5. **Marital Status**
6. **Gender**

## 👨‍💻 Author

**Clifford Akins**  
Master's in Business Analytics | Georgetown University McDonough School of Business

## 📝 License

This project is open source and available for educational purposes.
