# Diabetes Detection Using Streamlit

## Overview
This project utilizes machine learning random forest algorithm to predict diabetes outcomes based on patient data.
It features a user-friendly web interface built with Streamlit Community Cloud, allowing users to input their health metrics and receive instant visualized predictions.


## Website

[Diabetes-Detection](https://diabetes-detection-tm0p.onrender.com/)

## Project Structure and Description  

```- Diabetes Detection/
│
├── app.py                          # Main application file
├── train_model.py                  # Script to train the model
├── dataset/
│   └── diabetes.csv                # Dataset for training
├── templates/
│   ├── index.html                  # Input form
│   └── results.html                # Results display
├── models/
│   ├── diabetes_detection_model.h5  # Pre-trained model .h5 extension
│   └── rf_model.pkl                # Random Forest model
└── requirements.txt                # Dependencies


```
## Prerequisites

- Python 3.12
- Streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- TensorFlow (if using .h5 model)
- HTML/CSS for frontend design

## Installation

### Step 1: Clone the Repository
**Clone the repository to your local machine:**
```bash      
git clone <repository_url>
cd Diabetes Detection
```

### Step 2: Set Up a Virtual Environment (Optional but Recommended)
**Create and activate a virtual environment:**

# Create a virtual environment
```bash
python -m venv venv
```

# Activate the virtual environment
# On Windows
```bash
venv\Scripts\activate
```

# On macOS/Linux
```bash
source venv/bin/activate
```

### Step 3: Install Dependencies
**Install the required libraries:**

   ```bash
     pip install -r requirements.txt
```

### Step 4: Run the Application
1. **Ensure you're in the project directory.**

2. **Run the Streamlit app:**

   ```bash
   streamlit run app.py
   ```

## Usage

*   Input patient data using the sliders in the sidebar.
*   Click the "Submit" button to get predictions and visualizations.
*   The app will display the patient's data along with the prediction and accuracy of the model.

## Hosting 
Hosting is done using RENDER.


## Conclusion

The Diabetes Detection application leverages machine learning to provide insights into an individual's diabetes risk based on their health data. By using a Random Forest Classifier, the model can accurately predict outcomes, empowering users to make informed health decisions. This tool not only aids in early detection but also promotes awareness and proactive health management.

