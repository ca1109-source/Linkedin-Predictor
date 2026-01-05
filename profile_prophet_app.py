import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(
    page_title="The Profile Prophet",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mystical theme
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Oracle text styling */
    .oracle-text {
        font-family: 'Georgia', serif;
        font-size: 1.4rem;
        color: #ffffff;
        text-align: center;
        font-style: italic;
        padding: 25px;
        background: rgba(0, 0, 0, 0.6);
        border-radius: 15px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
        line-height: 1.8;
        margin: 20px auto;
        max-width: 800px;
    }
    
    /* Title styling */
    h1 {
        color: #ffd700 !important;
        text-align: center;
        text-shadow: 0 0 30px rgba(255, 215, 0, 1), 0 0 10px rgba(255, 215, 0, 0.8);
        font-family: 'Cinzel', serif;
        letter-spacing: 5px;
        font-size: 3.5rem !important;
        margin-bottom: 30px !important;
    }
    
    /* Hooded figure */
    .hooded-figure {
        font-size: 8rem;
        text-align: center;
        animation: float 3s ease-in-out infinite;
        margin: 20px 0;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-15px); }
    }
    
      /* Crystal ball button styling */
    .stButton > button {
        font-size: 6rem !important;
        background: transparent !important;
        border: none !important;
        cursor: pointer !important;
        animation: glow 2s ease-in-out infinite;
        padding: 20px !important;
        display: block !important;
        margin: 0 auto !important;
    }
    
    @keyframes glow {
        0%, 100% { 
            filter: drop-shadow(0 0 20px rgba(138, 43, 226, 0.6));
            transform: scale(1);
        }
        50% { 
            filter: drop-shadow(0 0 40px rgba(138, 43, 226, 1));
            transform: scale(1.05);
        }
    }
    
    /* Mystical divider */
    .mystical-divider {
        text-align: center;
        color: #ffd700;
        font-size: 2rem;
        margin: 30px 0;
    }
    
    /* Input labels - HIGH VISIBILITY */
    .stSelectbox label, .stSlider label {
        color: #ffffff !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 1);
        background: rgba(0, 0, 0, 0.5);
        padding: 5px 10px;
        border-radius: 5px;
    }
    
    /* Dropdown styling - HIGH VISIBILITY */
    .stSelectbox > div > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        border: 2px solid #ffd700 !important;
    }
    
    /* Dropdown text */
    .stSelectbox [data-baseweb="select"] {
        background-color: #2d3748 !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #2d3748 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
    }
    
    /* Section headers */
    h3 {
        color: #ffd700 !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
        background: rgba(0, 0, 0, 0.5);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem !important;
    }
    
    /* Success/Error messages */
    .success-oracle {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 100%);
        border: 3px solid #4ade80;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        font-size: 1.6rem;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(74, 222, 128, 0.5);
        margin: 30px 0;
        font-weight: 600;
    }
    
    .error-oracle {
        background: linear-gradient(135deg, #4a1a1a 0%, #5d2d2d 100%);
        border: 3px solid #ef4444;
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        font-size: 1.6rem;
        color: #ffffff;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
        margin: 30px 0;
        font-weight: 600;
    }
    
    /* Instruction text */
    .instruction-text {
        color: #ffd700;
        text-align: center;
        font-size: 1.3rem;
        margin-top: 30px;
        font-style: italic;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.8);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffd700 !important;
        font-size: 2rem !important;
        text-shadow: 0 0 10px rgba(255, 215, 0, 0.6);
    }
    
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    
    /* Expander styling - MAXIMUM VISIBILITY */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%) !important;
        color: #ffffff !important;
        border: 3px solid #ffd700 !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        padding: 15px 20px !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 1) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, #374151 0%, #1f2937 100%) !important;
        border-color: #ffed4e !important;
        box-shadow: 0 0 15px rgba(255, 215, 0, 0.5) !important;
    }
    
    .streamlit-expanderHeader p {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.3rem !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 1) !important;
    }
    
    .streamlit-expanderHeader svg {
        fill: #ffd700 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.6) !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 2px solid #ffd700 !important;
        margin-top: 5px !important;
    }
    
    /* Primary button styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        padding: 15px 30px !important;
        border: 2px solid #ffd700 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.5) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 30px rgba(139, 92, 246, 0.8) !important;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_inputs' not in st.session_state:
    st.session_state.show_inputs = False
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Load and prepare data
@st.cache_data
def load_data():
    s = pd.read_csv('social_media_usage (1).csv')
    
    def clean_sm(x):
        return np.where(x == 1, 1, 0)
    
    ss = pd.DataFrame()
    ss['sm_li'] = clean_sm(s['web1h'])
    ss['income'] = np.where(s['income'] > 9, np.nan, s['income'])
    ss['education'] = np.where(s['educ2'] > 8, np.nan, s['educ2'])
    ss['parent'] = np.where(s['par'] == 1, 1, 0)
    ss['married'] = np.where(s['marital'] == 1, 1, 0)
    ss['female'] = np.where(s['gender'] == 2, 1, 0)
    ss['age'] = np.where(s['age'] > 98, np.nan, s['age'])
    ss = ss.dropna()
    
    return ss

# Train model
@st.cache_resource
def train_model(ss):
    y = ss['sm_li']
    X = ss[['income', 'education', 'parent', 'married', 'female', 'age']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train, y_train)
    return lr

# Load data and train model
ss = load_data()
model = train_model(ss)

# Title
st.markdown("<h1>THE PROFILE PROPHET</h1>", unsafe_allow_html=True)

# Show landing page or input form based on state
if not st.session_state.show_inputs:
    # Landing page with hooded figure and crystal ball
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Hooded figure
    st.markdown("<div class='hooded-figure'>🧙‍♂️</div>", unsafe_allow_html=True)
    
    # Oracle greeting
    st.markdown("""
        <div class='oracle-text'>
            <strong>Welcome to the Profile Prophet</strong><br><br>
            Predict LinkedIn usage patterns using machine learning and demographic data.<br><br>
            Click the crystal ball below to begin your analysis.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Crystal ball button
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("🔮", key="crystal_ball", help="Click to begin prediction"):
            st.session_state.show_inputs = True
            st.rerun()
    
    st.markdown("<div class='instruction-text'>✨ Click the crystal ball to begin ✨</div>", unsafe_allow_html=True)

else:
    # Show the input form
    st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='oracle-text' style='font-size: 1.2rem;'>
            Enter demographic information to predict LinkedIn usage probability
        </div>
    """, unsafe_allow_html=True)
    
    # Create centered container for inputs
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Input sections in main area
    st.markdown("### Income & Education")
    col1, col2 = st.columns(2)
    
    with col1:
        income = st.slider(
            "Income Level",
            min_value=1,
            max_value=9,
            value=5,
            help="1 = Less than $10k, 9 = $150k+"
        )
    
    with col2:
        education = st.slider(
            "Education Level",
            min_value=1,
            max_value=8,
            value=4,
            help="1 = Less than HS, 8 = Postgraduate degree"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Age
    st.markdown("### Age")
    age = st.slider(
        "Age",
        min_value=18,
        max_value=97,
        value=42
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Personal Information
    st.markdown("### Personal Information")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        parent = st.selectbox(
            "Parent Status",
            options=[0, 1],
            format_func=lambda x: "Parent" if x == 1 else "Not a Parent",
            index=0
        )
    
    with col4:
        married = st.selectbox(
            "Marital Status",
            options=[0, 1],
            format_func=lambda x: "Married" if x == 1 else "Not Married",
            index=0
        )
    
    with col5:
        female = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 1 else "Male",
            index=0
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)
    
    # Center the predict button
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_b:
        predict_button = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if predict_button:
        st.session_state.prediction_made = True
        
        # Create input dataframe
        person = pd.DataFrame({
            'income': [income],
            'education': [education],
            'parent': [parent],
            'married': [married],
            'female': [female],
            'age': [age]
        })
        
        # Make prediction
        prediction = model.predict(person)[0]
        probability = model.predict_proba(person)[0]
        
        # Display results
        if prediction == 1:
            st.markdown(f"""
                <div class='success-oracle'>
                    <strong>PREDICTION: LinkedIn User</strong><br><br>
                    This user profile is predicted to use LinkedIn<br><br>
                    <em>Confidence: {probability[1]:.1%}</em>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='error-oracle'>
                    <strong>PREDICTION: Not a LinkedIn User</strong><br><br>
                    This user profile is predicted to NOT use LinkedIn<br><br>
                    <em>Confidence: {probability[0]:.1%}</em>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)
        
        # Display probabilities
        st.markdown("### The Balance of Probabilities")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.metric(
                label="Does NOT use LinkedIn",
                value=f"{probability[0]:.1%}"
            )
            st.progress(probability[0])
        
        with col_right:
            st.metric(
                label="DOES use LinkedIn",
                value=f"{probability[1]:.1%}"
            )
            st.progress(probability[1])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Detailed analysis
        with st.expander("View Detailed Analysis"):
            st.markdown("### Feature Impact on Prediction")
            
            import matplotlib.pyplot as plt
            
            # Set style for mystical charts
            plt.style.use('dark_background')
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(14, 16))
            fig.patch.set_facecolor('#1a1a2e')
            
            for ax in axes.flat:
                ax.set_facecolor('#16213e')
            
            # 1. Income Analysis
            income_range = range(1, 10)
            income_probs = []
            for inc in income_range:
                test = pd.DataFrame({
                    'income': [inc],
                    'education': [education],
                    'parent': [parent],
                    'married': [married],
                    'female': [female],
                    'age': [age]
                })
                prob = model.predict_proba(test)[0][1]
                income_probs.append(prob)
            
            axes[0,0].plot(income_range, income_probs, marker='o', linewidth=3, color='#ffd700', markersize=10)
            axes[0,0].axvline(x=income, color='#ff6b6b', linestyle='--', linewidth=2.5, label='Current User')
            axes[0,0].axhline(y=probability[1], color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=2)
            axes[0,0].set_xlabel('Income Level', color='#ffffff', fontsize=12, fontweight='bold')
            axes[0,0].set_ylabel('LinkedIn Usage Probability', color='#ffffff', fontsize=12, fontweight='bold')
            axes[0,0].set_title('Income Impact', color='#ffd700', fontsize=14, fontweight='bold')
            axes[0,0].legend(facecolor='#1a1a2e', edgecolor='#ffd700', fontsize=10)
            axes[0,0].grid(True, alpha=0.3, color='#9d84b7')
            axes[0,0].tick_params(colors='#ffffff')
            
            # 2. Education Analysis
            education_range = range(1, 9)
            education_probs = []
            for edu in education_range:
                test = pd.DataFrame({
                    'income': [income],
                    'education': [edu],
                    'parent': [parent],
                    'married': [married],
                    'female': [female],
                    'age': [age]
                })
                prob = model.predict_proba(test)[0][1]
                education_probs.append(prob)
            
            axes[0, 1].plot(education_range, education_probs, marker='o', linewidth=3, color='#00d4ff', markersize=10)
            axes[0, 1].axvline(x=education, color='#ff6b6b', linestyle='--', linewidth=2.5, label='Current User')
            axes[0, 1].axhline(y=probability[1], color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=2)
            axes[0, 1].set_xlabel('Education Level', color='#ffffff', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('LinkedIn Usage Probability', color='#ffffff', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Education Impact', color='#ffd700', fontsize=14, fontweight='bold')
            axes[0, 1].legend(facecolor='#1a1a2e', edgecolor='#ffd700', fontsize=10)
            axes[0, 1].grid(True, alpha=0.3, color='#9d84b7')
            axes[0, 1].tick_params(colors='#ffffff')
            
            # 3. Age Analysis
            age_range = range(18, 98)
            age_probs = []
            for age_val in age_range:
                test = pd.DataFrame({
                    'income': [income],
                    'education': [education],
                    'parent': [parent],
                    'married': [married],
                    'female': [female],
                    'age': [age_val]
                })
                prob = model.predict_proba(test)[0][1]
                age_probs.append(prob)
            
            axes[1,0].plot(age_range, age_probs, marker='.', linewidth=3, color='#b19cd9', markersize=5)
            axes[1,0].axvline(x=age, color='#ff6b6b', linestyle='--', linewidth=2.5, label='Current User')
            axes[1,0].axhline(y=probability[1], color='#ff6b6b', linestyle=':', alpha=0.6, linewidth=2)
            axes[1,0].set_xlabel('Age', color='#ffffff', fontsize=12, fontweight='bold')
            axes[1,0].set_ylabel('LinkedIn Usage Probability', color='#ffffff', fontsize=12, fontweight='bold')
            axes[1,0].set_title('Age Impact', color='#ffd700', fontsize=14, fontweight='bold')
            axes[1,0].legend(facecolor='#1a1a2e', edgecolor='#ffd700', fontsize=10)
            axes[1,0].grid(True, alpha=0.3, color='#9d84b7')
            axes[1,0].tick_params(colors='#ffffff')
            
            # 4. Parent Status Analysis
            parent_options = [0, 1]
            parent_probs = []
            for parent_val in parent_options:
                test = pd.DataFrame({
                    'income': [income],
                    'education': [education],
                    'parent': [parent_val],
                    'married': [married],
                    'female': [female],
                    'age': [age]
                })
                prob = model.predict_proba(test)[0][1]
                parent_probs.append(prob)
            
            colors_parent = ['#ff8c42' if p == parent else '#4a5568' for p in parent_options]
            bars1 = axes[1, 1].bar(['Not Parent', 'Parent'], parent_probs, color=colors_parent, edgecolor='#ffd700', linewidth=2.5)
            axes[1, 1].axhline(y=probability[1], color='#ff6b6b', linestyle='--', linewidth=2.5, label='Current User')
            axes[1, 1].set_ylabel('LinkedIn Usage Probability', color='#ffffff', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Parent Status Impact', color='#ffd700', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylim([0, 1])
            axes[1, 1].legend(facecolor='#1a1a2e', edgecolor='#ffd700', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3, axis='y', color='#9d84b7')
            axes[1, 1].tick_params(colors='#ffffff')
            
            # 5. Marital Status Analysis
            married_options = [0, 1]
            married_probs = []
            for married_val in married_options:
                test = pd.DataFrame({
                    'income': [income],
                    'education': [education],
                    'parent': [parent],
                    'married': [married_val],
                    'female': [female],
                    'age': [age]
                })
                prob = model.predict_proba(test)[0][1]
                married_probs.append(prob)
            
            colors_married = ['#3b82f6' if m == married else '#4a5568' for m in married_options]
            bars2 = axes[2, 0].bar(['Not Married', 'Married'], married_probs, color=colors_married, edgecolor='#ffd700', linewidth=2.5)
            axes[2, 0].axhline(y=probability[1], color='#ff6b6b', linestyle='--', linewidth=2.5, label='Current User')
            axes[2, 0].set_ylabel('LinkedIn Usage Probability', color='#ffffff', fontsize=12, fontweight='bold')
            axes[2, 0].set_title('Marital Status Impact', color='#ffd700', fontsize=14, fontweight='bold')
            axes[2, 0].set_ylim([0, 1])
            axes[2, 0].legend(facecolor='#1a1a2e', edgecolor='#ffd700', fontsize=10)
            axes[2, 0].grid(True, alpha=0.3, axis='y', color='#9d84b7')
            axes[2, 0].tick_params(colors='#ffffff')
            
            # 6. Gender Analysis
            female_options = [0, 1]
            female_probs = []
            for female_val in female_options:
                test = pd.DataFrame({
                    'income': [income],
                    'education': [education],
                    'parent': [parent],
                    'married': [married],
                    'female': [female_val],
                    'age': [age]
                })
                prob = model.predict_proba(test)[0][1]
                female_probs.append(prob)
            
            colors_gender = ['#ec4899' if f == female else '#4a5568' for f in female_options]
            bars3 = axes[2, 1].bar(['Male', 'Female'], female_probs, color=colors_gender, edgecolor='#ffd700', linewidth=2.5)
            axes[2, 1].axhline(y=probability[1], color='#ff6b6b', linestyle='--', linewidth=2.5, label='Current User')
            axes[2, 1].set_ylabel('LinkedIn Usage Probability', color='#ffffff', fontsize=12, fontweight='bold')
            axes[2, 1].set_title('Gender Impact', color='#ffd700', fontsize=14, fontweight='bold')
            axes[2, 1].set_ylim([0, 1])
            axes[2, 1].legend(facecolor='#1a1a2e', edgecolor='#ffd700', fontsize=10)
            axes[2, 1].grid(True, alpha=0.3, axis='y', color='#9d84b7')
            axes[2, 1].tick_params(colors='#ffffff')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    elif not st.session_state.prediction_made:
        st.markdown("""
            <div class='oracle-text' style='font-size: 1.1rem;'>
                Enter demographic information above and click 'Generate Prediction'
            </div>
        """, unsafe_allow_html=True)

# Add information section at the bottom
st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)

if st.session_state.show_inputs:
    st.markdown("### Model Information")
    
    with st.expander("How does this model work?"):
        st.markdown("""
            <div style='color: #ffffff; background: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px;'>
            This prediction model uses <strong>Logistic Regression</strong> trained on survey data 
            to predict LinkedIn usage based on demographic features.
            <br><br>
            <strong>Features Used:</strong><br>
            • Income Level (1-9 scale)<br>
            • Education Level (1-8 scale)<br>
            • Age<br>
            • Parent Status<br>
            • Marital Status<br>
            • Gender<br>
            <br>
            <strong>Model Performance:</strong><br>
            • Accuracy: 65.87%<br>
            • Precision: 51.94%<br>
            • Recall: 73.63%<br>
            • F1 Score: 60.91%
            </div>
        """, unsafe_allow_html=True)
    
    with st.expander("What do the scales mean?"):
        st.markdown("""
            <div style='color: #ffffff; background: rgba(0,0,0,0.5); padding: 20px; border-radius: 10px;'>
            <strong>Income Scale (1-9):</strong><br>
            1 = Less than $10,000<br>
            2 = $10,000 to under $20,000<br>
            3 = $20,000 to under $30,000<br>
            4 = $30,000 to under $40,000<br>
            5 = $40,000 to under $50,000<br>
            6 = $50,000 to under $75,000<br>
            7 = $75,000 to under $100,000<br>
            8 = $100,000 to under $150,000<br>
            9 = $150,000 or more<br>
            <br>
            <strong>Education Scale (1-8):</strong><br>
            1 = Less than high school<br>
            2 = High school incomplete<br>
            3 = High school graduate<br>
            4 = Some college, no degree<br>
            5 = Two-year associate degree<br>
            6 = Four-year bachelor's degree<br>
            7 = Some postgraduate school<br>
            8 = Postgraduate or professional degree
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #ffd700; font-style: italic; padding: 20px; background: rgba(0,0,0,0.5); border-radius: 10px; font-size: 1.1rem;'>
        Created by Clifford Akins | Georgetown MSBA
    </div>
""", unsafe_allow_html=True)