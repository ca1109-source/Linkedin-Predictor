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
        font-size: 1.3rem;
        color: #e8dcc4;
        text-align: center;
        font-style: italic;
        padding: 20px;
        text-shadow: 0 0 10px rgba(232, 220, 196, 0.5);
        line-height: 1.8;
    }
    
    /* Title styling */
    h1 {
        color: #ffd700 !important;
        text-align: center;
        text-shadow: 0 0 20px rgba(255, 215, 0, 0.8);
        font-family: 'Cinzel', serif;
        letter-spacing: 3px;
    }
    
    /* Crystal ball button */
    .crystal-ball {
        font-size: 8rem;
        text-align: center;
        cursor: pointer;
        animation: glow 2s ease-in-out infinite;
        filter: drop-shadow(0 0 30px rgba(138, 43, 226, 0.8));
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
        color: #9d84b7;
        font-size: 2rem;
        margin: 20px 0;
    }
    
    /* Result boxes */
    .stAlert {
        background-color: rgba(0, 0, 0, 0.4) !important;
        border: 2px solid #ffd700 !important;
        color: #e8dcc4 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b69 0%, #1a1a2e 100%);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ffd700 !important;
    }
    
    /* Success/Error messages */
    .success-oracle {
        background: linear-gradient(135deg, #1a472a 0%, #2d5016 100%);
        border: 2px solid #4ade80;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        color: #a7f3d0;
        box-shadow: 0 0 20px rgba(74, 222, 128, 0.3);
    }
    
    .error-oracle {
        background: linear-gradient(135deg, #4a1a1a 0%, #5d2d2d 100%);
        border: 2px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        font-size: 1.5rem;
        color: #fca5a5;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.3);
    }
    
    /* Oracle character */
    .oracle-character {
        font-size: 5rem;
        text-align: center;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Instruction text */
    .instruction-text {
        color: #9d84b7;
        text-align: center;
        font-size: 1.1rem;
        margin-top: 20px;
        font-style: italic;
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
    # Landing page with oracle and crystal ball
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Oracle character - simplified
    st.markdown("<div class='oracle-character'>🔮</div>", unsafe_allow_html=True)
    
    # Oracle greeting
    st.markdown("""
        <div class='oracle-text'>
            Welcome, seeker of insights.<br><br>
            I am the Profile Prophet. The crystal reveals patterns of LinkedIn presence.<br><br>
            Click below to begin your prediction.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Crystal ball button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔮", key="crystal_ball", help="Click to peer into the crystal..."):
            st.session_state.show_inputs = True
            st.rerun()
    
    st.markdown("<div class='instruction-text'>Click the crystal ball to begin</div>", unsafe_allow_html=True)

else:
    # Show the input form
    st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)
    st.markdown("<div class='oracle-text'>Enter demographic information below</div>", unsafe_allow_html=True)
    
    # Sidebar for user inputs
    st.sidebar.markdown("### User Demographics")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("#### Income & Education")
    income = st.sidebar.slider(
        "Income Level",
        min_value=1,
        max_value=9,
        value=5,
        help="1 = Less than $10k, 9 = $150k+"
    )
    
    st.sidebar.markdown("#### Education")
    education = st.sidebar.slider(
        "Education Level",
        min_value=1,
        max_value=8,
        value=4,
        help="1 = Less than HS, 8 = Postgraduate degree"
    )
    
    st.sidebar.markdown("#### Age")
    age = st.sidebar.slider(
        "Age",
        min_value=18,
        max_value=97,
        value=42
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Personal Information")
    
    parent = st.sidebar.selectbox(
        "Parent Status",
        options=[0, 1],
        format_func=lambda x: "Parent" if x == 1 else "Not a Parent",
        index=0
    )
    
    married = st.sidebar.selectbox(
        "Marital Status",
        options=[0, 1],
        format_func=lambda x: "Married" if x == 1 else "Not Married",
        index=0
    )
    
    female = st.sidebar.selectbox(
        "Gender",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 1 else "Male",
        index=0
    )
    
    st.sidebar.markdown("---")
    predict_button = st.sidebar.button("Predict LinkedIn Usage", type="primary", use_container_width=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### User Profile")
        
        profile_data = {
            "Attribute": ["Income Level", "Education Level", "Age", "Parent Status", "Marital Status", "Gender"],
            "Value": [
                f"Level {income}",
                f"Level {education}",
                f"{age} years",
                "Parent" if parent == 1 else "Not a Parent",
                "Married" if married == 1 else "Not Married",
                "Female" if female == 1 else "Male"
            ]
        }
        
        st.table(pd.DataFrame(profile_data))
    
    with col2:
        st.markdown("### Prediction Results")
        
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
            
            # Display mystical results
            st.markdown("<br>", unsafe_allow_html=True)
            
            if prediction == 1:
                st.markdown(f"""
                    <div class='success-oracle'>
                        <strong>PREDICTION: LinkedIn User</strong><br><br>
                        This profile is predicted to use LinkedIn<br>
                        Confidence: {probability[1]:.1%}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='error-oracle'>
                        <strong>PREDICTION: Not a LinkedIn User</strong><br><br>
                        This profile is predicted to NOT use LinkedIn<br>
                        Confidence: {probability[0]:.1%}
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)
            
            # Display probabilities
            st.markdown("#### Probability Breakdown")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    label="Does NOT use LinkedIn",
                    value=f"{probability[0]:.1%}"
                )
                st.progress(probability[0])
            
            with col_b:
                st.metric(
                    label="DOES use LinkedIn",
                    value=f"{probability[1]:.1%}"
                )
                st.progress(probability[1])
            
            # Detailed analysis
            with st.expander("View Detailed Analysis"):
                st.markdown("### Feature Impact on Prediction")
                
                import matplotlib.pyplot as plt
                
                # Set style for mystical charts
                plt.style.use('dark_background')
                
                # Create figure with subplots
                fig, axes = plt.subplots(3, 2, figsize=(12, 14))
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
                
                axes[0,0].plot(income_range, income_probs, marker='o', linewidth=2.5, color='#ffd700', markersize=8)
                axes[0,0].axvline(x=income, color='#ff6b6b', linestyle='--', linewidth=2, label='This Mortal')
                axes[0,0].axhline(y=probability[1], color='#ff6b6b', linestyle=':', alpha=0.5)
                axes[0,0].set_xlabel('Income Level', color='#e8dcc4', fontsize=11)
                axes[0,0].set_ylabel('LinkedIn Usage Probability', color='#e8dcc4', fontsize=11)
                axes[0,0].set_title('Income Impact', color='#ffd700', fontsize=13, fontweight='bold')
                axes[0,0].legend(facecolor='#1a1a2e', edgecolor='#ffd700')
                axes[0,0].grid(True, alpha=0.2, color='#9d84b7')
                axes[0,0].tick_params(colors='#e8dcc4')
                
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
                
                axes[0, 1].plot(education_range, education_probs, marker='o', linewidth=2.5, color='#00d4ff', markersize=8)
                axes[0, 1].axvline(x=education, color='#ff6b6b', linestyle='--', linewidth=2, label='This Mortal')
                axes[0, 1].axhline(y=probability[1], color='#ff6b6b', linestyle=':', alpha=0.5)
                axes[0, 1].set_xlabel('Education Level', color='#e8dcc4', fontsize=11)
                axes[0, 1].set_ylabel('LinkedIn Usage Probability', color='#e8dcc4', fontsize=11)
                axes[0, 1].set_title('Education Impact', color='#ffd700', fontsize=13, fontweight='bold')
                axes[0, 1].legend(facecolor='#1a1a2e', edgecolor='#ffd700')
                axes[0, 1].grid(True, alpha=0.2, color='#9d84b7')
                axes[0, 1].tick_params(colors='#e8dcc4')
                
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
                
                axes[1,0].plot(age_range, age_probs, marker='.', linewidth=2.5, color='#b19cd9', markersize=4)
                axes[1,0].axvline(x=age, color='#ff6b6b', linestyle='--', linewidth=2, label='This Mortal')
                axes[1,0].axhline(y=probability[1], color='#ff6b6b', linestyle=':', alpha=0.5)
                axes[1,0].set_xlabel('Age', color='#e8dcc4', fontsize=11)
                axes[1,0].set_ylabel('LinkedIn Usage Probability', color='#e8dcc4', fontsize=11)
                axes[1,0].set_title('Age Impact', color='#ffd700', fontsize=13, fontweight='bold')
                axes[1,0].legend(facecolor='#1a1a2e', edgecolor='#ffd700')
                axes[1,0].grid(True, alpha=0.2, color='#9d84b7')
                axes[1,0].tick_params(colors='#e8dcc4')
                
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
                bars1 = axes[1, 1].bar(['Not Parent', 'Parent'], parent_probs, color=colors_parent, edgecolor='#ffd700', linewidth=2)
                axes[1, 1].axhline(y=probability[1], color='#ff6b6b', linestyle='--', linewidth=2, label='This Mortal')
                axes[1, 1].set_ylabel('LinkedIn Usage Probability', color='#e8dcc4', fontsize=11)
                axes[1, 1].set_title('Parent Status Impact', color='#ffd700', fontsize=13, fontweight='bold')
                axes[1, 1].set_ylim([0, 1])
                axes[1, 1].legend(facecolor='#1a1a2e', edgecolor='#ffd700')
                axes[1, 1].grid(True, alpha=0.2, axis='y', color='#9d84b7')
                axes[1, 1].tick_params(colors='#e8dcc4')
                
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
                bars2 = axes[2, 0].bar(['Not Married', 'Married'], married_probs, color=colors_married, edgecolor='#ffd700', linewidth=2)
                axes[2, 0].axhline(y=probability[1], color='#ff6b6b', linestyle='--', linewidth=2, label='This Mortal')
                axes[2, 0].set_ylabel('LinkedIn Usage Probability', color='#e8dcc4', fontsize=11)
                axes[2, 0].set_title('Marital Status Impact', color='#ffd700', fontsize=13, fontweight='bold')
                axes[2, 0].set_ylim([0, 1])
                axes[2, 0].legend(facecolor='#1a1a2e', edgecolor='#ffd700')
                axes[2, 0].grid(True, alpha=0.2, axis='y', color='#9d84b7')
                axes[2, 0].tick_params(colors='#e8dcc4')
                
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
                bars3 = axes[2, 1].bar(['Male', 'Female'], female_probs, color=colors_gender, edgecolor='#ffd700', linewidth=2)
                axes[2, 1].axhline(y=probability[1], color='#ff6b6b', linestyle='--', linewidth=2, label='This Mortal')
                axes[2, 1].set_ylabel('LinkedIn Usage Probability', color='#e8dcc4', fontsize=11)
                axes[2, 1].set_title('Gender Impact', color='#ffd700', fontsize=13, fontweight='bold')
                axes[2, 1].set_ylim([0, 1])
                axes[2, 1].legend(facecolor='#1a1a2e', edgecolor='#ffd700')
                axes[2, 1].grid(True, alpha=0.2, axis='y', color='#9d84b7')
                axes[2, 1].tick_params(colors='#e8dcc4')
                
                plt.tight_layout()
                st.pyplot(fig)
        
        else:
            st.markdown("""
                <div class='oracle-text' style='font-size: 1.1rem;'>
                    Enter user information and click 'Predict LinkedIn Usage'
                </div>
            """, unsafe_allow_html=True)

# Add information section at the bottom
st.markdown("<div class='mystical-divider'>✦ ✧ ✦</div>", unsafe_allow_html=True)

if st.session_state.show_inputs:
    st.markdown("### Model Information")
    
    with st.expander("How does this model work?"):
        st.markdown("""
            <div style='color: #e8dcc4;'>
            This prediction model uses <strong>Logistic Regression</strong> trained on survey data 
            to predict LinkedIn usage based on demographic features.
            <br><br>
            <strong>Features used:</strong><br>
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
            <div style='color: #e8dcc4;'>
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
    <div style='text-align: center; color: #9d84b7; font-style: italic; padding: 20px;'>
        Created by Clifford Akins | Georgetown MSBA
    </div>
""", unsafe_allow_html=True)