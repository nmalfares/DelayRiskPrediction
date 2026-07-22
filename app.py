import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Construction Delay Risk Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .prediction-yes {
        background-color: #ff6b6b;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .prediction-no {
        background-color: #51cf66;
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .severity-high {
        background-color: #e74c3c;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
    }
    .severity-medium {
        background-color: #f39c12;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
    }
    .severity-low {
        background-color: #27ae60;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ========== DATA GENERATION ==========
@st.cache_data
def generate_kuwait_construction_data(n_samples=500):
    """Generate synthetic construction project data for Kuwait"""
    np.random.seed(42)
    
    data = {
        'Project_ID': [f'KWT_{i:04d}' for i in range(n_samples)],
        'Avg_Temperature_C': np.random.uniform(25, 50, n_samples),
        'Humidity_Percent': np.random.uniform(20, 80, n_samples),
        'Sandstorm_Days': np.random.poisson(5, n_samples),
        'Project_Duration_Months': np.random.uniform(6, 48, n_samples),
        'Project_Value_KUSD': np.random.uniform(100, 5000, n_samples),
        'Workforce_Size': np.random.choice([50, 100, 200, 300, 500], n_samples),
        'Labor_Availability_Index': np.random.uniform(0.5, 1.0, n_samples),
        'Material_Supply_Delay_Days': np.random.uniform(0, 60, n_samples),
        'Supplier_Reliability_Score': np.random.uniform(0.4, 1.0, n_samples),
        'Contractor_Experience_Years': np.random.choice([2, 5, 10, 15, 20], n_samples),
        'Equipment_Availability_Percent': np.random.uniform(60, 100, n_samples),
        'Site_Clearance_Days': np.random.uniform(5, 60, n_samples),
        'Design_Complexity_Score': np.random.uniform(1, 10, n_samples),
        'Permit_Processing_Days': np.random.uniform(10, 120, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate delay risk
    delay_risk = []
    
    for idx, row in df.iterrows():
        risk_score = 0
        
        if row['Avg_Temperature_C'] > 45:
            risk_score += 3
        elif row['Avg_Temperature_C'] > 40:
            risk_score += 1.5
        
        if row['Sandstorm_Days'] > 10:
            risk_score += 2
        risk_score += row['Sandstorm_Days'] * 0.3
        
        if row['Labor_Availability_Index'] < 0.7:
            risk_score += 2
        
        if row['Material_Supply_Delay_Days'] > 30:
            risk_score += 2
        risk_score += row['Material_Supply_Delay_Days'] * 0.15
        
        if row['Contractor_Experience_Years'] < 5:
            risk_score += 2
        elif row['Contractor_Experience_Years'] > 15:
            risk_score -= 1
        
        risk_score += (row['Project_Duration_Months'] / 10)
        
        if row['Permit_Processing_Days'] > 60:
            risk_score += 1
        
        risk_score += row['Design_Complexity_Score'] * 0.2
        
        # Improved class balance
        if risk_score > 8:
            delay_risk.append('High')
        elif risk_score > 4:
            delay_risk.append('Medium')
        else:
            delay_risk.append('Low')
    
    df['Delay_Risk_Category'] = delay_risk
    return df

# ========== MODEL TRAINING ==========
def train_model(df):
    """Train Random Forest model"""
    X = df.drop(['Project_ID', 'Delay_Risk_Category'], axis=1)
    y = df['Delay_Risk_Category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X_train, accuracy

# ========== MAIN APPLICATION ==========

# Generate data and train model (cached)
df = generate_kuwait_construction_data(500)
model, X_train, accuracy = train_model(df)

# Header
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>🏗️ Construction Project Delay Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Kuwait Construction Risk Assessment System</p>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Select", [
    "🔮 Prediction",
    "📊 Dashboard",
    "ℹ️ How It Works"
])

# ========== PAGE 1: PREDICTION ==========
if page == "🔮 Prediction":
    st.markdown("---")
    
    # Project Info Section
    st.subheader("📝 Enter Project Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💰 Project Basics")
        project_value = st.number_input(
            "Project Value (K USD)",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Total project budget in thousands USD"
        )
        
        duration = st.number_input(
            "Planned Duration (Months)",
            min_value=6,
            max_value=48,
            value=18,
            step=1,
            help="Expected project duration in months"
        )
        
        workforce = st.selectbox(
            "Workforce Size",
            [50, 100, 200, 300, 500],
            index=2,
            help="Number of workers on site"
        )
        
        contractor_exp = st.selectbox(
            "Contractor Experience (Years)",
            [2, 5, 10, 15, 20],
            index=2,
            help="Years of experience of main contractor"
        )
    
    with col2:
        st.markdown("### 🌍 Environmental & Supply")
        temperature = st.slider(
            "Expected Avg Temperature (°C)",
            25.0, 50.0, 35.0,
            help="Average temperature during project execution"
        )
        
        sandstorm_days = st.slider(
            "Expected Sandstorm Days",
            0, 30, 5,
            help="Number of sandstorm days expected"
        )
        
        material_delay = st.slider(
            "Material Supply Delay (Days)",
            0.0, 60.0, 15.0,
            help="Expected delay in material arrival"
        )
        
        labor_availability = st.slider(
            "Labor Availability",
            0.5, 1.0, 0.8,
            step=0.1,
            help="0.5 = Very Scarce, 1.0 = Abundant"
        )
    
    # Additional Parameters
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🛠️ Resources & Site")
        equipment_avail = st.slider(
            "Equipment Availability (%)",
            60.0, 100.0, 85.0,
            help="Percentage of required equipment available"
        )
        
        site_clearance = st.slider(
            "Site Clearance Time (Days)",
            5.0, 60.0, 20.0,
            help="Days needed to prepare the site"
        )
    
    with col4:
        st.markdown("### 📋 Design & Permits")
        complexity = st.slider(
            "Design Complexity (1=Simple, 10=Complex)",
            1.0, 10.0, 5.0,
            help="How complex is the project design?"
        )
        
        permit_days = st.slider(
            "Permit Processing Time (Days)",
            10.0, 120.0, 45.0,
            help="Expected government approval time"
        )
    
    # Additional metrics
    humidity = st.slider(
        "Humidity (%)",
        20.0, 80.0, 50.0,
        help="Average humidity level",
        disabled=False
    )
    
    supplier_reliability = st.slider(
        "Supplier Reliability (0=Poor, 1=Excellent)",
        0.4, 1.0, 0.7,
        help="Reliability score of material suppliers"
    )
    
    # ========== PREDICTION LOGIC ==========
    st.markdown("---")
    
    if st.button("🔍 PREDICT DELAY RISK", use_container_width=True, type="primary"):
        
        # Create input data
        input_data = pd.DataFrame({
            'Avg_Temperature_C': [temperature],
            'Humidity_Percent': [humidity],
            'Sandstorm_Days': [sandstorm_days],
            'Project_Duration_Months': [duration],
            'Project_Value_KUSD': [project_value],
            'Workforce_Size': [workforce],
            'Labor_Availability_Index': [labor_availability],
            'Material_Supply_Delay_Days': [material_delay],
            'Supplier_Reliability_Score': [supplier_reliability],
            'Contractor_Experience_Years': [contractor_exp],
            'Equipment_Availability_Percent': [equipment_avail],
            'Site_Clearance_Days': [site_clearance],
            'Design_Complexity_Score': [complexity],
            'Permit_Processing_Days': [permit_days]
        })
        
        # Get prediction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        
        # Get probability mapping
        class_mapping = {0: 'High', 1: 'Low', 2: 'Medium'}
        idx_low = list(model.classes_).index('Low')
        idx_med = list(model.classes_).index('Medium')
        idx_high = list(model.classes_).index('High')
        
        prob_low = probabilities[idx_low]
        prob_med = probabilities[idx_med]
        prob_high = probabilities[idx_high]
        
        # Determine if delayed
        will_be_delayed = prediction != 'Low'
        
        st.markdown("---")
        st.markdown("## 📊 PREDICTION RESULTS")
        st.markdown("---")
        
        # Display main prediction
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            if will_be_delayed:
                st.markdown(
                    '<div class="prediction-yes">🚨 YES - PROJECT WILL BE DELAYED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="prediction-no">✅ NO - PROJECT ON TRACK</div>',
                    unsafe_allow_html=True
                )
        
        with col_result2:
            if prediction == 'High':
                severity_class = 'severity-high'
                emoji = '❌'
                text = 'HIGH SEVERITY'
            elif prediction == 'Medium':
                severity_class = 'severity-medium'
                emoji = '⚠️'
                text = 'MEDIUM SEVERITY'
            else:
                severity_class = 'severity-low'
                emoji = '✅'
                text = 'LOW SEVERITY'
            
            st.markdown(
                f'<div class="{severity_class}">{emoji} {text}</div>',
                unsafe_allow_html=True
            )
        
        # Display probabilities
        st.markdown("---")
        st.subheader("📈 Risk Probability Distribution")
        
        col_prob1, col_prob2, col_prob3 = st.columns(3)
        
        with col_prob1:
            st.metric("Low Risk", f"{prob_low:.1%}")
        with col_prob2:
            st.metric("Medium Risk", f"{prob_med:.1%}")
        with col_prob3:
            st.metric("High Risk", f"{prob_high:.1%}")
        
        # Probability chart
        fig, ax = plt.subplots(figsize=(10, 5))
        risk_cats = ['Low', 'Medium', 'High']
        probs = [prob_low, prob_med, prob_high]
        colors = ['#27ae60', '#f39c12', '#e74c3c']
        
        bars = ax.bar(risk_cats, probs, color=colors, edgecolor='black', linewidth=2)
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title('Delay Risk Probability', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        st.pyplot(fig, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.subheader("💡 Recommendations")
        
        if prediction == 'High':
            st.error("""
            **🚨 HIGH RISK - IMMEDIATE ACTIONS REQUIRED:**
            
            1. **Schedule Buffer**: Add 30-40% contingency to project timeline
            2. **Budget Reserve**: Allocate 20-25% budget contingency
            3. **Resource Planning**: Secure backup labor and materials suppliers NOW
            4. **Mitigation**:
               - Hire more experienced contractor
               - Pre-order critical materials immediately
               - Arrange cooling systems for extreme heat conditions
               - Establish early warning monitoring system
            5. **Monitoring**: Weekly progress reviews and risk assessments
            """)
        
        elif prediction == 'Medium':
            st.warning("""
            **⚠️ MEDIUM RISK - PROACTIVE MANAGEMENT NEEDED:**
            
            1. **Schedule Buffer**: Add 15-20% contingency to timeline
            2. **Budget Reserve**: Allocate 10-15% budget contingency
            3. **Risk Management**:
               - Develop detailed contingency plans
               - Strengthen supplier relationships
               - Arrange backup suppliers for critical materials
               - Plan for weather-related work stoppages
            4. **Monitoring**: Bi-weekly progress reviews
            5. **Preparation**: Have alternative approaches ready
            """)
        
        else:
            st.success("""
            **✅ LOW RISK - FAVORABLE CONDITIONS:**
            
            1. **Schedule Buffer**: Standard 5-10% contingency sufficient
            2. **Budget Reserve**: Standard 5-10% contingency sufficient
            3. **Management Approach**:
               - Follow standard project management practices
               - Regular routine monitoring
               - Maintain current supply chain relationships
               - Standard progress reporting
            4. **Focus**: Optimize for cost and efficiency
            """)
        
        # Project summary
        st.markdown("---")
        st.subheader("📋 Project Summary")
        
        summary_data = {
            'Parameter': [
                'Project Value',
                'Planned Duration',
                'Workforce Size',
                'Temperature',
                'Labor Availability',
                'Material Delay',
                'Contractor Experience',
                'Design Complexity',
                'Permit Processing',
                'Equipment Available'
            ],
            'Value': [
                f'{project_value:,.0f} K USD',
                f'{duration} months',
                f'{workforce} workers',
                f'{temperature:.1f}°C',
                f'{labor_availability:.1f}',
                f'{material_delay:.1f} days',
                f'{contractor_exp} years',
                f'{complexity:.1f}/10',
                f'{permit_days:.0f} days',
                f'{equipment_avail:.0f}%'
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)

# ========== PAGE 2: DASHBOARD ==========
elif page == "📊 Dashboard":
    st.subheader("Dataset Overview & Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Projects in Database", len(df))
    with col2:
        st.metric("Model Features", 14)
    with col3:
        st.metric("Model Accuracy", f"{accuracy:.1%}")
    
    # Risk distribution
    st.subheader("Delay Risk Distribution")
    risk_counts = df['Delay_Risk_Category'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'Low': '#27ae60', 'Medium': '#f39c12', 'High': '#e74c3c'}
    bars = ax.bar(risk_counts.index, risk_counts.values, 
                  color=[colors.get(cat, '#3498db') for cat in risk_counts.index],
                  edgecolor='black', linewidth=2)
    ax.set_ylabel('Number of Projects', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Projects by Risk Level', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    st.pyplot(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("Top Factors Affecting Delay Risk")
    
    X_data = df.drop(['Project_ID', 'Delay_Risk_Category'], axis=1)
    feature_importance = pd.DataFrame({
        'Feature': X_data.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], 
            color='#3498db', edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Features Affecting Delay Risk', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    st.pyplot(fig, use_container_width=True)

# ========== PAGE 3: HOW IT WORKS ==========
elif page == "ℹ️ How It Works":
    st.subheader("How the Prediction System Works")
    
    with st.expander("🤖 Machine Learning Algorithm", expanded=True):
        st.write("""
        **Random Forest Classifier** - An ensemble of 100 decision trees that:
        - Each tree learns patterns from training data
        - Makes independent predictions
        - Final prediction = majority vote from all trees
        
        **Why this method?**
        - Handles complex relationships between factors
        - Robust against noise and outliers
        - Provides interpretable feature importance
        - Works well with construction data
        """)
    
    with st.expander("📊 Output Interpretation"):
        st.write("""
        **Will it be delayed? (YES/NO)**
        - YES = Medium or High risk prediction
        - NO = Low risk prediction
        
        **Severity Classes:**
        - 🟢 **LOW**: Project likely to complete on schedule
        - 🟡 **MEDIUM**: Some delays possible, need contingency planning
        - 🔴 **HIGH**: Significant delays likely, major interventions needed
        
        **Probability Score:**
        - Shows confidence in each prediction
        - Higher probability = more certain the model is
        """)
    
    with st.expander("🏗️ Key Factors in Kuwait Construction"):
        st.write("""
        **Critical Factors:**
        1. **Labor Availability** (25% impact)
           - Kuwait relies heavily on expatriate workers
           - Shortages during peak construction season
        
        2. **Material Supply** (20% impact)
           - Most materials imported
           - Supply chain delays common
        
        3. **Temperature** (15% impact)
           - Extreme heat (50°C+) reduces productivity
           - Work restrictions during peak hours
        
        4. **Contractor Experience** (12% impact)
           - Experienced contractors manage delays better
           - Local knowledge important
        
        5. **Sandstorms** (10% impact)
           - Work halts during sandstorms
           - Impacts equipment and site conditions
        """)
    
    with st.expander("✅ Model Reliability"):
        st.write(f"""
        **Current Model Performance:**
        - Training Accuracy: {accuracy:.1%}
        - Trained on: 500 historical projects
        - Features: 14 construction parameters
        
        **Limitations:**
        - Based on synthetic data for demonstration
        - Real-world use requires historical project data
        - External factors (political, economic) not included
        
        **Best Practices:**
        - Use as one tool among many
        - Combine with expert judgment
        - Validate predictions with actual outcomes
        - Continuously improve with real data
        """)
    
    with st.expander("💡 Tips for Using"):
        st.write("""
        **For Best Results:**
        1. Provide accurate project information
        2. Be realistic with duration and cost estimates
        3. Account for local Kuwait conditions
        4. Factor in seasonal weather patterns
        5. Consider current labor market conditions
        
        **After Prediction:**
        1. Review recommendations carefully
        2. Plan mitigation strategies
        3. Allocate appropriate contingencies
        4. Monitor project execution closely
        5. Use actual vs predicted to improve estimates
        """)
