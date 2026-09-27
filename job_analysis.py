import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration
st.set_page_config(page_title="Placement Analytics Dashboard", layout="wide")
st.title("📊 Placement Analytics & Key KPIs Dashboard")


@st.cache_data
def load_data():
    df = pd.read_csv("Hr_job_analysis.csv")
    if 'interview_avg_score' not in df.columns:
        df['interview_avg_score'] = (df['technical_score'] + df['aptitude_score'] + df['communication_score']) / 3
    return df

df = load_data()

st.markdown("---")

# ==========================================
# 9) KEY KPIS SECTION
# ==========================================
st.header("🎯 Key Metrics & KPIs")

total_candidates = len(df)
placed_candidates = len(df[df['status'] == 'Placed'])
placement_rate = (placed_candidates / total_candidates) * 100 if total_candidates > 0 else 0

# Offer dropout rate (Bond requirement or specific status )
bond_required = len(df[df['bond_requirement'] == 'Required'])
offer_dropout_rate = (bond_required / total_candidates) * 100 if total_candidates > 0 else 0

avg_interview_score = df['interview_avg_score'].mean()
avg_skills_match = df['skills_match_percentage'].mean()

# High risk candidate (% - Low skills match and low interview score)
high_risk = len(df[(df['skills_match_percentage'] < 50) & (df['interview_avg_score'] < 50)])
high_risk_pct = (high_risk / total_candidates) * 100 if total_candidates > 0 else 0

# KPI Display Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Candidates", f"{total_candidates:,}")
col2.metric("Placement Rate (%)", f"{placement_rate:.1f}%")
col3.metric("Job Acceptance Rate (%)", f"{placement_rate:.1f}%")
col4.metric("Avg Interview Score", f"{avg_interview_score:.1f}")

col5, col6, col7 = st.columns(3)
col5.metric("Avg Skills Match %", f"{avg_skills_match:.1f}%")
col6.metric("Offer Dropout Rate (%)", f"{offer_dropout_rate:.1f}%")
col7.metric("High-Risk Candidates (%)", f"{high_risk_pct:.1f}%")

st.markdown("---")

# ==========================================
# 8) ANALYST TASKS (EDA & ML ANALYTICS)
# ==========================================
st.header("📈 Analyst Tasks (EDA & ML Analytics)")

tab1, tab2, tab3 = st.tabs([
    "👨‍🎓 Candidate Performance", 
    "🏢 Placement & Acceptance", 
    "🎤 Interview & Evaluation"
])

# ------------------------------------------
# TAB 1: Candidate Performance Analysis
# ------------------------------------------
with tab1:
    st.subheader("Candidate Performance Analysis")
    
    # 1. Academic scores vs placement outcome
    fig1 = px.histogram(df, x="degree_percentage", color="status", barmode="overlay",
                        title="Academic Scores (Degree %) vs Placement Outcome")
    st.plotly_chart(fig1, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        # 2. Skills match vs interview performance
        fig2 = px.scatter(df, x="skills_match_percentage", y="interview_avg_score", color="status",
                          title="Skills Match % vs Interview Performance Score")
        st.plotly_chart(fig2, use_container_width=True)
        
    with col_b:
        # 3. Certification impact on job acceptance
        fig3 = px.histogram(df, x="certifications_count", color="status", barmode="group",
                            title="Certification Impact on Job Acceptance")
        st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------
# TAB 2: Placement & Acceptance Analysis
# ------------------------------------------
with tab2:
    st.subheader("Placement & Acceptance Analysis")
    
    col_c, col_d = st.columns(2)
    with col_c:
        # 1. Acceptance rate by company tier
        fig4 = px.histogram(df, x="company_tier", color="status", barmode="group",
                            title="Acceptance Rate by Company Tier")
        st.plotly_chart(fig4, use_container_width=True)
        
    with col_d:
        # 2. Experience vs placement success
        avg_exp = df.groupby('status')['years_of_experience'].mean().reset_index()
        avg_exp['status'] = avg_exp['status'].map({0: 'Not Placed', 1: 'Placed'})

        fig = px.bar(
        avg_exp, 
        x='status', 
        y='years_of_experience',
        color='status',
        text_auto='.2f',  
        title="Average Years of Experience by Placement Status",
        labels={'years_of_experience': 'Average Experience (Years)', 'status': 'Placement Status'},
        color_discrete_map={'Not Placed': '#EF553B', 'Placed': '#00CC96'}
         )
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# TAB 3: Interview & Evaluation Analysis
# ------------------------------------------
with tab3:
    st.subheader("Interview & Evaluation Analysis")
    
    col_e, col_f = st.columns(2)
    with col_e:
        # 1. Interview score vs placement probability
       bins = [0, 60, 70, 80, 100]
       labels = ['<60', '60-70', '70-80', '80+']
       df['score_group'] = pd.cut(df['interview_avg_score'], bins=bins, labels=labels)

       # Plot
       fig, ax = plt.subplots(figsize=(8, 5))
       sns.countplot(data=df, x='score_group', hue='status', palette='Set2', ax=ax)
       ax.set_title('Placement Status by Interview Score Group')
       ax.set_xlabel('Interview Score Range')
       ax.set_ylabel('Number of Students')
       ax.legend(title='Status', labels=['Not Placed (0)', 'Placed (1)'])

       st.pyplot(fig)
        
    with col_f:
        # 2. Employability test score analysis (Aptitude & Technical)
        fig7 = px.scatter(df, x="aptitude_score", y="technical_score", color="status",
                          title="Employability Test Score Analysis (Aptitude vs Technical)")
        st.plotly_chart(fig7, use_container_width=True)