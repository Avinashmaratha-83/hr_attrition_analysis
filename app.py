import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

try:
    import plotly
except ImportError:
    install("plotly")

try:
    import sklearn
except ImportError:
    install("scikit-learn")

try:
    import pandas
except ImportError:
    install("pandas")

try:
    import numpy
except ImportError:
    install("numpy")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import warnings, io
warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HR Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="metric-container"] {
    background:white; border:1px solid #e9ecef;
    border-radius:12px; padding:16px;
}
.section-header {
    font-size:20px; font-weight:700; color:#1a1a2e;
    margin:24px 0 12px 0; padding-bottom:8px;
    border-bottom:2px solid #378ADD;
}
.insight-box  { background:#e8f4fd; border-left:4px solid #378ADD; padding:12px 16px; border-radius:0 8px 8px 0; margin:6px 0; font-size:14px; }
.warning-box  { background:#fff3cd; border-left:4px solid #EF9F27; padding:12px 16px; border-radius:0 8px 8px 0; margin:6px 0; font-size:14px; }
.success-box  { background:#d4edda; border-left:4px solid #1D9E75; padding:12px 16px; border-radius:0 8px 8px 0; margin:6px 0; font-size:14px; }
.risk-card    { border-radius:16px; padding:28px; text-align:center; margin:16px 0; }
</style>
""", unsafe_allow_html=True)

BLUE  = "#378ADD"; RED   = "#E24B4A"
TEAL  = "#1D9E75"; AMBER = "#EF9F27"
GRAY  = "#888780"; COLORS = [BLUE, RED, TEAL, AMBER, "#7F77DD"]

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def make_sample_data():
    np.random.seed(42); n = 1470
    depts = np.random.choice(
        ['Sales','Research & Development','Human Resources'], n, p=[0.30,0.62,0.08])
    role_map = {
        'Sales': ['Sales Executive','Sales Representative','Manager'],
        'Research & Development': ['Research Scientist','Laboratory Technician',
                                   'Healthcare Representative','Research Director','Manager'],
        'Human Resources': ['Human Resources','Manager'],
    }
    roles    = [np.random.choice(role_map[d]) for d in depts]
    overtime = np.random.choice(['Yes','No'], n, p=[0.28,0.72])
    age      = np.random.randint(18, 61, n)
    yrs_co   = np.clip(np.random.exponential(6, n).astype(int), 0, 40)
    income   = np.clip(np.random.normal(6500,3000,n), 1009, 20000).astype(int)
    job_sat  = np.random.choice([1,2,3,4], n, p=[0.19,0.20,0.30,0.31])
    env_sat  = np.random.choice([1,2,3,4], n, p=[0.16,0.22,0.35,0.27])
    wlb      = np.random.choice([1,2,3,4], n, p=[0.10,0.23,0.61,0.06])
    jlevel   = np.random.choice([1,2,3,4,5], n, p=[0.26,0.37,0.21,0.10,0.06])
    num_co   = np.clip(np.random.poisson(2.7,n), 0, 9)
    dist     = np.random.randint(1,30,n)
    edu      = np.random.choice([1,2,3,4,5], n, p=[0.12,0.19,0.41,0.17,0.11])
    stock    = np.random.choice([0,1,2,3], n, p=[0.29,0.44,0.14,0.13])
    marital  = np.random.choice(['Single','Married','Divorced'], n, p=[0.32,0.46,0.22])
    gender   = np.random.choice(['Male','Female'], n, p=[0.60,0.40])
    edu_f    = np.random.choice(
        ['Life Sciences','Medical','Marketing','Technical Degree','Human Resources','Other'],
        n, p=[0.41,0.27,0.11,0.09,0.04,0.08])
    logit = (-1.2 + 1.4*(overtime=='Yes') - 0.04*(age-30)
             + 0.8*(np.array(roles)=='Sales Representative')
             + 0.4*(depts=='Sales') - 0.06*yrs_co
             - 0.3*job_sat - 0.2*env_sat - 0.2*wlb
             + 0.15*num_co + 0.01*dist - 0.00008*income
             + 0.4*(marital=='Single') - 0.2*stock)
    prob = 1/(1+np.exp(-logit))
    attr = np.where(np.random.rand(n) < prob, 'Yes','No')
    df = pd.DataFrame({
        'Age':age,'Attrition':attr,'Department':depts,
        'DistanceFromHome':dist,'Education':edu,'EducationField':edu_f,
        'EnvironmentSatisfaction':env_sat,'Gender':gender,
        'JobLevel':jlevel,'JobRole':roles,'JobSatisfaction':job_sat,
        'MaritalStatus':marital,'MonthlyIncome':income,
        'NumCompaniesWorked':num_co,'OverTime':overtime,
        'PercentSalaryHike':np.random.randint(11,26,n),
        'PerformanceRating':np.random.choice([3,4],n,p=[0.85,0.15]),
        'StockOptionLevel':stock,'TotalWorkingYears':np.clip(age-18,0,40),
        'TrainingTimesLastYear':np.random.choice([0,1,2,3,4,5,6],n,p=[0.06,0.15,0.30,0.26,0.14,0.06,0.03]),
        'WorkLifeBalance':wlb,'YearsAtCompany':yrs_co,
        'YearsInCurrentRole':np.clip(np.random.exponential(3,n).astype(int),0,18),
        'YearsSinceLastPromotion':np.clip(np.random.poisson(2,n),0,15),
        'YearsWithCurrManager':np.clip(np.random.poisson(4,n),0,17),
    })
    return df

@st.cache_data
def load_data(file_bytes=None):
    if file_bytes:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = make_sample_data()
    drop = [c for c in df.columns if df[c].nunique()==1]
    df.drop(columns=drop, inplace=True, errors='ignore')
    df['AgeGroup']   = pd.cut(df['Age'], bins=[18,25,35,45,55,65],
                               labels=['18-25','26-35','36-45','46-55','56-65'])
    df['TenureBand'] = pd.cut(df['YearsAtCompany'], bins=[0,1,3,6,10,100],
                               labels=['<1yr','1-3yr','3-6yr','6-10yr','10+yr'])
    df['AttritionNum'] = (df['Attrition']=='Yes').astype(int)
    return df

@st.cache_data
def train_models(_df):
    df2 = _df.copy()
    drop_cols = ['Attrition','AttritionNum','AgeGroup','TenureBand']
    cat_cols = [c for c in df2.select_dtypes(include='object').columns if c not in drop_cols]
    le = LabelEncoder()
    for col in cat_cols:
        df2[col] = le.fit_transform(df2[col].astype(str))
    X = df2.drop(columns=drop_cols, errors='ignore')
    y = df2['AttritionNum']
    X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    mdls = {
        'Logistic Regression': LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=150,class_weight='balanced',random_state=42),
        'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150,random_state=42),
    }
    out = {}
    for name, mdl in mdls.items():
        Xtr = X_tr_sc if 'Logistic' in name else X_tr
        Xte = X_te_sc  if 'Logistic' in name else X_te
        mdl.fit(Xtr, y_tr)
        yp  = mdl.predict(Xte); ypr = mdl.predict_proba(Xte)[:,1]
        rep = classification_report(y_te, yp, output_dict=True)
        out[name] = dict(model=mdl, y_pred=yp, y_prob=ypr, y_test=y_te, X_test=X_te,
                         scaler=sc, feat_cols=X.columns.tolist(),
                         accuracy=rep['accuracy'], precision=rep['1']['precision'],
                         recall=rep['1']['recall'], f1=rep['1']['f1-score'],
                         auc=roc_auc_score(y_te,ypr))
    rf   = out['Random Forest']['model']
    fimp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return out, fimp

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 👥 HR Attrition App")
    st.caption("Data Analytics Portfolio Project")
    st.divider()
    uploaded = st.file_uploader("Upload CSV (IBM HR dataset)", type=['csv'],
                                 help="Download from Kaggle: IBM HR Analytics Attrition Dataset")
    file_bytes = uploaded.read() if uploaded else None
    df_raw = load_data(file_bytes)
    results, feat_imp = train_models(df_raw)
    best_name = max(results, key=lambda k: results[k]['auc'])

    st.divider()
    st.markdown("**Filters**")
    dept_opts   = ['All'] + sorted(df_raw['Department'].unique())
    sel_dept    = st.selectbox('Department', dept_opts)
    gender_opts = ['All'] + sorted(df_raw['Gender'].unique())
    sel_gender  = st.selectbox('Gender', gender_opts)
    age_min, age_max = int(df_raw['Age'].min()), int(df_raw['Age'].max())
    age_range = st.slider('Age range', age_min, age_max, (age_min, age_max))
    st.divider()
    st.markdown("**Pages**")
    page = st.radio('', ['Overview','EDA','ML Models','Predict Risk','Report'],
                    label_visibility='collapsed')

df = df_raw.copy()
if sel_dept   != 'All': df = df[df['Department']==sel_dept]
if sel_gender != 'All': df = df[df['Gender']==sel_gender]
df = df[(df['Age']>=age_range[0]) & (df['Age']<=age_range[1])]

def pct_attr(group_col, src=None):
    src = src if src is not None else df
    return (src.groupby(group_col, observed=True)['AttritionNum']
               .mean().mul(100).reset_index()
               .rename(columns={'AttritionNum':'Rate'}))

# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == 'Overview':
    st.title('👥 HR Attrition Analysis Dashboard')
    st.markdown('Explore employee attrition, build ML models, and predict risk — all in one place.')
    st.divider()

    attr_rate = df['AttritionNum'].mean()*100
    n_left    = df['AttritionNum'].sum()
    avg_ten   = df['YearsAtCompany'].mean()
    avg_inc   = df['MonthlyIncome'].mean()
    ot_rate   = (df[df['OverTime']=='Yes']['AttritionNum'].mean()*100) if 'OverTime' in df.columns else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric('Total employees',    f'{len(df):,}')
    c2.metric('Attrition rate',     f'{attr_rate:.1f}%', delta=f'{n_left} left', delta_color='inverse')
    c3.metric('Avg tenure',         f'{avg_ten:.1f} yrs')
    c4.metric('Avg monthly income', f'Rs.{avg_inc:,.0f}')
    c5.metric('Overtime attrition', f'{ot_rate:.1f}%',   delta='High risk', delta_color='inverse')

    st.markdown('<div class="section-header">Attrition at a glance</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])
    with col1:
        cnts = df['Attrition'].value_counts().reset_index()
        cnts.columns = ['Attrition','Count']
        fig = px.pie(cnts, values='Count', names='Attrition', hole=0.58,
                     color='Attrition', color_discrete_map={'No':BLUE,'Yes':RED})
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=0,r=0), height=270)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        dept_d = pct_attr('Department').sort_values('Rate')
        fig = px.bar(dept_d, x='Rate', y='Department', orientation='h',
                     text=dept_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                     color='Rate', color_continuous_scale=[[0,BLUE],[1,RED]])
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False, xaxis_title='Attrition rate (%)',
                          yaxis_title='', margin=dict(t=10,b=10), height=270)
        fig.update_xaxes(range=[0, dept_d['Rate'].max()*1.3])
        st.plotly_chart(fig, use_container_width=True)

    top_dept = dept_d.sort_values('Rate',ascending=False).iloc[0]
    ot_yes   = df[df['OverTime']=='Yes']['AttritionNum'].mean()*100 if 'OverTime' in df.columns else 0
    ot_no    = df[df['OverTime']=='No']['AttritionNum'].mean()*100  if 'OverTime' in df.columns else 0
    col1,col2,col3 = st.columns(3)
    col1.markdown(f'<div class="insight-box">📊 <b>{top_dept["Department"]}</b> has the highest attrition at <b>{top_dept["Rate"]:.1f}%</b></div>', unsafe_allow_html=True)
    col2.markdown(f'<div class="warning-box">⚠️ Overtime employees leave at <b>{ot_yes:.1f}%</b> vs <b>{ot_no:.1f}%</b> — a <b>{ot_yes/max(ot_no,0.1):.1f}x</b> difference</div>', unsafe_allow_html=True)
    col3.markdown(f'<div class="success-box">✅ Best model: <b>{best_name}</b> with AUC = <b>{results[best_name]["auc"]:.3f}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Attrition by tenure and age</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ten_d = pct_attr('TenureBand').sort_values('TenureBand')
        fig = px.bar(ten_d, x='TenureBand', y='Rate',
                     text=ten_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                     color='Rate', color_continuous_scale=[[0,TEAL],[1,RED]],
                     title='Attrition rate by tenure band')
        fig.update_traces(textposition='outside')
        fig.update_layout(coloraxis_showscale=False,
                          yaxis_title='Attrition rate (%)', xaxis_title='Years at company',
                          yaxis_range=[0, ten_d['Rate'].max()*1.35])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        age_d = pct_attr('AgeGroup')
        fig = px.line(age_d, x='AgeGroup', y='Rate', markers=True,
                      title='Attrition rate by age group',
                      color_discrete_sequence=[RED])
        fig.update_traces(line_width=2.5, marker_size=9)
        fig.update_layout(yaxis_title='Attrition rate (%)', xaxis_title='Age group')
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'EDA':
    st.title('🔍 Exploratory Data Analysis')
    st.caption(f'Showing {len(df):,} employees after filters')
    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(['Demographics','Job factors','Compensation','Correlations'])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            age_d = pct_attr('AgeGroup')
            fig = px.bar(age_d, x='AgeGroup', y='Rate',
                         text=age_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                         color='Rate', color_continuous_scale=[[0,BLUE],[1,RED]],
                         title='Attrition rate by age group')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False,
                              yaxis_title='Attrition rate (%)',
                              yaxis_range=[0, age_d['Rate'].max()*1.35])
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            mar_d = (df.groupby(['MaritalStatus','Gender'], observed=True)['AttritionNum']
                       .mean().mul(100).reset_index().rename(columns={'AttritionNum':'Rate'}))
            fig = px.bar(mar_d, x='MaritalStatus', y='Rate', color='Gender', barmode='group',
                         color_discrete_map={'Male':BLUE,'Female':RED},
                         text=mar_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                         title='Attrition by marital status & gender')
            fig.update_traces(textposition='outside')
            fig.update_layout(yaxis_title='Attrition rate (%)')
            st.plotly_chart(fig, use_container_width=True)
        fig = px.histogram(df, x='YearsAtCompany', color='Attrition', barmode='overlay',
                           nbins=25, color_discrete_map={'No':BLUE,'Yes':RED},
                           title='Tenure distribution: leavers vs stayers', opacity=0.72)
        fig.update_layout(xaxis_title='Years at company', yaxis_title='Count')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1,col2 = st.columns(2)
        with col1:
            role_d = pct_attr('JobRole').sort_values('Rate')
            fig = px.bar(role_d, x='Rate', y='JobRole', orientation='h',
                         text=role_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                         color='Rate', color_continuous_scale=[[0,BLUE],[1,RED]],
                         title='Attrition rate by job role')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False,
                              xaxis_title='Attrition rate (%)', yaxis_title='')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            ot_d = pct_attr('OverTime')
            fig = px.bar(ot_d, x='OverTime', y='Rate',
                         color='OverTime', color_discrete_map={'No':BLUE,'Yes':RED},
                         text=ot_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                         title='Overtime vs attrition rate')
            fig.update_traces(textposition='outside', showlegend=False)
            fig.update_layout(xaxis_title='Works overtime?',
                              yaxis_title='Attrition rate (%)',
                              yaxis_range=[0, ot_d['Rate'].max()*1.35])
            st.plotly_chart(fig, use_container_width=True)
        col1,col2 = st.columns(2)
        with col1:
            js_d = pct_attr('JobSatisfaction')
            js_d['JobSatisfaction'] = js_d['JobSatisfaction'].astype(str)
            fig = px.line(js_d, x='JobSatisfaction', y='Rate', markers=True,
                          title='Job satisfaction vs attrition',
                          color_discrete_sequence=[TEAL])
            fig.update_traces(line_width=2.5, marker_size=9)
            fig.update_layout(xaxis_title='Job satisfaction (1=low, 4=high)',
                              yaxis_title='Attrition rate (%)')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            wl_d = pct_attr('WorkLifeBalance')
            wl_d['WorkLifeBalance'] = wl_d['WorkLifeBalance'].astype(str)
            fig = px.line(wl_d, x='WorkLifeBalance', y='Rate', markers=True,
                          title='Work-life balance vs attrition',
                          color_discrete_sequence=[AMBER])
            fig.update_traces(line_width=2.5, marker_size=9)
            fig.update_layout(xaxis_title='Work-life balance (1=low, 4=high)',
                              yaxis_title='Attrition rate (%)')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.box(df, x='Attrition', y='MonthlyIncome', color='Attrition',
                     color_discrete_map={'No':BLUE,'Yes':RED},
                     title='Monthly income: stayers vs leavers', points='outliers')
        fig.update_layout(yaxis_title='Monthly income (Rs.)', xaxis_title='', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        col1,col2 = st.columns(2)
        with col1:
            fig = px.box(df, x='Department', y='MonthlyIncome', color='Attrition',
                         color_discrete_map={'No':BLUE,'Yes':RED},
                         title='Income by department & attrition')
            fig.update_layout(yaxis_title='Monthly income (Rs.)')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sk_d = pct_attr('StockOptionLevel')
            sk_d['StockOptionLevel'] = sk_d['StockOptionLevel'].astype(str)
            fig = px.bar(sk_d, x='StockOptionLevel', y='Rate',
                         text=sk_d['Rate'].apply(lambda x:f'{x:.1f}%'),
                         color='Rate', color_continuous_scale=[[0,TEAL],[1,BLUE]],
                         title='Stock option level vs attrition rate')
            fig.update_traces(textposition='outside')
            fig.update_layout(coloraxis_showscale=False,
                              xaxis_title='Stock option level (0=none, 3=high)',
                              yaxis_title='Attrition rate (%)')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        num_df = df.select_dtypes(include=[np.number])
        corr   = (num_df.corr()['AttritionNum'].drop('AttritionNum')
                        .sort_values(key=abs, ascending=False).head(14))
        fig = px.bar(corr.reset_index(), x='AttritionNum', y='index', orientation='h',
                     color='AttritionNum',
                     color_continuous_scale=[[0,BLUE],[0.5,'#f0f0f0'],[1,RED]],
                     title='Top 14 features correlated with attrition')
        fig.update_layout(xaxis_title='Pearson correlation', yaxis_title='',
                          coloraxis_showscale=False)
        fig.add_vline(x=0, line_color=GRAY, line_width=1)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="insight-box">🔵 Blue = negatively correlated (higher value → less attrition)&nbsp;&nbsp;|&nbsp;&nbsp;🔴 Red = positively correlated (higher value → more attrition)</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'ML Models':
    st.title('🤖 Machine Learning Models')
    st.caption('Trained on 80% of data — evaluated on 20% held-out test set.')
    st.divider()

    tab1, tab2, tab3 = st.tabs(['Model comparison','ROC & Confusion matrix','Feature importance'])

    with tab1:
        metrics = ['accuracy','precision','recall','f1','auc']
        labels  = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
        fig = go.Figure()
        for i,(name,res) in enumerate(results.items()):
            vals = [round(res[m],3) for m in metrics]
            fig.add_trace(go.Bar(name=name, x=labels, y=vals,
                                 marker_color=COLORS[i],
                                 text=[f'{v:.3f}' for v in vals],
                                 textposition='outside'))
        fig.update_layout(barmode='group', yaxis_range=[0,1.15],
                          yaxis_title='Score', title='Model performance comparison',
                          legend=dict(orientation='h', y=1.12))
        fig.add_hline(y=0.5, line_dash='dot', line_color=GRAY, opacity=0.4)
        st.plotly_chart(fig, use_container_width=True)

        rows = [{'Model':n, 'Accuracy':f'{r["accuracy"]:.3f}',
                 'Precision':f'{r["precision"]:.3f}', 'Recall':f'{r["recall"]:.3f}',
                 'F1':f'{r["f1"]:.3f}', 'AUC':f'{r["auc"]:.3f}',
                 'Best?':'✅' if n==best_name else ''} for n,r in results.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.markdown(f'<div class="success-box">✅ Best model: <b>{best_name}</b> — AUC = <b>{results[best_name]["auc"]:.3f}</b></div>', unsafe_allow_html=True)

    with tab2:
        fig = go.Figure()
        for i,(name,res) in enumerate(results.items()):
            fpr,tpr,_ = roc_curve(res['y_test'], res['y_prob'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                     name=f'{name} (AUC={res["auc"]:.3f})',
                                     line=dict(color=COLORS[i], width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random baseline',
                                  line=dict(color=GRAY, width=1.5, dash='dot')))
        fig.update_layout(title='ROC curves', xaxis_title='False positive rate',
                          yaxis_title='True positive rate',
                          legend=dict(x=0.55, y=0.1))
        st.plotly_chart(fig, use_container_width=True)

        sel = st.selectbox('Confusion matrix for:', list(results.keys()))
        cm  = confusion_matrix(results[sel]['y_test'], results[sel]['y_pred'])
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x='Predicted', y='Actual'),
                        x=['Stayed','Left'], y=['Stayed','Left'],
                        color_continuous_scale=[[0,'white'],[1,BLUE]],
                        title=f'Confusion matrix — {sel}')
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        top_n = st.slider('Show top N features', 5, 20, 15)
        fi    = feat_imp.head(top_n).reset_index()
        fi.columns = ['Feature','Importance']
        fi = fi.sort_values('Importance')
        fig = px.bar(fi, x='Importance', y='Feature', orientation='h',
                     color='Importance',
                     color_continuous_scale=[[0,BLUE],[1,RED]],
                     title=f'Top {top_n} attrition drivers (Random Forest)')
        fig.update_layout(coloraxis_showscale=False,
                          yaxis_title='', xaxis_title='Feature importance')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('### Top 5 attrition drivers')
        for i,f in enumerate(feat_imp.head(5).index, 1):
            st.markdown(f'**{i}. {f}** — importance: `{feat_imp[f]:.4f}`')

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Predict Risk':
    st.title('🎯 Predict Employee Attrition Risk')
    st.caption('Fill in the employee details and click Predict.')
    st.divider()

    col1,col2,col3 = st.columns(3)
    with col1:
        st.markdown('**Personal details**')
        age      = st.slider('Age', 18, 60, 30)
        gender   = st.selectbox('Gender', ['Male','Female'])
        marital  = st.selectbox('Marital status', ['Single','Married','Divorced'])
        edu      = st.selectbox('Education level', [1,2,3,4,5],
                                 format_func=lambda x:{1:'Below college',2:'College',
                                 3:'Bachelor',4:'Master',5:'Doctor'}[x])
        edu_f    = st.selectbox('Education field', ['Life Sciences','Medical','Marketing',
                                                     'Technical Degree','Human Resources','Other'])
        dist     = st.slider('Distance from home (km)', 1, 29, 5)
    with col2:
        st.markdown('**Job details**')
        dept     = st.selectbox('Department', sorted(df_raw['Department'].unique()))
        role     = st.selectbox('Job role', sorted(df_raw['JobRole'].unique()))
        jlevel   = st.selectbox('Job level', [1,2,3,4,5])
        overtime = st.selectbox('Overtime', ['No','Yes'])
        num_co   = st.slider('Companies worked at', 0, 9, 1)
        training = st.slider('Trainings last year', 0, 6, 2)
        stock    = st.selectbox('Stock option level', [0,1,2,3])
    with col3:
        st.markdown('**Satisfaction & pay**')
        job_sat  = st.slider('Job satisfaction (1-4)', 1, 4, 3)
        env_sat  = st.slider('Environment satisfaction (1-4)', 1, 4, 3)
        wlb      = st.slider('Work-life balance (1-4)', 1, 4, 3)
        income   = st.number_input('Monthly income (Rs.)', 1009, 20000, 5000, step=500)
        yrs_co   = st.slider('Years at company', 0, 40, 3)
        yrs_role = st.slider('Years in current role', 0, 18, 2)
        yrs_pr   = st.slider('Years since last promotion', 0, 15, 1)

    st.divider()
    if st.button('🔮 Predict attrition risk', use_container_width=True, type='primary'):
        # Encode using training data distribution
        df2 = df_raw.copy()
        drop_enc = ['Attrition','AttritionNum','AgeGroup','TenureBand']
        cat_cols = [c for c in df2.select_dtypes(include='object').columns if c not in drop_enc]
        le_map = {}
        for col in cat_cols:
            le_tmp = LabelEncoder()
            df2[col] = le_tmp.fit_transform(df2[col].astype(str))
            le_map[col] = le_tmp
        X_all = df2.drop(columns=drop_enc, errors='ignore')

        def enc(col, val):
            try: return int(le_map[col].transform([str(val)])[0])
            except: return 0

        row = {c: 0 for c in X_all.columns}
        row.update({
            'Age':age, 'DistanceFromHome':dist, 'Education':edu,
            'EnvironmentSatisfaction':env_sat, 'JobLevel':jlevel,
            'JobSatisfaction':job_sat, 'MonthlyIncome':income,
            'NumCompaniesWorked':num_co, 'StockOptionLevel':stock,
            'TotalWorkingYears':max(age-18,0), 'TrainingTimesLastYear':training,
            'WorkLifeBalance':wlb, 'YearsAtCompany':yrs_co,
            'YearsInCurrentRole':yrs_role, 'YearsSinceLastPromotion':yrs_pr,
            'YearsWithCurrManager':max(yrs_co-1,0),
            'PercentSalaryHike':14, 'PerformanceRating':3,
        })
        for col,val in [('Department',dept),('Gender',gender),('MaritalStatus',marital),
                         ('JobRole',role),('OverTime',overtime),('EducationField',edu_f)]:
            if col in row: row[col] = enc(col, val)

        X_inp = pd.DataFrame([row])[X_all.columns]
        sc2   = StandardScaler(); sc2.fit(X_all)
        X_sc  = sc2.transform(X_inp)

        probs = {}
        for name,res in results.items():
            Xin = X_sc if 'Logistic' in name else X_inp
            probs[name] = round(float(res['model'].predict_proba(Xin)[0][1]) * 100, 1)

        avg_prob = np.mean(list(probs.values()))
        if avg_prob >= 60:   color,label,emoji = RED,  'High Risk',   '🔴'
        elif avg_prob >= 30: color,label,emoji = AMBER, 'Medium Risk','🟡'
        else:                color,label,emoji = TEAL, 'Low Risk',    '🟢'

        cx,cy,cz = st.columns([1,2,1])
        with cy:
            st.markdown(f"""
            <div style="background:{color}18; border:2.5px solid {color};
                        border-radius:20px; padding:32px; text-align:center; margin:12px 0;">
                <div style="font-size:52px; margin-bottom:8px">{emoji}</div>
                <div style="font-size:42px; font-weight:800; color:{color}">{avg_prob:.1f}%</div>
                <div style="font-size:20px; font-weight:600; color:{color}; margin:4px 0">{label}</div>
                <div style="font-size:13px; color:#666; margin-top:6px">
                    Avg attrition probability across 3 models
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('### Prediction by model')
        c1,c2,c3 = st.columns(3)
        for col,(name,prob) in zip([c1,c2,c3], probs.items()):
            col.metric(name, f'{prob}%')

        st.markdown('### Risk factors')
        flags = []
        if overtime=='Yes':  flags.append(('Overtime',    'Working overtime is the strongest predictor of attrition'))
        if marital=='Single': flags.append(('Single',      'Single employees leave more frequently'))
        if yrs_co < 3:       flags.append(('Low tenure',  f'Only {yrs_co} yr(s) at company — highest-risk window'))
        if job_sat <= 2:     flags.append(('Low job satisfaction', f'Score {job_sat}/4'))
        if env_sat <= 2:     flags.append(('Low env satisfaction', f'Score {env_sat}/4'))
        if num_co >= 4:      flags.append(('Job hopper',   f'{num_co} companies previously'))
        if wlb <= 2:         flags.append(('Poor work-life balance', f'Score {wlb}/4'))
        if dist > 20:        flags.append(('Long commute', f'{dist} km from home'))
        if not flags:        flags.append(('No major risk flags', 'Stable profile across all key factors'))
        for factor,desc in flags:
            icon = '✅' if 'No major' in factor else '⚠️'
            st.markdown(f'- {icon} **{factor}**: {desc}')

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == 'Report':
    st.title('📄 Executive Report')
    st.divider()

    attr_rate = df['AttritionNum'].mean()*100
    dept_top  = pct_attr('Department').sort_values('Rate',ascending=False).iloc[0]
    role_top  = pct_attr('JobRole').sort_values('Rate',ascending=False).iloc[0]
    ot_yes    = df[df['OverTime']=='Yes']['AttritionNum'].mean()*100 if 'OverTime' in df.columns else 0
    ot_no     = df[df['OverTime']=='No']['AttritionNum'].mean()*100  if 'OverTime' in df.columns else 0
    top5      = feat_imp.head(5).index.tolist()

    report_md = f"""
## HR Attrition Analysis — Executive Summary

**Organisation size:** {len(df):,} employees &nbsp;|&nbsp;
**Attrition rate:** {attr_rate:.1f}% &nbsp;|&nbsp;
**Best model:** {best_name} (AUC = {results[best_name]['auc']:.3f})

---

### 1. Overview
This analysis covers {len(df):,} employees. The overall attrition rate is **{attr_rate:.1f}%**,
meaning approximately **{int(len(df)*attr_rate/100)} employees** are expected to leave annually.

---

### 2. Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | Highest-risk department | {dept_top['Department']} at {dept_top['Rate']:.1f}% |
| 2 | Highest-risk role | {role_top['JobRole']} at {role_top['Rate']:.1f}% |
| 3 | Overtime multiplier | Overtime: {ot_yes:.1f}% vs No overtime: {ot_no:.1f}% ({ot_yes/max(ot_no,0.1):.1f}x) |
| 4 | Early tenure risk | Most leavers have 1-3 years at company |
| 5 | Satisfaction impact | Low satisfaction employees leave at ~2x the rate |

---

### 3. ML Model Results

| Model | Accuracy | AUC | Recall | F1 |
|-------|----------|-----|--------|----|
{"".join(f'| {n} | {r["accuracy"]:.3f} | {r["auc"]:.3f} | {r["recall"]:.3f} | {r["f1"]:.3f} |' + chr(10) for n,r in results.items())}

**Top 5 predictors:** {', '.join(top5)}

---

### 4. Recommendations

1. **Control overtime** — Especially in {dept_top['Department']}. Introduce compensatory policies.
2. **Strengthen onboarding** — Mentoring and check-ins for employees in years 0-3.
3. **Target {dept_top['Department']}** — Quarterly engagement surveys, career-pathing workshops.

---
*Built with Python, Scikit-learn, Plotly & Streamlit.*
"""
    st.markdown(report_md)
    st.divider()
    st.markdown('### Download data')
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button('Download filtered dataset (CSV)', csv,
                       'hr_attrition_filtered.csv', 'text/csv',
                       use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption('HR Attrition Analysis App · Streamlit + Plotly + Scikit-learn · Portfolio Project')
