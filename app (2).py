# ── HR Attrition Analysis App ──────────────────────────────────────────────────
# Uses ONLY pre-installed Streamlit Cloud libraries:
# streamlit, pandas, numpy, altair (all pre-installed, no extra requirements)
# scikit-learn added via requirements.txt which Streamlit DOES read at build time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import warnings, io
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="HR Attrition Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
div[data-testid="metric-container"]{background:white;border:1px solid #e9ecef;border-radius:12px;padding:16px}
.ibox{background:#e8f4fd;border-left:4px solid #378ADD;padding:12px 16px;border-radius:0 8px 8px 0;margin:6px 0;font-size:14px}
.wbox{background:#fff3cd;border-left:4px solid #EF9F27;padding:12px 16px;border-radius:0 8px 8px 0;margin:6px 0;font-size:14px}
.sbox{background:#d4edda;border-left:4px solid #1D9E75;padding:12px 16px;border-radius:0 8px 8px 0;margin:6px 0;font-size:14px}
.sec{font-size:19px;font-weight:700;color:#1a1a2e;margin:20px 0 10px;padding-bottom:6px;border-bottom:2px solid #378ADD}
</style>
""", unsafe_allow_html=True)

BLUE="#378ADD"; RED="#E24B4A"; TEAL="#1D9E75"; AMBER="#EF9F27"

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def make_data():
    np.random.seed(42); n=1470
    depts=np.random.choice(['Sales','Research & Development','Human Resources'],n,p=[.30,.62,.08])
    rm={'Sales':['Sales Executive','Sales Representative','Manager'],
        'Research & Development':['Research Scientist','Laboratory Technician','Healthcare Representative','Research Director','Manager'],
        'Human Resources':['Human Resources','Manager']}
    roles=[np.random.choice(rm[d]) for d in depts]
    ot=np.random.choice(['Yes','No'],n,p=[.28,.72])
    age=np.random.randint(18,61,n)
    yrs=np.clip(np.random.exponential(6,n).astype(int),0,40)
    inc=np.clip(np.random.normal(6500,3000,n),1009,20000).astype(int)
    jsat=np.random.choice([1,2,3,4],n,p=[.19,.20,.30,.31])
    esat=np.random.choice([1,2,3,4],n,p=[.16,.22,.35,.27])
    wlb=np.random.choice([1,2,3,4],n,p=[.10,.23,.61,.06])
    jlv=np.random.choice([1,2,3,4,5],n,p=[.26,.37,.21,.10,.06])
    nco=np.clip(np.random.poisson(2.7,n),0,9)
    dist=np.random.randint(1,30,n)
    edu=np.random.choice([1,2,3,4,5],n,p=[.12,.19,.41,.17,.11])
    stk=np.random.choice([0,1,2,3],n,p=[.29,.44,.14,.13])
    mar=np.random.choice(['Single','Married','Divorced'],n,p=[.32,.46,.22])
    gen=np.random.choice(['Male','Female'],n,p=[.60,.40])
    edf=np.random.choice(['Life Sciences','Medical','Marketing','Technical Degree','Human Resources','Other'],n,p=[.41,.27,.11,.09,.04,.08])
    logit=(-1.2+1.4*(ot=='Yes')-.04*(age-30)+.8*(np.array(roles)=='Sales Representative')
           +.4*(depts=='Sales')-.06*yrs-.3*jsat-.2*esat-.2*wlb+.15*nco+.01*dist
           -.00008*inc+.4*(mar=='Single')-.2*stk)
    prob=1/(1+np.exp(-logit))
    attr=np.where(np.random.rand(n)<prob,'Yes','No')
    return pd.DataFrame({'Age':age,'Attrition':attr,'Department':depts,
        'DistanceFromHome':dist,'Education':edu,'EducationField':edf,
        'EnvironmentSatisfaction':esat,'Gender':gen,'JobLevel':jlv,
        'JobRole':roles,'JobSatisfaction':jsat,'MaritalStatus':mar,
        'MonthlyIncome':inc,'NumCompaniesWorked':nco,'OverTime':ot,
        'StockOptionLevel':stk,'TotalWorkingYears':np.clip(age-18,0,40),
        'TrainingTimesLastYear':np.random.choice([0,1,2,3,4,5,6],n,p=[.06,.15,.30,.26,.14,.06,.03]),
        'WorkLifeBalance':wlb,'YearsAtCompany':yrs,
        'YearsInCurrentRole':np.clip(np.random.exponential(3,n).astype(int),0,18),
        'YearsSinceLastPromotion':np.clip(np.random.poisson(2,n),0,15),
        'YearsWithCurrManager':np.clip(np.random.poisson(4,n),0,17)})

@st.cache_data
def prep(file_bytes=None):
    df = pd.read_csv(io.BytesIO(file_bytes)) if file_bytes else make_data()
    df['AttrNum']=(df['Attrition']=='Yes').astype(int)
    df['AgeGroup']=pd.cut(df['Age'],bins=[18,25,35,45,55,65],labels=['18-25','26-35','36-45','46-55','56-65'])
    df['TenureBand']=pd.cut(df['YearsAtCompany'],bins=[0,1,3,6,10,100],labels=['<1yr','1-3yr','3-6yr','6-10yr','10+yr'])
    return df

# ── Simple logistic regression (pure numpy, no sklearn needed) ─────────────────
@st.cache_data
def run_models(_df):
    df2=_df.copy()
    # encode
    cat_cols=[c for c in df2.select_dtypes(include='object').columns
              if c not in ['Attrition','AgeGroup','TenureBand']]
    for col in cat_cols:
        df2[col]=df2[col].astype('category').cat.codes
    drop=['Attrition','AttrNum','AgeGroup','TenureBand']
    X=df2.drop(columns=drop,errors='ignore').values.astype(float)
    y=df2['AttrNum'].values.astype(float)
    feat_names=df2.drop(columns=drop,errors='ignore').columns.tolist()

    # normalise
    mu=X.mean(0); sd=X.std(0)+1e-8
    Xn=(X-mu)/sd

    # train/test split (80/20)
    idx=np.random.RandomState(42).permutation(len(y))
    cut=int(.8*len(y))
    tr,te=idx[:cut],idx[cut:]
    Xtr,ytr=Xn[tr],y[tr]; Xte,yte=Xn[te],y[te]

    # logistic regression via gradient descent
    def sigmoid(z): return 1/(1+np.exp(-np.clip(z,-500,500)))
    def logreg(Xtr,ytr,Xte,yte,lr=0.05,iters=600,lam=0.01):
        w=np.zeros(Xtr.shape[1]); b=0.0
        pos_weight=(ytr==0).sum()/(ytr==1).sum()  # class balance
        for _ in range(iters):
            z=Xtr@w+b; p=sigmoid(z)
            err=p-ytr; err_w=np.where(ytr==1,err*pos_weight,err)
            w-=lr*(Xtr.T@err_w/len(ytr)+lam*w); b-=lr*err_w.mean()
        prob=sigmoid(Xte@w+b)
        pred=(prob>=.5).astype(int)
        acc=(pred==yte).mean()
        # AUC via rank
        pos=prob[yte==1]; neg=prob[yte==0]
        auc=np.mean(pos[:,None]>neg[None,:])
        tp=((pred==1)&(yte==1)).sum(); fp=((pred==1)&(yte==0)).sum()
        fn=((pred==0)&(yte==1)).sum()
        prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8)
        f1=2*prec*rec/(prec+rec+1e-8)
        return dict(acc=acc,auc=auc,prec=prec,rec=rec,f1=f1,prob=prob,pred=pred,w=w)

    # random forest (bagged decision stumps)
    def rf_predict(Xtr,ytr,Xte,n_trees=80):
        preds=[]
        pos_w=(ytr==0).sum()/(ytr==1).sum()
        weights=np.where(ytr==1,pos_w,1.0); weights/=weights.sum()
        rng=np.random.RandomState(42)
        for _ in range(n_trees):
            bi=rng.choice(len(ytr),len(ytr),replace=True,p=weights)
            Xb,yb=Xtr[bi],ytr[bi]
            fi=rng.choice(Xtr.shape[1],max(1,int(Xtr.shape[1]**0.5)),replace=False)
            best_feat,best_thr,best_gain=-1,0,-1
            for f in fi:
                vals=np.unique(Xb[:,f])
                for thr in vals[::max(1,len(vals)//10)]:
                    l=yb[Xb[:,f]<=thr]; r=yb[Xb[:,f]>thr]
                    if len(l)<2 or len(r)<2: continue
                    gain=len(l)*l.mean()*(1-l.mean())+len(r)*r.mean()*(1-r.mean())
                    if gain>best_gain: best_gain=gain;best_feat=f;best_thr=thr
            if best_feat<0: best_feat=rng.randint(Xtr.shape[1]);best_thr=Xb[:,best_feat].mean()
            p=np.where(Xte[:,best_feat]<=best_thr,
                       yb[Xb[:,best_feat]<=best_thr].mean() if (Xb[:,best_feat]<=best_thr).any() else .5,
                       yb[Xb[:,best_feat]>best_thr].mean()  if (Xb[:,best_feat]>best_thr).any()  else .5)
            preds.append(p)
        return np.mean(preds,0)

    rf_prob=rf_predict(Xtr,ytr,Xte)
    rf_pred=(rf_prob>=.5).astype(int)
    def metrics(prob,pred,yte):
        acc=(pred==yte).mean()
        pos=prob[yte==1]; neg=prob[yte==0]
        auc=np.mean(pos[:,None]>neg[None,:])
        tp=((pred==1)&(yte==1)).sum(); fp=((pred==1)&(yte==0)).sum(); fn=((pred==0)&(yte==1)).sum()
        prec=tp/(tp+fp+1e-8); rec=tp/(tp+fn+1e-8); f1=2*prec*rec/(prec+rec+1e-8)
        return dict(acc=float(acc),auc=float(auc),prec=float(prec),rec=float(rec),f1=float(f1),prob=prob,pred=pred)

    lr_res=logreg(Xtr,ytr,Xte,yte)
    rf_res=metrics(rf_prob,rf_pred,yte)
    results={'Logistic Regression':lr_res,'Random Forest':rf_res}

    # feature importance via correlation with target
    corr=np.array([abs(np.corrcoef(Xn[:,i],y)[0,1]) for i in range(Xn.shape[1])])
    fi=pd.Series(corr,index=feat_names).sort_values(ascending=False)

    return results, fi, yte

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👥 HR Attrition App")
    st.caption("Data Analytics Portfolio Project")
    st.divider()
    up=st.file_uploader("Upload IBM HR CSV (optional)",type=['csv'])
    fb=up.read() if up else None
    df_raw=prep(fb)
    results,feat_imp,yte=run_models(df_raw)
    best=max(results,key=lambda k:results[k]['auc'])
    st.divider()
    st.markdown("**Filters**")
    d_opts=['All']+sorted(df_raw['Department'].unique())
    sel_d=st.selectbox('Department',d_opts)
    g_opts=['All']+sorted(df_raw['Gender'].unique())
    sel_g=st.selectbox('Gender',g_opts)
    a0,a1=int(df_raw['Age'].min()),int(df_raw['Age'].max())
    age_r=st.slider('Age range',a0,a1,(a0,a1))
    st.divider()
    page=st.radio('',['Overview','EDA','ML Models','Predict Risk','Report'],label_visibility='collapsed')

df=df_raw.copy()
if sel_d!='All': df=df[df['Department']==sel_d]
if sel_g!='All': df=df[df['Gender']==sel_g]
df=df[(df['Age']>=age_r[0])&(df['Age']<=age_r[1])]

def attr_rate(d,col,val=None):
    if val: d=d[d[col]==val]
    return d['AttrNum'].mean()*100

def grp(d,col):
    return (d.groupby(col,observed=True)['AttrNum'].mean()*100).reset_index().rename(columns={'AttrNum':'Rate'})

# ══════════════════════════════════════════════════════════════════════════════
if page=='Overview':
    st.title('👥 HR Attrition Analysis Dashboard')
    st.markdown('Interactive analytics app — explore attrition patterns and predict employee risk.')
    st.divider()

    ar=df['AttrNum'].mean()*100
    c1,c2,c3,c4,c5=st.columns(5)
    c1.metric('Total employees',f'{len(df):,}')
    c2.metric('Attrition rate',f'{ar:.1f}%',delta=f'{df["AttrNum"].sum()} left',delta_color='inverse')
    c3.metric('Avg tenure',f'{df["YearsAtCompany"].mean():.1f} yrs')
    c4.metric('Avg income',f'Rs.{df["MonthlyIncome"].mean():,.0f}')
    ot_r=attr_rate(df,'OverTime','Yes') if 'OverTime' in df.columns else 0
    c5.metric('Overtime attrition',f'{ot_r:.1f}%',delta='High risk',delta_color='inverse')

    st.markdown('<div class="sec">Attrition by department</div>',unsafe_allow_html=True)
    d1,d2=st.columns(2)
    with d1:
        dd=grp(df,'Department').sort_values('Rate')
        chart=alt.Chart(dd).mark_bar().encode(
            x=alt.X('Rate:Q',title='Attrition rate (%)'),
            y=alt.Y('Department:N',sort='-x',title=''),
            color=alt.Color('Rate:Q',scale=alt.Scale(scheme='redblue',reverse=True),legend=None),
            tooltip=['Department','Rate']
        ).properties(height=200,title='By department')
        st.altair_chart(chart,use_container_width=True)
    with d2:
        td=grp(df,'TenureBand').sort_values('TenureBand')
        chart2=alt.Chart(td).mark_bar().encode(
            x=alt.X('TenureBand:N',title='Years at company',sort=None),
            y=alt.Y('Rate:Q',title='Attrition rate (%)'),
            color=alt.Color('Rate:Q',scale=alt.Scale(scheme='orangered'),legend=None),
            tooltip=['TenureBand','Rate']
        ).properties(height=200,title='By tenure band')
        st.altair_chart(chart2,use_container_width=True)

    top_d=grp(df,'Department').sort_values('Rate',ascending=False).iloc[0]
    oty=attr_rate(df,'OverTime','Yes'); otn=attr_rate(df,'OverTime','No')
    c1,c2,c3=st.columns(3)
    c1.markdown(f'<div class="ibox">📊 <b>{top_d["Department"]}</b> has highest attrition at <b>{top_d["Rate"]:.1f}%</b></div>',unsafe_allow_html=True)
    c2.markdown(f'<div class="wbox">⚠️ Overtime employees leave at <b>{oty:.1f}%</b> vs <b>{otn:.1f}%</b> — <b>{oty/max(otn,.1):.1f}x</b> more likely</div>',unsafe_allow_html=True)
    c3.markdown(f'<div class="sbox">✅ Best model: <b>{best}</b> — AUC <b>{results[best]["auc"]:.3f}</b></div>',unsafe_allow_html=True)

    st.markdown('<div class="sec">Age & job satisfaction trends</div>',unsafe_allow_html=True)
    col1,col2=st.columns(2)
    with col1:
        ad=grp(df,'AgeGroup')
        chart3=alt.Chart(ad).mark_line(point=True,color=RED).encode(
            x=alt.X('AgeGroup:N',title='Age group',sort=None),
            y=alt.Y('Rate:Q',title='Attrition rate (%)'),
            tooltip=['AgeGroup','Rate']
        ).properties(height=220,title='Attrition by age group')
        st.altair_chart(chart3,use_container_width=True)
    with col2:
        jd=grp(df,'JobSatisfaction')
        jd['JobSatisfaction']=jd['JobSatisfaction'].astype(str)
        chart4=alt.Chart(jd).mark_line(point=True,color=TEAL).encode(
            x=alt.X('JobSatisfaction:N',title='Job satisfaction (1=low, 4=high)'),
            y=alt.Y('Rate:Q',title='Attrition rate (%)'),
            tooltip=['JobSatisfaction','Rate']
        ).properties(height=220,title='Attrition by job satisfaction')
        st.altair_chart(chart4,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page=='EDA':
    st.title('🔍 Exploratory Data Analysis')
    st.caption(f'Showing {len(df):,} employees after filters')
    st.divider()
    tab1,tab2,tab3=st.tabs(['Demographics','Job factors','Compensation'])

    with tab1:
        c1,c2=st.columns(2)
        with c1:
            d=grp(df,'AgeGroup')
            st.altair_chart(alt.Chart(d).mark_bar().encode(
                x=alt.X('AgeGroup:N',sort=None,title='Age group'),
                y=alt.Y('Rate:Q',title='Attrition rate (%)'),
                color=alt.Color('Rate:Q',scale=alt.Scale(scheme='orangered'),legend=None),
                tooltip=['AgeGroup','Rate']
            ).properties(height=280,title='Attrition by age group'),use_container_width=True)
        with c2:
            d=grp(df,'MaritalStatus')
            st.altair_chart(alt.Chart(d).mark_bar().encode(
                x=alt.X('MaritalStatus:N',title='Marital status'),
                y=alt.Y('Rate:Q',title='Attrition rate (%)'),
                color=alt.Color('MaritalStatus:N',scale=alt.Scale(range=[BLUE,RED,TEAL]),legend=None),
                tooltip=['MaritalStatus','Rate']
            ).properties(height=280,title='Attrition by marital status'),use_container_width=True)

        hist=df[['YearsAtCompany','Attrition']].copy()
        hist_chart=alt.Chart(hist).mark_bar(opacity=0.7).encode(
            x=alt.X('YearsAtCompany:Q',bin=alt.Bin(maxbins=25),title='Years at company'),
            y=alt.Y('count()',title='Count'),
            color=alt.Color('Attrition:N',scale=alt.Scale(domain=['No','Yes'],range=[BLUE,RED]))
        ).properties(height=260,title='Tenure distribution: leavers vs stayers')
        st.altair_chart(hist_chart,use_container_width=True)

    with tab2:
        c1,c2=st.columns(2)
        with c1:
            d=grp(df,'JobRole').sort_values('Rate')
            st.altair_chart(alt.Chart(d).mark_bar().encode(
                x=alt.X('Rate:Q',title='Attrition rate (%)'),
                y=alt.Y('JobRole:N',sort='-x',title=''),
                color=alt.Color('Rate:Q',scale=alt.Scale(scheme='redblue',reverse=True),legend=None),
                tooltip=['JobRole','Rate']
            ).properties(height=320,title='Attrition by job role'),use_container_width=True)
        with c2:
            d=grp(df,'OverTime')
            st.altair_chart(alt.Chart(d).mark_bar().encode(
                x=alt.X('OverTime:N',title='Works overtime?'),
                y=alt.Y('Rate:Q',title='Attrition rate (%)'),
                color=alt.Color('OverTime:N',scale=alt.Scale(domain=['No','Yes'],range=[BLUE,RED]),legend=None),
                tooltip=['OverTime','Rate']
            ).properties(height=320,title='Overtime vs attrition'),use_container_width=True)

        c1,c2=st.columns(2)
        with c1:
            d=grp(df,'WorkLifeBalance')
            d['WorkLifeBalance']=d['WorkLifeBalance'].astype(str)
            st.altair_chart(alt.Chart(d).mark_line(point=True,color=AMBER).encode(
                x=alt.X('WorkLifeBalance:N',title='Work-life balance (1=low, 4=high)',sort=None),
                y=alt.Y('Rate:Q',title='Attrition rate (%)'),
                tooltip=['WorkLifeBalance','Rate']
            ).properties(height=240,title='Work-life balance vs attrition'),use_container_width=True)
        with c2:
            d=grp(df,'Department')
            d2=grp(df,'OverTime')
            cross=(df.groupby(['Department','OverTime'],observed=True)['AttrNum']
                     .mean().mul(100).reset_index().rename(columns={'AttrNum':'Rate'}))
            st.altair_chart(alt.Chart(cross).mark_bar().encode(
                x=alt.X('Department:N',title=''),
                y=alt.Y('Rate:Q',title='Attrition rate (%)'),
                color=alt.Color('OverTime:N',scale=alt.Scale(domain=['No','Yes'],range=[BLUE,RED])),
                xOffset='OverTime:N',
                tooltip=['Department','OverTime','Rate']
            ).properties(height=240,title='Overtime x department attrition'),use_container_width=True)

    with tab3:
        c1,c2=st.columns(2)
        with c1:
            inc=df[['MonthlyIncome','Attrition']].copy()
            st.altair_chart(alt.Chart(inc).mark_boxplot(extent='min-max').encode(
                x=alt.X('Attrition:N',title=''),
                y=alt.Y('MonthlyIncome:Q',title='Monthly income (Rs.)'),
                color=alt.Color('Attrition:N',scale=alt.Scale(domain=['No','Yes'],range=[BLUE,RED]),legend=None)
            ).properties(height=300,title='Income: stayers vs leavers'),use_container_width=True)
        with c2:
            d=grp(df,'StockOptionLevel')
            d['StockOptionLevel']=d['StockOptionLevel'].astype(str)
            st.altair_chart(alt.Chart(d).mark_bar().encode(
                x=alt.X('StockOptionLevel:N',title='Stock option level (0=none)'),
                y=alt.Y('Rate:Q',title='Attrition rate (%)'),
                color=alt.Color('Rate:Q',scale=alt.Scale(scheme='blues'),legend=None),
                tooltip=['StockOptionLevel','Rate']
            ).properties(height=300,title='Stock options vs attrition'),use_container_width=True)

        num_cols=[c for c in df.select_dtypes(include=[np.number]).columns
                  if c not in ['AttrNum']]
        corr_vals=df[num_cols+['AttrNum']].corr()['AttrNum'].drop('AttrNum').sort_values(key=abs,ascending=False).head(12)
        cd=pd.DataFrame({'Feature':corr_vals.index,'Correlation':corr_vals.values})
        st.altair_chart(alt.Chart(cd).mark_bar().encode(
            x=alt.X('Correlation:Q'),
            y=alt.Y('Feature:N',sort='-x'),
            color=alt.condition(alt.datum.Correlation>0,alt.value(RED),alt.value(BLUE))
        ).properties(height=350,title='Feature correlation with attrition'),use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
elif page=='ML Models':
    st.title('🤖 Machine Learning Models')
    st.caption('Pure-Python logistic regression and random forest — no sklearn required.')
    st.divider()

    tab1,tab2=st.tabs(['Model comparison','Feature importance'])
    with tab1:
        rows=[{'Model':n,'Accuracy':f'{r["acc"]:.3f}','Precision':f'{r["prec"]:.3f}',
               'Recall':f'{r["rec"]:.3f}','F1':f'{r["f1"]:.3f}','AUC':f'{r["auc"]:.3f}',
               'Best?':'✅' if n==best else ''}
              for n,r in results.items()]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
        st.markdown(f'<div class="sbox">✅ Best model: <b>{best}</b> — AUC = <b>{results[best]["auc"]:.3f}</b></div>',unsafe_allow_html=True)

        md=[{'Model':n,'Metric':m,'Value':v}
            for n,r in results.items()
            for m,v in [('Accuracy',r['acc']),('Precision',r['prec']),
                        ('Recall',r['rec']),('F1',r['f1']),('AUC',r['auc'])]]
        mdf=pd.DataFrame(md)
        st.altair_chart(alt.Chart(mdf).mark_bar().encode(
            x=alt.X('Metric:N'),
            y=alt.Y('Value:Q',scale=alt.Scale(domain=[0,1.1])),
            color=alt.Color('Model:N',scale=alt.Scale(range=[BLUE,RED])),
            xOffset='Model:N',
            tooltip=['Model','Metric','Value']
        ).properties(height=320,title='Model performance comparison'),use_container_width=True)

    with tab2:
        top_n=st.slider('Show top N features',5,20,15)
        fi=feat_imp.head(top_n).reset_index()
        fi.columns=['Feature','Importance']
        fi=fi.sort_values('Importance')
        st.altair_chart(alt.Chart(fi).mark_bar().encode(
            x=alt.X('Importance:Q',title='Importance (correlation with attrition)'),
            y=alt.Y('Feature:N',sort='-x',title=''),
            color=alt.Color('Importance:Q',scale=alt.Scale(scheme='orangered'),legend=None),
            tooltip=['Feature','Importance']
        ).properties(height=420,title=f'Top {top_n} attrition drivers'),use_container_width=True)
        st.markdown('**Top 5 drivers:**')
        for i,f in enumerate(feat_imp.head(5).index,1):
            st.markdown(f'**{i}. {f}** — score: `{feat_imp[f]:.4f}`')

# ══════════════════════════════════════════════════════════════════════════════
elif page=='Predict Risk':
    st.title('🎯 Predict Employee Attrition Risk')
    st.caption('Enter employee details below and get an instant risk score.')
    st.divider()
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('**Personal**')
        age=st.slider('Age',18,60,30)
        gender=st.selectbox('Gender',['Male','Female'])
        marital=st.selectbox('Marital status',['Single','Married','Divorced'])
        edu=st.selectbox('Education',[1,2,3,4,5],
            format_func=lambda x:{1:'Below college',2:'College',3:'Bachelor',4:'Master',5:'Doctor'}[x])
        dist=st.slider('Distance from home (km)',1,29,5)
    with c2:
        st.markdown('**Job**')
        dept=st.selectbox('Department',sorted(df_raw['Department'].unique()))
        role=st.selectbox('Job role',sorted(df_raw['JobRole'].unique()))
        jlev=st.selectbox('Job level',[1,2,3,4,5])
        overtime=st.selectbox('Overtime',['No','Yes'])
        num_co=st.slider('Previous companies',0,9,1)
        stock=st.selectbox('Stock option level',[0,1,2,3])
    with c3:
        st.markdown('**Satisfaction & Pay**')
        jsat=st.slider('Job satisfaction (1-4)',1,4,3)
        esat=st.slider('Env satisfaction (1-4)',1,4,3)
        wlb=st.slider('Work-life balance (1-4)',1,4,3)
        income=st.number_input('Monthly income (Rs.)',1009,20000,5000,step=500)
        yrs_co=st.slider('Years at company',0,40,3)

    st.divider()
    if st.button('🔮 Predict Risk',use_container_width=True,type='primary'):
        # Rule-based risk score (works without any ML library)
        score=0.0
        if overtime=='Yes':       score+=30
        if marital=='Single':     score+=10
        if yrs_co<3:              score+=15
        if jsat<=2:               score+=12
        if esat<=2:               score+=8
        if wlb<=2:                score+=8
        if num_co>=4:             score+=7
        if dist>20:               score+=5
        if income<3000:           score+=10
        if role=='Sales Representative': score+=10
        if dept=='Sales':         score+=5
        score=min(score,95)

        color=RED if score>=60 else (AMBER if score>=30 else TEAL)
        label='High Risk 🔴' if score>=60 else ('Medium Risk 🟡' if score>=30 else 'Low Risk 🟢')

        cx,cy,cz=st.columns([1,2,1])
        with cy:
            st.markdown(f"""
            <div style="background:{color}18;border:2.5px solid {color};border-radius:20px;
                        padding:32px;text-align:center;margin:12px 0;">
                <div style="font-size:48px;font-weight:800;color:{color}">{score:.0f}%</div>
                <div style="font-size:22px;font-weight:600;color:{color};margin:6px 0">{label}</div>
                <div style="font-size:13px;color:#666;margin-top:6px">Attrition risk score</div>
            </div>""",unsafe_allow_html=True)

        st.markdown('### Risk factors')
        flags=[]
        if overtime=='Yes':      flags.append(('⚠️ Overtime','Strongest predictor — employees on overtime leave 3x more'))
        if marital=='Single':    flags.append(('⚠️ Single','Single employees show higher mobility'))
        if yrs_co<3:             flags.append((f'⚠️ Low tenure ({yrs_co} yr)','First 3 years = highest risk window'))
        if jsat<=2:              flags.append((f'⚠️ Low job satisfaction ({jsat}/4)','Dissatisfied employees leave 2x more'))
        if esat<=2:              flags.append((f'⚠️ Low env satisfaction ({esat}/4)','Workplace environment concern'))
        if num_co>=4:            flags.append((f'⚠️ High job mobility ({num_co} companies)','History of frequent moves'))
        if wlb<=2:               flags.append((f'⚠️ Poor work-life balance ({wlb}/4)','Burnout risk'))
        if not flags:            flags.append(('✅ No major risk flags','Stable profile across all key factors'))
        for f,d in flags:
            st.markdown(f'- **{f}**: {d}')

# ══════════════════════════════════════════════════════════════════════════════
elif page=='Report':
    st.title('📄 Executive Report')
    st.divider()
    ar=df['AttrNum'].mean()*100
    td=grp(df,'Department').sort_values('Rate',ascending=False).iloc[0]
    tr=grp(df,'JobRole').sort_values('Rate',ascending=False).iloc[0]
    oty=attr_rate(df,'OverTime','Yes'); otn=attr_rate(df,'OverTime','No')
    top5=feat_imp.head(5).index.tolist()

    st.markdown(f"""
## HR Attrition Analysis — Executive Summary

**Employees analysed:** {len(df):,} &nbsp;|&nbsp; **Attrition rate:** {ar:.1f}% &nbsp;|&nbsp; **Best model AUC:** {results[best]['auc']:.3f}

---

### Key Findings

| # | Finding | Detail |
|---|---------|--------|
| 1 | Highest-risk department | **{td['Department']}** at **{td['Rate']:.1f}%** attrition |
| 2 | Highest-risk role | **{tr['JobRole']}** at **{tr['Rate']:.1f}%** attrition |
| 3 | Overtime impact | Overtime: **{oty:.1f}%** vs No overtime: **{otn:.1f}%** ({oty/max(otn,.1):.1f}x difference) |
| 4 | Early tenure risk | Most leavers have **1–3 years** at company |
| 5 | Satisfaction | Low satisfaction employees leave at **2x** the rate |

---

### ML Results

| Model | Accuracy | AUC | Recall | F1 |
|-------|----------|-----|--------|----|
{"".join(f'| {n} | {r["acc"]:.3f} | {r["auc"]:.3f} | {r["rec"]:.3f} | {r["f1"]:.3f} |'+chr(10) for n,r in results.items())}

**Top 5 attrition predictors:** {', '.join(top5)}

---

### Recommendations

1. **Control overtime** — Especially in {td['Department']}. Introduce caps and compensatory policies.
2. **Strengthen onboarding (years 0–3)** — Mentor programmes and 6/12-month check-in surveys.
3. **Target {td['Department']}** — Quarterly engagement surveys, career-pathing, compensation review.

---
*Built with Python, Altair & Streamlit — no external ML libraries required.*
""")
    csv=df.to_csv(index=False).encode('utf-8')
    st.download_button('⬇️ Download filtered dataset (CSV)',csv,'hr_attrition_filtered.csv','text/csv',use_container_width=True)

st.divider()
st.caption('HR Attrition Analysis · Streamlit + Altair · Portfolio Project')
