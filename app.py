from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title='ChurnIQ | Telecom Analytics', page_icon='📡', layout='wide')

@st.cache_resource
def get_model(): return joblib.load(BASE / 'churn_model.joblib')
@st.cache_data
def get_data():
    d = pd.read_csv(BASE / 'data.csv')
    d['TotalCharges'] = pd.to_numeric(d['TotalCharges'], errors='coerce')
    return d

try:
    model = get_model(); df = get_data(); metrics = json.loads((BASE/'metrics.json').read_text())
except Exception:
    st.error('Model files are missing. Run `python train.py` first, then restart the app.')
    st.stop()

st.markdown('''<style>
.block-container{padding-top:1.6rem;padding-bottom:2rem}.main{background:#0b1020}
[data-testid="stMetric"]{background:linear-gradient(135deg,#171f36,#11182b);padding:18px;border:1px solid #283653;border-radius:14px}
[data-testid="stSidebar"]{background:#10182a}.stTabs [data-baseweb="tab"]{font-weight:600}
.hero{padding:24px 28px;border-radius:18px;background:linear-gradient(110deg,#172b4d,#35245c);border:1px solid #34466d;margin-bottom:22px}
.hero h1{margin:0;color:white;font-size:2rem}.hero p{color:#c6d4f2;margin:8px 0 0}
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## 📡 ChurnIQ')
    st.caption('Telecom Customer Intelligence')
    page = st.radio('NAVIGATION', ['Overview', 'Predict Churn', 'Model Performance'])
    st.divider(); st.caption('ML model: Random Forest')
    st.caption('Dataset: Telco Customer Churn')

st.markdown('<div class="hero"><h1>Telecom Customer Churn Intelligence</h1><p>Understand customer behavior. Identify churn risk. Make data-driven retention decisions.</p></div>', unsafe_allow_html=True)

if page == 'Overview':
    total=len(df); churn=int((df.Churn=='Yes').sum()); rate=churn/total*100
    a,b,c,d=st.columns(4)
    a.metric('Total Customers',f'{total:,}')
    b.metric('Churned Customers',f'{churn:,}')
    c.metric('Churn Rate',f'{rate:.1f}%')
    d.metric('Model Accuracy',f"{metrics['accuracy']*100:.1f}%")
    st.markdown('### Customer insights')
    left,right=st.columns(2)
    with left:
        contract = df.groupby(['Contract','Churn']).size().reset_index(name='Customers')
        fig=px.bar(contract,x='Contract',y='Customers',color='Churn',barmode='group',title='Churn by Contract Type',color_discrete_map={'Yes':'#ef6675','No':'#39c6a5'})
        st.plotly_chart(fig,use_container_width=True)
    with right:
        pay=df.groupby(['PaymentMethod','Churn']).size().reset_index(name='Customers')
        fig=px.bar(pay,x='PaymentMethod',y='Customers',color='Churn',barmode='group',title='Churn by Payment Method',color_discrete_map={'Yes':'#ef6675','No':'#39c6a5'})
        fig.update_layout(xaxis_tickangle=-20)
        st.plotly_chart(fig,use_container_width=True)
    left,right=st.columns(2)
    with left:
        fig=px.histogram(df,x='tenure',color='Churn',nbins=24,barmode='overlay',opacity=.78,title='Customer Tenure Distribution',color_discrete_map={'Yes':'#ef6675','No':'#39c6a5'})
        st.plotly_chart(fig,use_container_width=True)
    with right:
        fig=px.box(df,x='Churn',y='MonthlyCharges',color='Churn',title='Monthly Charges vs Churn',color_discrete_map={'Yes':'#ef6675','No':'#39c6a5'})
        st.plotly_chart(fig,use_container_width=True)
    st.info('Business observation: compare month-to-month contracts, shorter tenure, and service-support patterns to identify customer segments that may need retention offers.')

elif page == 'Predict Churn':
    st.markdown('### Customer churn prediction')
    st.write('Enter a customer profile. The trained model will estimate the probability of churn.')
    with st.form('prediction_form'):
        c1,c2,c3=st.columns(3)
        with c1:
            gender=st.selectbox('Gender',['Female','Male']); senior=st.selectbox('Senior Citizen',[0,1]); partner=st.selectbox('Has Partner',['Yes','No']); dependents=st.selectbox('Has Dependents',['Yes','No']); tenure=st.slider('Tenure (months)',0,72,12)
        with c2:
            phone=st.selectbox('Phone Service',['Yes','No']); lines=st.selectbox('Multiple Lines',['No','Yes','No phone service']); internet=st.selectbox('Internet Service',['DSL','Fiber optic','No']); security=st.selectbox('Online Security',['No','Yes','No internet service']); backup=st.selectbox('Online Backup',['No','Yes','No internet service'])
        with c3:
            device=st.selectbox('Device Protection',['No','Yes','No internet service']); support=st.selectbox('Tech Support',['No','Yes','No internet service']); tv=st.selectbox('Streaming TV',['No','Yes','No internet service']); movies=st.selectbox('Streaming Movies',['No','Yes','No internet service']); contract=st.selectbox('Contract',['Month-to-month','One year','Two year'])
        c1,c2,c3=st.columns(3)
        with c1: paper=st.selectbox('Paperless Billing',['Yes','No'])
        with c2: payment=st.selectbox('Payment Method',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
        with c3: monthly=st.number_input('Monthly Charges ($)',min_value=0.0,max_value=200.0,value=70.0,step=1.0)
        total_charges=st.number_input('Total Charges ($)',min_value=0.0,max_value=10000.0,value=float(monthly*tenure),step=10.0)
        submitted=st.form_submit_button('🔍 Predict Churn Risk',use_container_width=True)
    if submitted:
        row=pd.DataFrame([{'gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':dependents,'tenure':tenure,'PhoneService':phone,'MultipleLines':lines,'InternetService':internet,'OnlineSecurity':security,'OnlineBackup':backup,'DeviceProtection':device,'TechSupport':support,'StreamingTV':tv,'StreamingMovies':movies,'Contract':contract,'PaperlessBilling':paper,'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total_charges}])
        probability=float(model.predict_proba(row)[0][1]); prediction=int(probability>=0.5)
        st.divider(); st.markdown('### Prediction result')
        x,y=st.columns([1,2])
        with x:
            st.metric('Churn Probability',f'{probability*100:.1f}%')
            st.metric('Predicted Class','Likely to Churn' if prediction else 'Likely to Stay')
        with y:
            if probability>=.65: st.error('HIGH RISK — Prioritize this customer for retention outreach.')
            elif probability>=.35: st.warning('MEDIUM RISK — Monitor engagement and consider a targeted offer.')
            else: st.success('LOW RISK — Customer is less likely to churn based on this model.')
            fig=go.Figure(go.Indicator(mode='gauge+number',value=probability*100,title={'text':'Estimated churn risk'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#ef6675'},'steps':[{'range':[0,35],'color':'#173c37'},{'range':[35,65],'color':'#514222'},{'range':[65,100],'color':'#4a2632'}]}))
            fig.update_layout(height=230,margin=dict(l=20,r=20,t=45,b=10))
            st.plotly_chart(fig,use_container_width=True)
        st.caption('This is a statistical prediction, not a guarantee. Risk bands are illustrative thresholds for the demo.')

else:
    st.markdown('### Model evaluation')
    st.write('Metrics are calculated on a held-out test set that was not used to fit the model.')
    a,b,c,d,e=st.columns(5)
    for col,label,key in zip([a,b,c,d,e],['Accuracy','Precision','Recall','F1 Score','ROC-AUC'],['accuracy','precision','recall','f1_score','roc_auc']): col.metric(label,f"{metrics[key]*100:.1f}%")
    st.caption(f"Training rows: {metrics['train_rows']:,} · Test rows: {metrics['test_rows']:,} · Algorithm: Random Forest")
    cm=metrics['confusion_matrix']
    fig=px.imshow(cm,text_auto=True,x=['Predicted Retained','Predicted Churn'],y=['Actual Retained','Actual Churn'],color_continuous_scale='Blues',title='Confusion Matrix')
    st.plotly_chart(fig,use_container_width=True)
    report=metrics['classification_report']
    st.markdown('#### Class-wise performance')
    st.dataframe(pd.DataFrame(report).T.round(3),use_container_width=True)
    st.warning('Accuracy alone is not enough for churn prediction. Recall for the churn class is important because missing a customer who is about to leave can be costly.')
