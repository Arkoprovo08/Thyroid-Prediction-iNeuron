import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTENC,RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('https://raw.githubusercontent.com/Arkoprovo08/iNeuron-Thyroid-Prediction/master/hypothyroid.csv')
columns_thyroid_csv = ['age', 'sex', 'on_thyroxine', 'query_on_thyroxine',
       'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
       'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured', 'TSH',
       'T3_measured', 'T3', 'TT4_measured', 'TT4', 'T4U_measured', 'T4U',
       'FTI_measured', 'FTI', 'TBG_measured', 'TBG', 'referral_source',
       'Class']
col = columns_thyroid_csv
df[col]=df[col].replace('?',np.nan)
df.drop('TBG',inplace=True,axis=1)
columns_to_drop = ['TT4','TSH','FTI','T4U','T3']
df.drop(columns_to_drop,inplace = True,axis=1)
#REMOVING ROWS HAVING NAN VALUES IN age AND sex
remove_rows = ['age','sex']
for i in remove_rows:
  df = df[df[i].notna()]
cols_to_binary = ['on_thyroxine', 'query_on_thyroxine',
       'on_antithyroid_medication', 'sick', 'pregnant', 'thyroid_surgery',
       'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
       'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH_measured',
       'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured',
       'TBG_measured']
for column in cols_to_binary:
  df[column] = df[column].map({'f' : 0, 't' : 1})

df['sex'] = df['sex'].map({'F' : 0, 'M' : 1})
df = pd.get_dummies(df, columns=['referral_source'])
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder().fit(df['Class'])

df['Class'] = encode.transform(df['Class'])
#importing RandomOverSampler to manage the imbalance
x = df.drop(['Class'],axis=1)
y = df['Class']
r = RandomOverSampler()
x_sampled,y_sampled  = r.fit_resample(x,y)
x_sampled = pd.DataFrame(data = x_sampled, columns = x.columns)
X_train, X_test, y_train, y_test = train_test_split(x_sampled, y_sampled, test_size=0.3, random_state=2)
rfc_model = RandomForestClassifier(n_estimators=1000)
rfc_model.fit(X_train, y_train)
rfc_pred = rfc_model.predict(X_test)



l = []
#Dashboard

st.title("Thyroid Prediction")
st.sidebar.header('Dashboard `iNeuron.ai`')
nav = st.sidebar.radio("Navigation",["Prediction"])

if nav == "Prediction":  
    name = st.text_input("Type your name")
    

    age = st.slider("Select your age",0,100,value=40)
    st.write(age)
    sex = st.selectbox("Enter sex",["Male","Female"])
    
    tsh = st.slider("Thyroid Stimulating Hormone Level",0.0, 530.0)
    st.write(tsh)
    tt4 = st.slider("Total Thyroxine TT4",2.0,430.0)
    st.write(tt4)
    fth = st.slider("Free Thyroxine Index",2.0,395.0)
    st.write(fth)
    t3m = st.slider("T3 Measure",0.0,11.0)
    st.write(t3m)
    t4u = st.slider("T4U Measure",0.0,2.0) 
    st.write(t4u)
    if tsh > 0.0:
      tsh = 1
    else:
      tsh = 0
    if tt4 > 2.0:
      tt4 = 1
    else:
      tt4 = 0
    if fth > 2.0:
      fth = 1
    else:
      fth = 0
    if t3m > 0.0:
      t3m = 1
    else:
      t3m = 0
    if t4u > 0.0:
      t4u = 1
    else:
      t4u = 0
      
    thm = st.selectbox("On Thyroxine Medication",["Yes","No"])
    hyp = st.selectbox("Hypopituitary Present",["Yes","No"])
    gtr = st.selectbox("Goitre Present",["Yes","No"])
    psy = st.selectbox("Pshylogical Symptoms Present",["Yes","No"])
    ant = st.selectbox("Anti-Thyroid Medication",["Yes","No"])
    
    hel = st.selectbox("Health Condition",["Sick","Fit"])
    
    pre = st.selectbox("Pregnant?",["Yes","No"])
    thy = st.selectbox("Thyroid surgery?",["Yes","No"])
    rad = st.selectbox("I-131 radiotherapy Treatment",["Yes","No"])
    qhyper = st.selectbox("Suspicion regarding Hyperthyroidism",["Yes","No"])
    qhypo = st.selectbox("Suspicion regarding Hypothyroidism",["Yes","No"])
    lit = st.selectbox("Lithium treatment?",["Yes","No"])
    
    ref = st.selectbox("Referal source",["STMW","SVHC","SVHD","SVI","Other"])
    stmw = 0
    svhc = 0
    svhd = 0
    svi = 0
    oth = 0

    if ref == 'STMW':
      stmw = 1
    if ref == 'SVHC':
      svhc = 1
    if ref == 'SVHD':
      svhd = 1
    if ref == 'SVI':
      svi = 1
    if ref == 'Other':
      oth = 1

    #st.subheader("Selected inputs are:")
    #st.write(tsh)
    if sex == 'Male':
      sex = 1
    else:
      sex = 0
    if hel == 'Sick':
       hel = 1
    else:
       hel = 0
    binary = [thm,hyp,gtr,psy,ant,pre,thy,rad,qhypo,qhyper,lit]
    for i in range(11):
      if binary[i] == 'Yes':
        binary[i] = 1
      else:
        binary[i] = 0
       
    thm = binary[0]
    hyp = binary[1]
    gtr = binary[2]
    psy = binary[3]
    ant = binary[4]
    pre = binary[5]
    thy = binary[6]
    rad = binary[7]
    qhypo = binary[8]
    qhyper = binary[9]
    lit = binary[10]

    if st.button("Predict"):
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        #user_input = [[15,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1]]
        user_input = [[age,sex,thm,0,ant,hel,pre,thy,rad,qhypo,qhyper,lit,gtr,0,hyp,psy,tsh,t3m,tt4,t4u,fth,0,stmw,svhc,svhd,svi,oth]]
        
        predictions = clf.predict(user_input)
        out = 'Hi'
        if predictions == 1:
          out = 'Negative'
        if predictions == 0:
          out = 'Compensated Hypothyroid'
        if predictions == 2:
          out = 'Primary Hypothyroid'
        if predictions == 3:
          out = 'Secondary Hypothyroid'
        st.success('Your Result is Ready')
        st.write("Predicted Output: ",out)
        l.append(user_input)
        st.write(l)
st.sidebar.markdown('''



---
Created by [Arkoprovo Ghosh](https://github.com/Arkoprovo08)
and [Pradipta Sharma](https://github.com/Pradipta19)
''')
