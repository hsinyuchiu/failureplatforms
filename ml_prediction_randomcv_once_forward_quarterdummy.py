# -*- coding: utf-8 -*-


import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import det_curve
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import uniform
from sklearn import preprocessing
from scipy import stats
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
#from numpy_ext import rolling_apply as rolling_apply_ext
path='D:\\Google drive\\P2P platform failure'
data=pd.read_excel(path+'\\Data 原始檔-Total_rename.xlsx')
'''
data=pd.read_csv(path+'\\Sample Data-complete 2021 all-more than 9.csv')
data[' Registered capital '].astype('int64')
data=data[['Operating months','Address',' Registered capital ','Private','Automatic bidding',\
           'Car loan','Personal credit loan','Corporate credit','Other loans','Multiple loans',\
               'Borrowing fee','Stored fee','Withdrawal fee','interest rate',\
                   'Third party guarantee','Bank guarantee','Risk reserve','Platform advance payment','Financing guarantee','Bank deposit','Other guarantee','No guarantee',\
                       'Join the association','Accepting VC','Third party credit','Equity listing','Company license','Business Permits','No supervised feature','Operation']]
data=data.rename(columns={'Operating months':'The number of months of operation',\
                           ' Registered capital ':'Registered capital','Private':'State-run enterprise',\
                               'Automatic bidding':'Auto bidding','Persoal credit loan':'Personal credit',\
                                   'Borrowing fee':'Borrow fee','Stored fee':'Deposit fee','interest rate':'Average interest rate',\
                                       'Risk reserve':'Risk margin','Platform advance payment':'Capital advance processing mechanism',\
                                           'Bank deposit':'Bank deposit management',\
                                               'Join the association':'NIFA membership','Accepting VC':'Acceptance venture capital',\
                                                   'Third party credit':'Third Party Supervisory Association',\
                                                       'Equity listing':'Listed Company','Business Permits':'Operation Permit',\
                                                           'No supervised feature':'No supervisory mechanism'})
'''
#ds=data.describe().T
#ds.to_excel(path+'\\summary.xlsx')
data=data[data['NoMO']>9]
data=data.dropna()
X=data[data.columns.drop('Operating Status')]
y=data['Operating Status']


for i in range(1,18,4):
    X['Q'+str(i)]=0
    X['Q'+str(i+1)]=0
    X['Q'+str(i+2)]=0
    X['Q'+str(i+3)]=0
    X['Q'+str(i)][(X['Year']==2014+i//4) & (X['Month']<=3)]=1
    X['Q'+str(i+1)][(X['Year']==2014+i//4) & (X['Month']>=4) & (X['Month']<=6)]=1
    X['Q'+str(i+2)][(X['Year']==2014+i//4) & (X['Month']>=7) & (X['Month']<=9)]=1
    X['Q'+str(i+3)][(X['Year']==2014+i//4) & (X['Month']>=10) & (X['Month']<=12)]=1

X=X.drop(columns=['Year','Month','day'])
X=X.drop(columns='Q20')

res=stats.spearmanr(X,y)
correlate=res.correlation
correlate=pd.DataFrame(correlate,index=list(X.columns)+['Operating Status'],columns=list(X.columns)+['Operating Status'])
correlate.to_excel(path+'\\corr_table_new.xlsx')



X[['NoMO','Registered Capital','AIR']]=preprocessing.normalize(X[['NoMO','Registered Capital','AIR']])


seq_var=np.abs(correlate['Operating Status'])
seq_var=seq_var.sort_values(ascending=False)
seq_var=seq_var.drop('Operating Status')
X=X[list(seq_var.index)]


#X=X[X.columns[1:]]



    

'''
X=data[data.columns.drop('Operation')]
y=data['Operation']

res=stats.spearmanr(X,y)
correlate=res.correlation
correlate=pd.DataFrame(correlate,index=list(X.columns)+['Operation'],columns=list(X.columns)+['Operation'])
correlate.to_excel(path+'\\corr_table.xlsx')
#res.statistic

X[['The number of months of operation','Registered capital','Average interest rate']]=preprocessing.normalize(X[['The number of months of operation','Registered capital','Average interest rate']])

seq_var=np.abs(correlate['Operation'])
seq_var=seq_var.sort_values(ascending=False)
seq_var=seq_var.drop('Operation')
X=X[list(seq_var.index)]

#X=X[X.columns[1:]]
'''


    
X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


model_lr=LogisticRegression(random_state=0,max_iter=1000)
param=dict(C=uniform(loc=0,scale=300),penalty=['l2'],solver=['newton-cg'])
clf_lr = RandomizedSearchCV(model_lr, param, random_state=0,n_iter=1000)
search_lr = clf_lr.fit(X_train_tmp,y_train)
param_best_lr=search_lr.best_params_
#uniform範圍[loc, loc+scale]

#randomized cv 會考慮分層抽樣stratifiedKfold及default的5 fold
#分層抽樣會自動考慮unbalanced data 
 
model_svm=svm.SVC(probability=True, random_state=0)
#uniform範圍[loc, loc+scale]
param=dict(C=uniform(loc=0,scale=300),gamma=uniform(loc=0, scale=100),kernel=['rbf'])
clf_svm = RandomizedSearchCV(model_svm, param, random_state=0,n_iter=1000)
search_svm = clf_svm.fit(X_train_tmp,y_train)
param_best_svm=search_svm.best_params_

model_rf=RandomForestClassifier(random_state=0)
param=dict(max_depth=[10,20,30,40,50,60,70,80,90,100,None],\
           min_samples_split=[2,5,10],\
               min_samples_leaf=[1,2,4],\
                   max_features=['sqrt','log2',None],\
                       n_estimators=[10,50,100,200,300,400,500],\
                           bootstrap=[True, False])
clf_rf = RandomizedSearchCV(model_rf, param, random_state=0,n_iter=1000)
search_rf = clf_rf.fit(X_train_tmp,y_train)
param_best_rf=search_rf.best_params_ 

model_ann=MLPClassifier(random_state=0,max_iter=50000)
param = {
    'hidden_layer_sizes': [(5,),(10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [1e-5,0.0001,0.001,0.01,0.1,1,10,100],
    #'learning_rate': ['constant','adaptive'],
}
#clf_ann = RadonmizedSearchCV(model_ann, param, random_state=0,n_iter=1000)
clf_ann = GridSearchCV(model_ann, param)
search_ann = clf_ann.fit(X_train_tmp,y_train)
param_best_ann=search_ann.best_params_


model_xgb=XGBClassifier(random_state=0)
param = {'n_estimators':[10,20,50,100,200,300],\
         'max_depth':[1,5,10,20],\
             'learning_rate':[0.001,0.01,0.05,0.1,0.2],\
                 'colsample_bytree':[0.1,0.3,0.5,0.7,0.9,1]}
clf_xgb = GridSearchCV(model_xgb, param)
search_xgb = clf_xgb.fit(X_train_tmp,y_train)
param_best_xgb=search_xgb.best_params_

saveObject=(param_best_lr,param_best_svm,param_best_rf,\
            param_best_ann,param_best_xgb)
with open(path+'\\param_best_quarter.pickle','wb') as f:
    pickle.dump(saveObject, f)
with open(path+'\\param_best_quarter.pickle','rb') as f:
    testout=pickle.load(f)   
param_best_lr=testout[0]
param_best_svm=testout[1]
param_best_rf=testout[2]
param_best_ann=testout[3]
param_best_xgb=testout[4]

all_result=pd.DataFrame(columns=['model','no_var','param_best','acc',\
                                 'prec','recall','fscore','op','auc',\
                                     'fpr','fnr','threshold','cm'])

pos=0
for i in range(0,len(X.columns)):
    X_train=X_train_tmp[X_train_tmp.columns[0:i+1]]
    X_test=X_test_tmp[X_test_tmp.columns[0:i+1]]

    #LogisticRegression########################################################
    model_lr=LogisticRegression(random_state=0,max_iter=1000)
#    model_lr=LogisticRegression(random_state=0,max_iter=1000,\
#                                C=param_best_lr['C'],penalty=param_best_lr['penalty'],solver=param_best_lr['solver'])
    param=dict(C=uniform(loc=0,scale=300),penalty=['l2'],solver=['newton-cg'])
    clf_lr = RandomizedSearchCV(model_lr, param, random_state=0,n_iter=1000).fit(X_train,y_train)
#    search_lr = clf_lr.fit(X_train,y_train)
#    clf_lr=model_lr.fit(X_train,y_train)    
    y_pred_lr=clf_lr.predict(X_test)
    acc_lr=clf_lr.score(X_test,y_test)
    prec_lr,recall_lr,fscore_lr,op_lr=precision_recall_fscore_support(y_test, y_pred_lr, average='binary')
    auc_lr=roc_auc_score(y_test, clf_lr.predict_proba(X_test)[:, 1])
    fpr_lr, fnr_lr, thresholds_lr = det_curve(y_test, y_pred_lr)
    cm_lr = metrics.confusion_matrix(y_test, y_pred_lr)
    list_lr=['lr',i+1,param_best_lr,acc_lr,prec_lr,recall_lr,fscore_lr,op_lr,\
             auc_lr,fpr_lr[-1],fnr_lr[-1],thresholds_lr,cm_lr]
    all_result.loc[pos,:]=list_lr
    pos=pos+1

    #SVM########################################################    
    #C, 1以上越大越好
    #gamma auto比scale好很多
    model_svm=svm.SVC(probability=True, random_state=0)
#uniform範圍[loc, loc+scale]
    param=dict(C=uniform(loc=0,scale=300),gamma=uniform(loc=0, scale=100),kernel=['rbf'])
    clf_svm = RandomizedSearchCV(model_svm, param, random_state=0,n_iter=1000).fit(X_train, y_train)
#    search_svm = clf_svm.fit(X_train_tmp,y_train)
#    model_svm=svm.SVC(probability=True, random_state=0,\
#                      C=param_best_svm['C'],gamma=param_best_svm['gamma'],kernel=param_best_svm['kernel'])
#    clf_svm= model_svm.fit(X_train,y_train)
    y_pred_svm=clf_svm.predict(X_test)
    acc_svm=clf_svm.score(X_test,y_test)
    prec_svm,recall_svm,fscore_svm,op_svm=precision_recall_fscore_support(y_test, y_pred_svm, average='binary')
    auc_svm=roc_auc_score(y_test, clf_svm.predict_proba(X_test)[:, 1])
    fpr_svm, fnr_svm, thresholds_svm = det_curve(y_test, y_pred_svm)
    cm_svm = metrics.confusion_matrix(y_test, y_pred_svm)
    list_svm=['svm',i+1,param_best_svm,acc_svm,prec_svm,recall_svm,fscore_svm,op_svm,\
             auc_svm,fpr_svm[-1],fnr_svm[-1],thresholds_svm,cm_svm]
    all_result.loc[pos,:]=list_svm
    pos=pos+1
    
    
    #test:
    '''
    clf2 = svm.SVC(probability=True,random_state=0,kernel='rbf',C=40,gamma=60).fit(X_train, y_train)
    acc2 = clf2.score(X_test, y_test)
    y_pred2 = clf2.predict(X_test)
    prec2,recall2,fscore2,op2=precision_recall_fscore_support(y_test, y_pred2, average='binary')
    auc2=roc_auc_score(y_test, clf2.predict_proba(X_test)[:, 1])
    fpr2, fnr2, thresholds2 = det_curve(y_test, y_pred2)
    cm2 = metrics.confusion_matrix(y_test, y_pred2)
    '''
    
    
    #Random Forest########################################################        
    model_rf=RandomForestClassifier(random_state=0)
    param=dict(max_depth=[10,20,30,40,50,60,70,80,90,100,None],\
               min_samples_split=[2,5,10],\
                   min_samples_leaf=[1,2,4],\
                       max_features=['sqrt','log2',None],\
                           n_estimators=[10,50,100,200,300,400,500],\
                               bootstrap=[True, False])
    clf_rf = RandomizedSearchCV(model_rf, param, random_state=0,n_iter=1000).fit(X_train,y_train)
#    search_rf = clf_rf.fit(X_train_tmp,y_train)
#    model_rf=RandomForestClassifier(random_state=0,\
#                                    bootstrap=param_best_rf['bootstrap'],\
#                                        max_depth=param_best_rf['max_depth'],\
#                                            max_features=param_best_rf['max_features'],\
#                                                min_samples_leaf=param_best_rf['min_samples_leaf'],\
#                                                    min_samples_split=param_best_rf['min_samples_split'],\
#                                                        n_estimators=param_best_rf['n_estimators'])
#    clf_rf = model_rf.fit(X_train,y_train)
    y_pred_rf=clf_rf.predict(X_test)
    acc_rf=clf_rf.score(X_test,y_test)
    prec_rf,recall_rf,fscore_rf,op_rf=precision_recall_fscore_support(y_test, y_pred_rf, average='binary')
    auc_rf=roc_auc_score(y_test, clf_rf.predict_proba(X_test)[:, 1])
    fpr_rf, fnr_rf, thresholds_rf = det_curve(y_test, y_pred_rf)
    cm_rf = metrics.confusion_matrix(y_test, y_pred_rf)
    list_rf=['rf',i+1,param_best_rf,acc_rf,prec_rf,recall_rf,fscore_rf,op_rf,\
             auc_rf,fpr_rf[-1],fnr_rf[-1],thresholds_rf,cm_rf]
    all_result.loc[pos,:]=list_rf
    pos=pos+1
    
    #test:
    '''
    clf3 = RandomForestClassifier(random_state=42,max_depth=5).fit(X_test, y_test)
    acc3 = clf3.score(X_test, y_test)
    y_pred3 = clf3.predict(X_test)
    prec3,recall3,fscore3,op3=precision_recall_fscore_support(y_test, y_pred3, average='binary')
    auc3=roc_auc_score(y_test, clf3.predict_proba(X_test)[:, 1])
    fpr3, fnr3, thresholds3 = det_curve(y_test, y_pred3)
    cm3 = metrics.confusion_matrix(y_test, y_pred3)
    '''
    
    #ANN########################################################    
    model_ann=MLPClassifier(random_state=0,max_iter=50000)
    param = {
        'hidden_layer_sizes': [(5,),(10,),(20,),(30,),(40,),(50,),(60,),(70,),(80,),(90,),(100,)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [1e-5,0.0001,0.001,0.01,0.1,1,10,100],
        #'learning_rate': ['constant','adaptive'],
    }
    #clf_ann = RadonmizedSearchCV(model_ann, param, random_state=0,n_iter=1000)
    clf_ann = GridSearchCV(model_ann, param).fit(X_train,y_train)
#    search_ann = clf_ann.fit(X_train_tmp,y_train)

#    model_ann=MLPClassifier(random_state=0,max_iter=50000,\
#                            activation=param_best_ann['activation'],\
#                                alpha=param_best_ann['alpha'],\
#                                    hidden_layer_sizes=param_best_ann['hidden_layer_sizes'],\
#                                        solver=param_best_ann['solver'])
#    clf_ann = model_ann.fit(X_train,y_train)
    #param_best_ann=search_ann.best_params_
    y_pred_ann=clf_ann.predict(X_test)
    acc_ann=clf_ann.score(X_test,y_test)
    prec_ann,recall_ann,fscore_ann,op_ann=precision_recall_fscore_support(y_test, y_pred_ann, average='binary')
    auc_ann=roc_auc_score(y_test, clf_ann.predict_proba(X_test)[:, 1])
    fpr_ann, fnr_ann, thresholds_ann = det_curve(y_test, y_pred_ann)
    cm_ann = metrics.confusion_matrix(y_test, y_pred_ann)
    list_ann=['ann',i+1,param_best_ann,acc_ann,prec_ann,recall_ann,fscore_ann,op_ann,\
             auc_ann,fpr_ann[-1],fnr_ann[-1],thresholds_ann,cm_ann]
    all_result.loc[pos,:]=list_ann
    pos=pos+1

    #XGBoost########################################################    

    model_xgb=XGBClassifier(random_state=0)
    param = {'n_estimators':[10,20,50,100,200,300],\
             'max_depth':[1,5,10,20],\
                 'learning_rate':[0.001,0.01,0.05,0.1,0.2],\
                     'colsample_bytree':[0.1,0.3,0.5,0.7,0.9,1]}
    clf_xgb = GridSearchCV(model_xgb, param).fit(X_train, y_train)
#    model_xgb=XGBClassifier(random_state=0,\
#                            n_estimators=param_best_xgb['n_estimators'],\
#                                max_depth=param_best_xgb['max_depth'],\
#                                    learning_rate=param_best_xgb['learning_rate'],\
#                                        colsample_bytree=param_best_xgb['colsample_bytree'])
#    clf_xgb = model_xgb.fit(X_train,y_train)
    #param_best_xgb=search_xgb.best_params_
    y_pred_xgb=clf_xgb.predict(X_test)
    acc_xgb=clf_xgb.score(X_test,y_test)
    prec_xgb,recall_xgb,fscore_xgb,op_xgb=precision_recall_fscore_support(y_test, y_pred_xgb, average='binary')
    auc_xgb=roc_auc_score(y_test, clf_xgb.predict_proba(X_test)[:, 1])
    fpr_xgb, fnr_xgb, thresholds_xgb = det_curve(y_test, y_pred_xgb)
    cm_xgb = metrics.confusion_matrix(y_test, y_pred_xgb)
    list_xgb=['xgb',i+1,param_best_xgb,acc_xgb,prec_xgb,recall_xgb,fscore_xgb,op_xgb,\
             auc_xgb,fpr_xgb[-1],fnr_xgb[-1],thresholds_xgb,cm_xgb]
    all_result.loc[pos,:]=list_xgb
    pos=pos+1


    model_stack=StackingClassifier(estimators=[('lr',model_lr),\
                                                  ('svm',model_svm),\
                                                      ('rf',model_rf),\
                                                          ('mpl',model_ann),\
                                                              ('xgb',model_xgb)])
    clf_stack = model_stack.fit(X_train,y_train)
#    param_best_stack=search_stack.best_params_
    y_pred_stack=clf_stack.predict(X_test)
    acc_stack=clf_stack.score(X_test,y_test)
    prec_stack,recall_stack,fscore_stack,op_stack=precision_recall_fscore_support(y_test, y_pred_stack, average='binary')
    auc_stack=roc_auc_score(y_test, clf_stack.predict_proba(X_test)[:, 1])
    fpr_stack, fnr_stack, thresholds_stack = det_curve(y_test, y_pred_stack)
    cm_stack = metrics.confusion_matrix(y_test, y_pred_stack)
    list_stack=['stack',i+1,'param_best_stack',acc_stack,prec_stack,recall_stack,fscore_stack,op_stack,\
             auc_stack,fpr_stack[-1],fnr_stack[-1],thresholds_stack,cm_stack]
    all_result.loc[pos,:]=list_stack
    pos=pos+1
    
    all_result.to_excel(path+'\\all_result_rcvonce_forward_temp.xlsx')
    with open(path+'\\all_result_rcvonce_forward_temp.pickle', 'wb') as f:
        pickle.dump(all_result, f)


'''
all_result.to_excel(path+'\\all_result_20230417.xlsx')

with open(path+'\\all_result_20230417.pickle', 'wb') as f:
    pickle.dump(all_result, f)
'''
#test:
'''
clf4 = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10,), random_state=42).fit(X_train, y_train)
acc4 = clf4.score(X_test, y_test)
y_pred4 = clf4.predict(X_test)
prec4,recall4,fscore4,op4=precision_recall_fscore_support(y_test, y_pred4, average='binary')
auc4=roc_auc_score(y_test, clf4.predict_proba(X_test)[:, 1])
fpr4, fnr4, thresholds4 = det_curve(y_test, y_pred4)
cm4 = metrics.confusion_matrix(y_test, y_pred4)
'''

#print(cm)
'''
param_dist = dict(n_neighbors=k_range, weights=weight_options)
kf = KFold(n_splits=5, shuffle = False).split(range(25))
'''