from django.shortcuts import render
import random
from django.http import JsonResponse
import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

training = pd.read_csv('C:/Users/saad/OneDrive/Desktop/CS comm/char/Data/Training.csv')
testing= pd.read_csv('C:/Users/saad/OneDrive/Desktop/CS comm/char/Data/Testing.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y


reduced_data = training.groupby(training['prognosis']).max()

#mapping strings to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']  
testy    = le.transform(testy)


clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
print(clf.score(x_train,y_train))
print ("cross result========")
scores = cross_val_score(clf, x_test, y_test, cv=3)
print (scores)
print (scores.mean())


model=SVC()
model.fit(x_train,y_train)
print("for svm: ")
print(model.score(x_test,y_test))

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

def readn(nstr):
    engine = pyttsx3.init()

    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)

    engine.say(nstr)
    engine.runAndWait()
    engine.stop()


severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()

symptoms_dict = {}

for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index

tree_ = clf.tree_
feature_name = [
    cols[i] if i != _tree.TREE_UNDEFINED else "undefined!"
    for i in tree_.feature
]

chk_dis=",".join(cols).split(",")
symptoms_present = []

def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
    
def getDescription():
    global description_list
    with open('C:/Users/saad/OneDrive/Desktop/CS comm/char/MasterData/symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)

def getSeverityDict():
    global severityDictionary
    with open('C:/Users/saad/OneDrive/Desktop/CS comm/char/MasterData/symptom_severity.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass


def getprecautionDict():
    global precautionDictionary
    with open('C:/Users/saad/OneDrive/Desktop/CS comm/char/MasterData/symptom_precaution.csv') as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)  


length = 0
def handle_user_input(symptom):
    conf, cnf_dis = check_pattern(chk_dis, symptom)
    global length
    length = len(cnf_dis)
    print(conf, cnf_dis)
    print(le)
    response = ""
    if conf == 1:
        response += "Searches related to input:\n"
        for num, it in enumerate(cnf_dis):
            response += str(num) + ")" + it + "\n"
    else:
        response = "Enter valid symptom.\n"
    return response

def print_disease(node):
    node = node[0]
    val  = node.nonzero() 
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))

global symps_map
global present_disease
present_disease = print_disease(tree_.value[0])
symps_map = []
symptoms_exp = []
def recurse(node, depth, disease_input):
    indent = "  " * depth
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
        name = feature_name[node]
        threshold = tree_.threshold[node]

        if name == disease_input:
            val = 1
        else:
            val = 0
        if  val <= threshold:
            recurse(tree_.children_left[node], depth + 1, disease_input)
        else:
            symptoms_present.append(name)
            recurse(tree_.children_right[node], depth + 1, disease_input)
    else:
        present_disease = print_disease(tree_.value[node])
        # print( "You may have " +  present_disease )
        red_cols = reduced_data.columns 
        symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
        for syms in list(symptoms_given):
            inp=""
            symps_map.append(syms)
            
def calc_condition(exp,days):
    days = int(days)
    sum=0
    for item in exp:
         sum=sum+severityDictionary[item]
    if((sum*days)/(len(exp)+1)>13):
        return "You should take the consultation from doctor. "
    else:
        return "It might not be that bad but you should take precautions."
        
def sec_predict(symptoms_exp):
    df = pd.read_csv('C:/Users/saad/OneDrive/Desktop/CS comm/char/Data/Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])

def disease_check(symptoms_exp):
    precution_list = precautionDictionary[present_disease[0]]
    print("Take following measures : ")
    for i, j in enumerate(precution_list):
        print(i + 1, ")", j)

def get_precaution():
    precution_list = precautionDictionary[present_disease[0]]
    server_response = "Take following measures : "
    for i, j in enumerate(precution_list):
        server_response += " " + str(i + 1) + ")" + j
    return server_response


cnt = 0
name = ""
symptom = ""
symptom_option = ""
num_days = 0
symps_map_cnt = 0
def home(request):
    global cnt, name, symptom, symptom_option, num_days, symps_map_cnt
    if request.method == 'POST':
        if cnt == 1:
            cnt = cnt + 1
            name = request.POST.get('message')
            server_response = "Hello " + name + "!! Enter the symptom you are experiencing?"
            return JsonResponse({'message': server_response})
        elif cnt == 2:
            symptom = request.POST.get('message')
            server_response = handle_user_input(symptom)
            if "Enter valid symptom." in server_response:
                return JsonResponse({'message': server_response})
            else:
                cnt = cnt + 1
                # print(le)
                if length > 0:
                    server_response += "Select the one you meant (0 - " + str(length-1) + "):  "
                else:
                    server_response = "Enter valid symptom: "
                return JsonResponse({'message': server_response})
        elif cnt == 3:
            cnt = cnt + 1
            sympton_option = request.POST.get('message')
            server_response = "Okay. From how many days ? : "
            return JsonResponse({'message': server_response})
        elif cnt == 4:
            num_days = request.POST.get('message')
            if not num_days.isdigit():
                server_response = "Enter Valid Number"
                return JsonResponse({'message' : server_response})
            else:
                cnt += 1
                server_response = "Provide yes/no only, if You are expriencing Any[tpye okay to proceed]"
                recurse(0, 1, symptom)
                return JsonResponse({'message' : server_response})
        elif cnt == 5:
            user_msg = request.POST.get('message')
            if user_msg == "yes" and symps_map_cnt != 0:
                symptoms_exp.append(symps_map[symps_map_cnt-1])
            l = len(symps_map)
            if l == symps_map_cnt:
                cnt += 1
                print(symptoms_exp)
                getSeverityDict()
                getDescription()
                getprecautionDict()
                second_prediction = sec_predict(symptoms_exp)
                server_response = calc_condition(symptoms_exp, num_days)
                if present_disease[0] == second_prediction[0]:
                    server_response += "You may have ", present_disease[0]
                    server_response += description_list[present_disease[0]]
                    # readn(f"You may have {present_disease[0]}")
                    # readn(f"{description_list[present_disease[0]]}")

                else:
                    # server_response += "You may have ", present_disease[0], "or ", second_prediction[0]
                    server_response += "You may have " + str(present_disease[0]) + " or " + str(second_prediction[0])
                    server_response += description_list[present_disease[0]]
                    server_response += description_list[second_prediction[0]]
                server_response += " Type Okay to get precautions for mentioned disease"
                return JsonResponse({'message': server_response})
            else:
                server_reponse = symps_map[symps_map_cnt] + "? :"
                symps_map_cnt += 1
                return JsonResponse({'message': server_reponse})    
        elif cnt == 6:
            user_msg = request.POST.get('message')
            if user_msg == "okay" or user_msg == "Okay":
                server_response = get_precaution()
                cnt += 1
                return JsonResponse({'message': server_response})
            else:
                return JsonResponse({'message': "Type Okay to get precautions"})
        
        else:
            user_message = request.POST.get('message')
            server_response = "Thank You!! if you have any other questions let me know"
            cnt = 1
            return JsonResponse({'message': server_response})
    else:
        cnt = cnt + 1
        initial_message = get_initial_message()
        return render(request, 'home.html', {'server_message': initial_message})



def get_initial_message():
    greetings = ["Hello! I am the HealthCare ChatBot. What is your name?", 
                 "Hi there! I'm here to assist you with your health concerns. What's your name?",
                 "Greetings! I'm the HealthCare ChatBot. What's your name?"]
    return random.choice(greetings)
