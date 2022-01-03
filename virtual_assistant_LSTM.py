
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pickle


model = load_model('models/virtual_assistant_LSTM_model.h5')

pkl_file = open('models/tokenizer.pkl', 'rb')
tokenizer = pickle.load(pkl_file)
pkl_file = open('models/y_train.pkl', 'rb')
y_train = pickle.load(pkl_file)
pkl_file = open('models/y_test.pkl', 'rb')
y_test = pickle.load(pkl_file)
fp_qns = pd.read_pickle('models/fp_qns.pkl')

le = LabelEncoder()
y_train1 = le.fit_transform(y_train)
y_test1 = le.fit_transform(y_test)


def preproc(newdf):
    newdf.loc[:,"line1"] = newdf.Line.str.lower()
    newdf.loc[:,"line1"] = newdf.line1.str.replace("inr|rupees","rs", regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace("\r"," ", regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace("\n"," ", regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace("[\s]+"," ", regex=True)

    newdf.loc[:,"line1"] = newdf.line1.str.replace('http[0-9A-Za-z:\/\/\.\?\=]*',' url_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]+\/[0-9]+\/[0-9]+',' date_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('91[7-9][0-9]{9}', ' mobile_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[7-9][0-9]{9}', ' mobile_pp ', regex=True)

    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]+%', ' digits_percent_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]+percentage', ' digits_percent_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]+th', ' digits_th_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('rs[., ]*[0-9]+[,.]?[0-9]+[,.]?[0-9]+[,.]?[0-9]+[,.]?',' money_digits_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('rs[., ]*[0-9]+',' money_digits_small_pp ', regex=True)

    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]+[x]+[0-9]*',' cardnum_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[x]+[0-9]+',' cardnum_pp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]{4,7}',' simp_digit_otp ', regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[0-9]+',' simp_digit_pp ', regex=True)

    newdf.loc[:,"line1"] = newdf.line1.str.replace("a/c"," ac_pp ", regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[^a-z _]',' ', regex=True)

    newdf.loc[:,"line1"] = newdf.line1.str.replace('[\s]+,'," ", regex=True)
    newdf.loc[:,"line1"] = newdf.line1.str.replace('[^A-Za-z_]+', ' ', regex=True)
    
    return newdf


def pred_new_text(txt1):
    
    newdf = pd.DataFrame([txt1])
    newdf.columns = ["Line"]
    newdf1 = preproc(newdf)
    
    col_te1 = tokenizer.texts_to_sequences(newdf1["line1"])
    col_te2 = pad_sequences(col_te1, maxlen=273, dtype='int32', padding='post')

    class_pred = le.inverse_transform(np.argmax(model.predict(col_te2), axis=-1))[0]
    
    try:
        resp = fp_qns.loc[fp_qns.cat==class_pred,"sent"].values[0]
    except:
        resp = 'Please contact XXXX-XXXXX'
    
    print ("Bot:",resp,"\n")

    return 


# custumer_text = ["want to report card lost", "good morning", "where is atm", "cancel my card"]


def main():
    while True:
        new_predi = input('Customer: ')

        if new_predi=='bye' or new_predi=='Bye':
            break
        else:
            pred_new_text(new_predi)




if __name__ == '__main__':
    main()


