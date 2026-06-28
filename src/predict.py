
import sys
import os
sys.path.append(os.path.abspath(".."))

import joblib
from src.preprocess import preprocess_text

#loaded model and vector joblib file for prediction 
loaded_model = joblib.load("../models/spam_model.joblib")
loaded_Vectorizer = joblib.load("../models/vectorizer.joblib")

# taking email from a user 
email = input("Enter your email : ")

#preprocess the email and convert it into string
preprocess_email = preprocess_text(email)
preprocess_email_join = " ".join(preprocess_email)

#passing the clean email to vectorrizer to convert to numerical so model understand the message
vector_message =loaded_Vectorizer.transform([preprocess_email_join])

#predicting the email using loaded_model 
predict = loaded_model.predict(vector_message)
if predict[0] == 'spam':
    print("The email is classified as: SPAM")
else:
    print("The email is classified as: HAM")