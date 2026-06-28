import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
words = [
    "playing",
    "played",
    "plays",
    "player",
    "running",
    "studies",
    "better"
]
def preprocess_text(text):
        
        texts=text.lower().translate(str.maketrans('', '', string.punctuation)).split()
        new_text = [word for word in texts if word not in stop_words]
        stemlist = [ps.stem(word) for word in new_text]            
        return stemlist


       