import pickle
import re, string
import sklearn

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

model_path = 'NB_SVM_model.sav'
vectorizer_path = 'vectorizer.sav'
model = pickle.load(open(model_path, 'rb'))
vec = pickle.load(open(vectorizer_path, 'rb'))

while(1):
    message = input("Enter a message and I'll tell you if you're being mean\n")
    check = vec.transform([message])
    toxicity = model.predict(check)
    if toxicity[0] == 1:
        print("That wasn't very nice\n")
    else:
        print("I'm pretty sure that wasn't toxic\n")