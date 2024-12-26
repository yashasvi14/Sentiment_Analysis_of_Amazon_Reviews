import json
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import fasttext
#Load data
data = []
file = 'All_Beauty.jsonl'
#Helper Function
def preprocess(text):
    text = re.sub(r'[^\w\s\']',' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip().lower()
###
with open(file, 'r') as fp:
    for line in fp:
        if line.strip():  # avoid empty lines
            record = json.loads(line)
            data.append({'rating': record.get('rating'), 'text': record.get('text')})
#load as df
df = pd.DataFrame(data)
#Processing labels
df['rating'] = df['rating'].replace({5: 'LoveIt', 4: 'LoveIt', 3: 'LikeIt', 2: 'DontLikeIt', 1: 'DontLikeIt'})
df['category'] = '__label__' + df['rating'].astype(str)
df['category_description'] = df['category'] + ' ' + df['text']



df['category_description'] = df['category_description'].map(preprocess)
#Train_Test splite
train,test = train_test_split(df,test_size=0.2)
train.to_csv("Data.train", columns=["category_description"], index=False, header=False)
test.to_csv("Data.test", columns=["category_description"], index=False, header=False)


model = fasttext.train_supervised(input="Data.train")
model.save_model("All_Beauty_2.bin")
model = fasttext.load_model("All_Beauty_2.bin")
#Testing
print(model.test("ecommerce.test"))
print(model.predict("i love this one!"))
print(model.predict("this did not work for me at all and i had an allergic reaction"))
print(model.predict('it is good but price is too high'))