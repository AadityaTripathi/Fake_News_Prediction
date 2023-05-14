# importing libraries
import streamlit as st
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import requests
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from transformers import BertTokenizer
from keras.utils import pad_sequences
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import io
st.markdown("""
        <style>
               .block-container {
                    padding-top: 0.7rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

################# English News starts here ####################

# extracting text from news article (for english)
def urlToText(url):
    def extract_website_text():
        # Send a GET request to the URL
        response = requests.get(url)

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract the text from the website
        website_text = ''
        for paragraph in soup.find_all('p'):
            website_text += paragraph.text + '\n'

        # Return the website text
        return website_text.strip()

    #webText=extract_website_text("https://www.thehindu.com/news/cities/Delhi/delhi-court-summons-bbc-and-two-others-in-connection-to-pm-modis-documentary/article66808804.ece")
    #webText=extract_website_text("https://www.ndtv.com/india-news/made-it-clear-to-pm-narendra-modi-in-2019-that-sharad-pawar-in-book-4001670")
    #webText=extract_website_text("https://indianexpress.com/elections/karnataka-elections-campaign-modi-priyanka-8589639/")
    #webText=extract_website_text("https://news.abplive.com/elections/karnataka-assembly-elections-bjp-plans-roadshow-pm-modi-greater-bengaluru-ahead-of-polls-1599580")
    webText=extract_website_text()
    trigger=[webText.find("BACK TO TOP" ),webText.find("(Except for the headline," )] 
    trigger.sort(reverse=True)
    endIn=trigger[0]
    if(endIn == -1) : endIn = len(webText)
    final=""
    for i in range (endIn):
        final+=webText[i]

    return final

# english news prediction
def predict(text):
    ps = PorterStemmer()
    model = pickle.load(open('english_model.pkl', 'rb'))
    tfidfvect = pickle.load(open('english_tfidfvect2.pkl', 'rb'))
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    
    return prediction

################# English News ends here ####################


################# Hindi News Starts here ####################

f=open('hindi_model.pkl','rb')
class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...
model = CPU_Unpickler(f).load()


MAX_LEN = 96
#model=pickle.load(open('pk_mod.pkl','rb'))


def preprocess(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True) #bert-base-uncased
    sentences = text
    labels=[0]
    lables=torch.tensor(labels)
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    
    encoded_sent = tokenizer.encode(
                        sentences,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 256
                )
        
    input_ids.append(encoded_sent)
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, 
                            dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask) 
    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)
    # Set the batch size.  
    batch_size = 32  
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    return prediction_dataloader
    
def prepro2(text):
    prediction_dataloader=preprocess(text)
    model.eval()
    device = torch.device("cpu")
    predictions , true_labels = [], []
    # Predict 
    for batch in prediction_dataloader:
    # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
    
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    

    with torch.no_grad():
        # Forward pass, calculate logit predictions
        outputs = model(b_input_ids, token_type_ids=None, 
                        attention_mask=b_input_mask)
    logits = outputs[0]
    # print(type(logits))
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
    
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

    count=0
    guesses=[]
    actual=[]
    # For each input batch...
    for i in range(len(true_labels)):
    
    # The predictions for this batch are a 2-column ndarray (one column for "0" and one column for "1"). Pick the label with the highest value and turn this in to a list of 0s and 1s.
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
    

    #calculate accuracy
    for j in range(len(pred_labels_i)):
        if(pred_labels_i[j]==true_labels[i][j]):
            count+=1
        guesses.append(pred_labels_i[j])
        actual.append(true_labels[i][j])
    
    return guesses[0]


################# Hindi News ends here ####################

# ui starts
def app():
    st.header("Fake news detector")
    subhead = '<p style="font-family:Lucida Console; color:#33ccff; font-size: 28px;"><b> Enter The Language of Your News Article</b></p>'
    st.markdown(subhead, unsafe_allow_html=True)
    lang = ["None","English",'Hindi']
    selected_lang=st.selectbox("",lang)
    
    if selected_lang==lang[0]:
        st.image('https://dl.acm.org/cms/attachment/85d0d2f7-5abf-45ce-af59-28d98b34825a/f1.jpg', caption='Fake news trends against data',use_column_width=True ) #
    if selected_lang==lang[1]:
        #st.write("eng")
        text=st.text_input("Enter The URL Of NEWS Article")
        m=st.markdown("""<style>div.stButton > button:first-child {background-color: #0099ff;color:#ffffff;} div.stButton > button:hover { background-color: #00ff00; color:#ffffff;}</style>""",unsafe_allow_html=True)
        if st.button('Submit') :
            #"Click me", key="my_button", bg_color='green', fg_color='white'
            text=urlToText(text)
            if predict(text) == "REAL" :
                st.text("The prediction is :")
                pred = '<p style="font-family:Lucida Console; color:#1aff8c; font-size: 28px;"><b>Real News</b></p>'
                st.markdown(pred, unsafe_allow_html=True)
            elif predict(text) == "FAKE":
                st.text("The prediction is :")
                pred = '<p style="font-family:Lucida Console; color:#ff0000; font-size: 28px;"><b>Fake News</b></p>'
                st.markdown(pred, unsafe_allow_html=True)
            #st.write(predict(text))
            #st.write(text)
    elif selected_lang==lang[2]:
        #st.write("hin")
        text=st.text_area("Enter the Text of the NEWS article")
        m=st.markdown("""<style>div.stButton > button:first-child {background-color: #0099ff;color:#ffffff;} div.stButton > button:hover { background-color: #00ff00; color:#ffffff;}</style>""",unsafe_allow_html=True)
        if st.button('Submit'):
            out=prepro2(text)
            if out == 0 :
                #st.write("Fake")
                st.text("The prediction is :")
                pred = '<p style="font-family:Lucida Console; color:#ff0000; font-size: 28px;"><b>Fake News</b></p>'
                st.markdown(pred, unsafe_allow_html=True)
            else :
                #st.write("Real")
                st.text("The prediction is :")
                pred = '<p style="font-family:Lucida Console; color:#1aff8c; font-size: 28px;"><b>Real News</b></p>'
                st.markdown(pred, unsafe_allow_html=True)
    
    
if __name__ == '__main__':  
    app()