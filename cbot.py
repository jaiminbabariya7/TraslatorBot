#!/usr/bin/env python
# coding: utf-8

# In[1]:

# English, French, Hindi Translator Chatbot using pre-trained model using Hugging Face Transformers library

# Import required libraries
from transformers import pipeline, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch
#import random


# In[2]:


# Define Path for en, fr, hi model
en_fr_model_path = "Helsinki-NLP/opus-mt-en-fr"
fr_en_model_path = "Helsinki-NLP/opus-mt-fr-en"
en_hi_model_path = "Helsinki-NLP/opus-mt-en-hi"
hi_en_model_path = "Helsinki-NLP/opus-mt-hi-en"


# In[3]:


# Load pre-trained tokenizer
en_fr_tokenizer = AutoTokenizer.from_pretrained(en_fr_model_path)
fr_en_tokenizer = AutoTokenizer.from_pretrained(fr_en_model_path)
en_hi_tokenizer = AutoTokenizer.from_pretrained(en_hi_model_path)
hi_en_tokenizer = AutoTokenizer.from_pretrained(hi_en_model_path)


# In[4]:


# Load pre-trained models for en, fr, hi languages
en_fr_model = AutoModelForSeq2SeqLM.from_pretrained(en_fr_model_path)
fr_en_model = AutoModelForSeq2SeqLM.from_pretrained(fr_en_model_path)
en_hi_model = AutoModelForSeq2SeqLM.from_pretrained(en_hi_model_path)
hi_en_model = AutoModelForSeq2SeqLM.from_pretrained(hi_en_model_path)


# In[5]:


# Create translation class

class TranslatorBot:
    def __init__(self):
        self.translator = {}
        self.translator['en_fr'] = pipeline("translation_en_to_fr", model=en_fr_model, tokenizer=en_fr_tokenizer)
        self.translator['fr_en'] = pipeline("translation_fr_to_en", model=fr_en_model, tokenizer=fr_en_tokenizer)
        self.translator['en_hi'] = pipeline("translation_en_to_hi", model=en_hi_model, tokenizer=en_hi_tokenizer)
        self.translator['hi_en'] = pipeline("translation_hi_to_en", model=hi_en_model, tokenizer=hi_en_tokenizer)
        
    def translate(self, text, source_lang, target_lang):
        if source_lang == target_lang:
            return text
        
        translator = self.translator.get(f'{source_lang}_{target_lang}')
        if translator is None:
            return f'Sorry, I don\'t support {source_lang}_{target_lang} translation.'
        
        return translator(text, max_length=400)[0]['translation_text']

# Include argument device=0 in pipeline function to run torch library on GPU

# direct use of pre-trained model pipelines
#en_fr_translator = pipeline("translation_en_to_fr")
#fr_en_translator = pipeline("translation_fr_to_en")
#en_hi_translator = pipeline("translation_en_to_hi")
#hi_en_translator = pipeline("translation_hi_to_en")


# In[6]:


# Define the greeting messages for the chatbot
#greetings = ["Hello! I'm a language translator chatbot. How can I help you today?",
#             "Hi there! I can translate between English, French, and Hindi. What do you need translated?",
#             "Greetings! What can I do for you today?"]

# Define the goodbye messages for the chatbot
#goodbyes = ["Goodbye! Have a great day!",
#            "See you later!",
#            "Take care!"]

# Define a function to generate a random greeting message
#def get_greeting():
#    return random.choice(greetings)

# Define a function to generate a random goodbye message
#def get_goodbye():
#    return random.choice(goodbyes)


# In[7]:


# Define a function to translate text from one language to another
#def translate(text, source_lang, target_lang):
#    if source_lang == "en" and target_lang == "fr":
#        return en_fr_translator(text, max_length=100)[0]['translation_text']
#    elif source_lang == "fr" and target_lang == "en":
#        return fr_en_translator(text, max_length=100)[0]['translation_text']
#    elif source_lang == "en" and target_lang == "hi":
#        return en_hi_translator(text, max_length=100)[0]['translation_text']
#    elif source_lang == "hi" and target_lang == "en":
#        return hi_en_translator(text, max_length=100)[0]['translation_text']
#    else:
#        return "Sorry, I don't support that translation."


# In[8]:


# Define the main function for the chatbot
#def main():
#    print(get_greeting())
#    while True:
        # Get input from the user
#        user_input = input("You: ")

        # Check if the user wants to quit
#        if user_input.lower() in ["bye", "goodbye", "exit"]:
#            print(get_goodbye())
#            break

        # Parse the user input for the source and target languages and the text to translate
#        try:
#            source_lang, target_lang, text = user_input.split(maxsplit=2)
#        except ValueError:
#            print("Sorry, I didn't understand that. Please try again.")
#            continue

        # Translate the text
#        translated_text = translate(text, source_lang, target_lang)

        # Print the translated text
#        print(f"Bot: {translated_text}")
        
#if __name__ == "__main__":
#    main()


# In[ ]:




