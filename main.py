from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen, ScreenManager
from kivymd.uix.card import MDCard
from kivy.properties import ObjectProperty, StringProperty, NumericProperty
import datetime
import pytz
# from chatterbot import ChatBot

import tensorflow as tf
import Chatbot_class
from preprocess import clean_text
import numpy as np
import json
from keras.preprocessing.text import tokenizer_from_json

# load the tokenziers
with open('./processed_data/inp_lang.json', 'r') as f:
    json_data = json.load(f)
    inp_lang = tokenizer_from_json(json_data)
    f.close()
    
with open('./processed_data/targ_lang.json', 'r') as f:
    json_data = json.load(f)
    targ_lang = tokenizer_from_json(json_data)
    f.close()

# define hyperparameters
embedding_dim = 128
units = 256
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1
max_sentence_length = 15

# create encoder from Chatbot class
encoder = Chatbot_class.create_encoder(vocab_inp_size, embedding_dim, units, max_sentence_length)
print('Encoder model initialized...')
encoder.load_weights('/Users/harsh/Downloads/seq2seq-attention-bot-master/trained_model/encoder_weights.h5')  # load the weights, we shall use them to make inference
print('Encoder model trained weights loaded...')

# create decoder from Chatbot class
decoder = Chatbot_class.create_decoder(vocab_tar_size, embedding_dim, units, units, max_sentence_length)
print('Decoder model initialized...')
decoder.load_weights('/Users/harsh/Downloads/seq2seq-attention-bot-master/trained_model/decoder_weights.h5')
print('Decoder model trained weights loaded...')


def evaluate(sentence, samp_type = 1):
    sentence = clean_text(sentence)
    inputs = []
    # split the sentence and replace unknown words by <unk> token.
    for i in sentence.split(' '):
        try:
            inputs.append(inp_lang.word_index[i])
        except KeyError:
            inputs.append(inp_lang.word_index['<unk>'])
    
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen = max_sentence_length, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    enc_output, enc_hidden = encoder(inputs)
    
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)
    
    for t in range(max_sentence_length):
        predictions, dec_hidden = decoder([enc_output, dec_hidden, dec_input])
        if samp_type == 1:
            # that means simple greedy sampling
            predicted_id = tf.argmax(predictions[0]).numpy()
        elif samp_type == 2:
            predicted_id = np.random.choice(vocab_tar_size, p = predictions[0].numpy())
        elif samp_type == 3:
            _ , indices = tf.math.top_k(predictions[0], k = 3)
            predicted_id = np.random.choice(indices)

        if predicted_id!= 0:
            if targ_lang.index_word[predicted_id] == '<end>':
                return result, sentence
            else:
                result += targ_lang.index_word[predicted_id] + ' '
        
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence



class UserInput(MDCard):
    text = StringProperty()
    font_size= NumericProperty()

class BotResponce(MDCard):
    text = StringProperty()
    font_size= NumericProperty()

class ChatScreen(Screen):
    chat_area = ObjectProperty()
    message = ObjectProperty()

    def __init__(self, **kwargs):
        super(ChatScreen, self).__init__(**kwargs)
        # self.chatbot = ChatBot("Donna")

    def send_message(self):
        self.user_input = self.ids.message.text
        self.ids.message.text = ""
        length = len(self.user_input)

        if length >= 40:
            self.ids.chat_area.add_widget(
                UserInput(text=self.user_input, font_size=17, height = length)
            )
        else:
            self.ids.chat_area.add_widget(
                UserInput(text=self.user_input, font_size=17)
            )
        
    def bot_response(self):
        # response = self.chatbot.get_response(self.user_input)
        if self.user_input == "Hello" or self.user_input == "hello" or self.user_input == "hey" or self.user_input == "Hey" or self.user_input == "hi" or self.user_input == "Hi" or self.user_input == "hy" or self.user_input == "Hy" :
            response = "Hello! I'm Donna your personal assistant, how can i help you,"
        elif self.user_input == "what's your name" or self.user_input == "What's your name" or self.user_input == "What is your name" or self.user_input == "what is your name":
            response = "I'm Donna your personal assistant"
        elif self.user_input == "what's the time right now" or self.user_input == "what is the time right now" or self.user_input == "whats the time right now":
            response = "It's ",datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
        else:
            samp_type = 3
            response, sentence = evaluate(self.user_input, samp_type)
        length = len(str(response))

        if length >= 40:
            self.ids.chat_area.add_widget(
                BotResponce(text="{}".format(response), font_size=17, height=length)
            )
        else:
            self.ids.chat_area.add_widget(
                BotResponce(text="{}".format(response), font_size=17)
            )




class ChatApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = 'Teal'
        sm = ScreenManager()
        sm.add_widget(ChatScreen(name='chat'))
        return sm
    
if __name__=='__main__':
    ChatApp().run()