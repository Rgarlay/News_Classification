import sys
from ..exception import CustomException
from ..logger import logging
import pandas as pd
from ..utils import OpenTokenizer,load_object,data_preprocessing
import tensorflow as tf
from keras.utils import pad_sequences
import joblib

class CustomData:
    def __init__(self, text: str):
        logging.info("CustomData initialized with text: %s", text)
        self.text = text

    def get_as_dataframe(self):
        try:
            logging.info("Converting input text to DataFrame.")
            custom_data_input = {
                'text' : [self.text]
            }
            logging.info("DataFrame created successfully.")
            logging.info(f'The dataframe is')
            return pd.DataFrame(custom_data_input)

        except Exception as e:
            logging.error("Error occurred while creating DataFrame: %s", str(e))
            raise CustomException(e,sys)
        


class PredictPipeline:
    def __init__(self):
        logging.info("Initialized PredictPipeline class.")
        pass

    def predict(self,features):
        self.features = features
        try:

            ##just the data paths and imports here
            model_path = r'C:\Users\rgarlay\Desktop\DS\news_classification\News_Classification\archieve\model.h5'
            preprocessor_path = r'C:\Users\rgarlay\Desktop\DS\news_classification\News_Classification\archieve\encoder.pkl'
            token_path=r'C:\Users\rgarlay\Desktop\DS\news_classification\News_Classification\archieve\tokenizer.json'

            logging.info("Loading model")
            model = tf.keras.models.load_model(model_path)

            logging.info("Loading preprocessor")
            target_encoder = load_object(preprocessor_path)             
            
            logging.info("Preprocessing input features...")
            sequence2 = data_preprocessing(features,col_name='text') 
            
            logging.info(f'sequence 2 value is {sequence2}')
            
            logging.info("Initializing tokenizer...")
            token = OpenTokenizer(token_path)

            logging.info("Encoding sequences...")
            sequences_encoded2 = token.texts_to_sequences(sequence2)

            logging.info("Padding sequences to fixed length...")
            y_train_final = pad_sequences(sequences_encoded2, maxlen=11499, padding='pre')   ## i found max_len value and hard coded it.

            logging.info("Making predictions...")
            prediction = model.predict(y_train_final)

            prediction_inverted = target_encoder.inverse_transform(prediction)

            prediction_inverted_without_bracket = prediction_inverted[0][0]

        except Exception as e:
            logging.error("An error occurred during prediction: %s", str(e))
            raise CustomException(e,sys)
        
        logging.info("Prediction successful.")
        return prediction_inverted_without_bracket



         