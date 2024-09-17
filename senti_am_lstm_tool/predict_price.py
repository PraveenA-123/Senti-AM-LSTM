import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import numpy as np
import argparse
from senti_am_lstm_tool.model_utils import load_models  # Import the model loader from model_utils

def vader_sentiment(text, analyzer):
    """ Get VADER sentiment scores. """
    scores = analyzer.polarity_scores(text)
    return scores['compound'], scores['pos'], scores['neu'], scores['neg']

def textblob_sentiment(text):
    """ Get TextBlob sentiment scores. """
    analysis = TextBlob(text)
    return analysis.sentiment.polarity, analysis.sentiment.subjectivity

def preprocess_new_data(new_data, analyzer):
    """ Preprocess data with sentiment analysis and feature selection. """
    new_data['Cleaned_Headlines'] = new_data['Headlines']
    new_data[['Compound_Score', 'Positive', 'Neutral', 'Negative']] = new_data['Cleaned_Headlines'].apply(lambda x: pd.Series(vader_sentiment(x, analyzer)))
    new_data[['Polarity', 'Subjectivity']] = new_data['Cleaned_Headlines'].apply(lambda x: pd.Series(textblob_sentiment(x)))
    
    # Select the features used for the price prediction model
    features = new_data[['Open', 'High', 'Low', 'Positive', 'Negative', 'Neutral', 'Compound_Score', 'Polarity', 'Subjectivity']]
    return features

def scale_and_reshape(features, scaler_features):
    """ Scale and reshape features for LSTM input. """
    features_scaled = scaler_features.transform(features)
    features_reshaped = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))  # Reshape for LSTM
    return features_reshaped

def parse_arguments():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description='Predict prices using Senti-AM-LSTM based on news headlines and market data.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input Excel file containing the headlines and market data.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output Excel file with predicted prices.')
    return parser.parse_args()

def main():
    """ Main function for loading data, running predictions, and saving results. """
    # Parse command-line arguments
    args = parse_arguments()

    # Load models and scalers from the 'models' folder
    attention_model, scaler_features, scaler_target = load_models()

    # Initialize VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Load input data from the Excel file provided by the user
    new_data = pd.read_excel(args.input_file)

    # Preprocess the data
    preprocessed_features = preprocess_new_data(new_data, analyzer)

    # Scale and reshape the features for LSTM model input
    X_new = scale_and_reshape(preprocessed_features, scaler_features)

    # Use the pre-trained AM-LSTM model to predict prices
    new_predictions_scaled = attention_model.predict(X_new)

    # Inverse transform the predictions back to the original price scale
    new_predictions = scaler_target.inverse_transform(new_predictions_scaled)

    # Add the predictions to the original data
    new_data['Predicted_Price'] = new_predictions

    # Display the new data with the predicted prices
    print(new_data[['Months', 'Headlines', 'Open', 'High', 'Low', 'Predicted_Price']])

    # Save the predictions to the specified output file
    new_data.to_excel(args.output_file, index=False)
    print(f"Predictions saved to {args.output_file}")

if __name__ == '__main__':
    main()
