# Visa Prediction Classification Using Machine Learning.

## Project Overview:

news companies store vast amounts of data and seek valuable insights. Automated news classification is crucial as the internet generates massive daily news, and users demand quick access to relevant information. Here we intend to develop a deep learning based model to sort the news data.This will enable efficient and effective news classification for improved user experience.

## Methods Used:

- Inferential Statistics
- Machine Learning
- Deep Learning
- Data Visualization
- Predictive Modeling


## Technologies Used: 
- VS code, Windows
- Python, Anaconda, Git, Jupyter   
- Sklearn, Pandas, Matplotlib, Flask, Tensorflow, keras

### Project Description: 
We get about 2100 instances of news articles, labelled into 5 target categories from kaggle. We use pandas, matplotlib to understand the news article category distribution. Then we employed LSTM,, Count-Vectorization along with xgbclassifier to train our model. LSTM provides over 88% accuracy in both test and training data. Hence, we train our final model with LSTM to sort our news articels into their categories. 

## Getting Started 


1. Clone the repository into your VS code inside a folder.
```
git clone https://github.com/Rgarlay/News_Classification.git

```
2. Activate the virtual environment and install all dependencies

```
conda activate venv/
```
pip install -r path_to_txt_file/req.txt 
```

3. Run the code to train the model

NOTE: Change all pre defined local paths to the paths in your local environment.

```
py -m src.components.data_ingestion.py
```

4. Run the app.py to run the model locally

```
py app.py
```

## Contact:

For Any Further Query, you can contact me at : ravigarlay@gmail.com





