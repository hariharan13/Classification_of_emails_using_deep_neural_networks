# Classification_of_emails_using_deep_neural_networks
In this program , deep neural networks is used  to classify emails into one of the 20 pre-trained categories based on the words present in each email after generating TF-IDF
The 20 newsgroups dataset from scikit-learn have been utilized to illustrate the concept
Total dataset=18,846
Train dataset=11,314 and test dataset =7,532
Major steps:
Pre-processing
TF-IDF vector conversion
Deep learning model training and testing.
Model evaluation and results discussion
Pre-processing step consists of:Removal of punctuations,Word tokenization,Converting words into lowercase,Stop word removal,Keeping words of length of at least 3,Stemming words,POS tagging,Lemmatization of words
Due to unavalibility of gpu and limited space in my system  I have trained this network for 2 epochs
However in large scale more epochs is needed for better results


