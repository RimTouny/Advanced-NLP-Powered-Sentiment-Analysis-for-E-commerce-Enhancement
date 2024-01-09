# **Sentiment Analysis NLP**
Using NLP and a smart chatbot, this project gauges customer sentiments online, offering customization and real-time feedback. Employing TF-BOW-LDA and ML models on (train.csv dataset)[[train_data.csv](https://github.com/RimTouny/Sentiment-Analysis-NLP/files/13878717/train_data.csv)], it empowers e-commerce decisions, culminating in an NLP course at uOttawa in 2023.


- Required libraries: scikit-learn, pandas, matplotlib.
- Execute cells in a Jupyter Notebook environment.
- The uploaded code has been executed and tested successfully within the [Google Colab](https://colab.google/) environment.


## Supervised Sentiment Analysis for Text classification problem
Perform supervised sentiment analysis to categorize user sentiments into three classes: Positive, Negative, and Neutral.

### Independent Variables:
  + 'name': Name of the product.
  + 'brand': Brand of the product.
  + 'categories': Categories associated with the product.
  + 'primaryCategories': Primary category of the product.
  + 'reviews.date': Date of the review.
  + 'reviews.text': Text content of the review.
  + 'reviews.title': Title of the review.
    
### Target variable:
   +	'sentiment': Dependent variable indicating the sentiment (Positive, Negative, Neutral) of the review.

## **Key Tasks Undertaken**
1. **Data Explore**
   The most common keywords and their counts using WorldCloud.
   ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/65d02e4a-1ef0-4052-96d9-df39abf0c90a)

3. **Data Preparation:**
   - Data Cleaning
     + 	Handling Missing Data: The dataset has a very low percentage of missing cells (less than 0.1%) (10 values in reviews.title ), so we can safely drop or impute those missing values based on the specific context.
     +  Handling Duplicate Rows: The dataset has 1.5% duplicate rows, which can be removed to ensure data integrity
       
   - Renaming and Dropping Columns:Renamed the columns 'reviews.text,' 'reviews.title,' and 'reviews.date' to 'reviews_text,' 'reviews_title,' and 'reviews_date,' respectively. Additionally, Dropped the columns 'name,' 'brand,' 'categories,' 'primaryCategories,' and 'reviews.date' from the dataset.

   - Sentiment Label Encoding: Created a mapping dictionary for sentiment labels and encoded the 'sentiment' column into numerical form (1 for 'Positive,' -1 for 'Negative,' and 0 for 'Neutral').

   - Create new Column ‘Polarity Scores’: Apply the SentimentIntensityAnalyzer to the 'reviews_text' column to calculate polarity scores for each review. Polarity scores represent the sentiment of the text as a continuous value between -1 (negative) and 1 (positive).

   - Balancing Data : The classes are imbalanced, you may consider applying techniques like SMOTE to balance the data.
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/fb1b5cdc-17ca-4592-9828-06ee6d08e91f)

4. **Text Feature Engineering:**
   - Normalizing Case Folding: Convert all text to lowercase to ensure consistent comparisons between words.
   - Removing Punctuation: Eliminate special characters and punctuation marks from the text to avoid any interference in analysis.
   - Removing Numbers: Exclude numerical digits from the text as they may not be relevant for certain tasks like sentiment analysis.
   - Removing Stopwords: Remove common words that do not carry much meaning (e.g., "the," "and," "is") using stopwords from the
   - English language.Remove Rare Words: Eliminate words that appear infrequently in the dataset, as they may not contribute significantly to the analysis.
   -  Lemmatization: Convert words to their base or root form (lemmas) to reduce inflected words to a common base form. For example, "running," "runs," and "ran" will all be transformed to "run."

5. **Text Transformations"
   - TF (Term Frequency): Convert the text data into a bag-of-words representation, where each document is represented as a vector of word frequencies in the corpus.
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/74ff0455-0fc3-40dd-9222-389d47ae2517)
     
   - Bag-of-Words (BOW): Similar to TF, but it also ignores the frequency and considers only whether a word appears or not (binary representation).
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/909af4b9-4d36-43a2-ae11-8dfbfa987d4b)

   - LDA (Latent Dirichlet Allocation): Perform topic modeling to extract latent topics from the text data. Each document is represented as a mixture of topics.
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/5ceabe97-a8b2-479e-ac3f-4b878211a702)

6. **Modeling**
   - Classfication (Random Forest , SVM , Logistic Regression , Gaussian Navie Bayes"
     + TF-IDF Technique
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/37325589-d305-47da-bf6b-345ddea21b8d)
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/e32ec1a5-efab-405b-905e-64beb8dd6392)
       
     + BOW Technique
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/37f39cda-b7e8-47de-9c5e-5b49ba13c1c4)
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/a9923780-abbd-48fc-a4d5-30d1e8923d0f)

     + LDA Technique
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/ef5bdf94-0aba-401d-8889-da257af4b8aa)
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/1165335b-1104-43fa-a296-8c41e5f3c0a7)


   - Clustering ( K-Means , Hierarchica)
     + TF-IDF Technique
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/cc73056d-ed5d-41b0-97ea-22fa43c9bfee)

     + BOW Technique
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/280360b0-19d5-4f91-b0dd-34202c7e1020)

     + LDA Technique
       ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/54476d43-91e9-4572-93a4-27c053de2efb)

7. **Evaluations**
   ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/b601048c-f2fb-4eef-ba6e-f455e389cd71)
   ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/26c9951a-c126-4bb7-aaa8-333c4dccfeb8)

8. **Champion Model**
   ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/901be233-0977-4e33-9f21-7759931907a1)


