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

1. **Data Preparation:**
   - Data Cleaning
     + 	Handling Missing Data: The dataset has a very low percentage of missing cells (less than 0.1%) (10 values in reviews.title ), so we can safely drop or impute those missing values based on the specific context.
     +  Handling Duplicate Rows: The dataset has 1.5% duplicate rows, which can be removed to ensure data integrity
       
   - Renaming and Dropping Columns:Renamed the columns 'reviews.text,' 'reviews.title,' and 'reviews.date' to 'reviews_text,' 'reviews_title,' and 'reviews_date,' respectively. Additionally, Dropped the columns 'name,' 'brand,' 'categories,' 'primaryCategories,' and 'reviews.date' from the dataset.

   - Sentiment Label Encoding: Created a mapping dictionary for sentiment labels and encoded the 'sentiment' column into numerical form (1 for 'Positive,' -1 for 'Negative,' and 0 for 'Neutral').

   - Create new Column ‘Polarity Scores’: Apply the SentimentIntensityAnalyzer to the 'reviews_text' column to calculate polarity scores for each review. Polarity scores represent the sentiment of the text as a continuous value between -1 (negative) and 1 (positive).

   - Balancing Data : The classes are imbalanced, you may consider applying techniques like SMOTE to balance the data.
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/fb1b5cdc-17ca-4592-9828-06ee6d08e91f)

2. **Text Feature Engineering:**
   - Normalizing Case Folding: Convert all text to lowercase to ensure consistent comparisons between words.
   - Removing Punctuation: Eliminate special characters and punctuation marks from the text to avoid any interference in analysis.
   - Removing Numbers: Exclude numerical digits from the text as they may not be relevant for certain tasks like sentiment analysis.
   - Removing Stopwords: Remove common words that do not carry much meaning (e.g., "the," "and," "is") using stopwords from the
   - English language.Remove Rare Words: Eliminate words that appear infrequently in the dataset, as they may not contribute significantly to the analysis.
   -  Lemmatization: Convert words to their base or root form (lemmas) to reduce inflected words to a common base form. For example, "running," "runs," and "ran" will all be transformed to "run."

3. 
