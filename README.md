# **Sentiment Analysis NLP**
Using NLP, this project gauges customer sentiments online, offering customization and real-time feedback. Employing TF-BOW-LDA and ML models on [train.csv dataset](https://github.com/RimTouny/Sentiment-Analysis-NLP/files/13878717/train_data.csv), it empowers e-commerce decisions, culminating in an NLP course at uOttawa in 2023.


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
1. **Data Explore:**
   - The most common keywords and their counts.
   ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/65d02e4a-1ef0-4052-96d9-df39abf0c90a)

   - The most common Positive words using WorldCloud.
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/ce4f49c5-844d-47d9-8935-d10027d56dcf)
     
    - The most common Negative words using WorldCloud.
      ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/45029a36-66db-4b66-8b38-52ba6ddf4de5)
      
    - The most common Neutral words using WorldCloud.
      ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/2465af39-c477-43b6-ae50-92b5426a8e27)

2. **Data Preparation:**
   - Data Cleaning
     + 	Handling Missing Data: The dataset has a very low percentage of missing cells (less than 0.1%) (10 values in reviews.title ), so we can safely drop or impute those missing values based on the specific context.
     +  Handling Duplicate Rows: The dataset has 1.5% duplicate rows, which can be removed to ensure data integrity
       
   - Renaming and Dropping Columns:Renamed the columns 'reviews.text,' 'reviews.title,' and 'reviews.date' to 'reviews_text,' 'reviews_title,' and 'reviews_date,' respectively. Additionally, Dropped the columns 'name,' 'brand,' 'categories,' 'primaryCategories,' and 'reviews.date' from the dataset.

   - Sentiment Label Encoding: Created a mapping dictionary for sentiment labels and encoded the 'sentiment' column into numerical form (1 for 'Positive,' -1 for 'Negative,' and 0 for 'Neutral').

   - Create new Column ‘Polarity Scores’: Apply the SentimentIntensityAnalyzer to the 'reviews_text' column to calculate polarity scores for each review. Polarity scores represent the sentiment of the text as a continuous value between -1 (negative) and 1 (positive).

   - Balancing Data : The classes are imbalanced, you may consider applying techniques like SMOTE to balance the data.
     ![image](https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/e2909310-c615-4fec-a2d6-81393546dab1)

3. **Text Feature Engineering:**
   - Normalizing Case Folding: Convert all text to lowercase to ensure consistent comparisons between words.
   - Removing Punctuation: Eliminate special characters and punctuation marks from the text to avoid any interference in analysis.
   - Removing Numbers: Exclude numerical digits from the text as they may not be relevant for certain tasks like sentiment analysis.
   - Removing Stopwords: Remove common words that do not carry much meaning (e.g., "the," "and," "is") using stopwords from the
   - English language.Remove Rare Words: Eliminate words that appear infrequently in the dataset, as they may not contribute significantly to the analysis.
   -  Lemmatization: Convert words to their base or root form (lemmas) to reduce inflected words to a common base form. For example, "running," "runs," and "ran" will all be transformed to "run."

4. **Text Transformations:**
   - Bag-of-Words (BOW): Similar to TF, but it also ignores the frequency and considers only whether a word appears or not (binary representation).
    <p align="center">
      <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/be49cca1-5a4e-4219-995f-0089ce11fa06"/>
    </p>
     
    - Term Frequency-Inverse Document Frequency (TF-IDF): Convert the text data into a bag-of-words representation, where each document is represented as a vector of word frequencies in the corpus.
     <p align="center">
      <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/28ba8211-3901-4b57-9889-748c900f7980" />
     </p>

   - Latent Dirichlet Allocation (LDA): Perform topic modeling to extract latent topics from the text data. Each document is represented as a mixture of topics.
     <p align="center">
      <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/57f35aeb-47a1-4cf6-9dd0-fceac1ece501" />
     </p>

5. **Modeling**
   - Classfication (Random Forest , SVM , Logistic Regression , Gaussian Navie Bayes)
     + BOW Technique
       <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/a248d331-b918-4cc0-a1b5-5c332126004c" />
      </p>
       <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/6aedb3ef-9979-40bb-a83e-940f55cea2ab" />
      </p>
   
     + TF-IDF Technique
       <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/73a3f3d0-9f6d-49ff-b5f3-3340076963cb)" />
      </p>
       <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/32241660-3dce-4501-b039-8d0459ebea17" />
      </p>

     + LDA Technique
        <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/567637f0-7000-4408-a8e8-193b54242d9c" />
      </p>
        <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/d4bccc55-cb6c-4aef-acdd-32481ff5c989" />
      </p>


   - Clustering ( K-Means , Hierarchical)
     + BOW Technique
       - Silhouette Score (K-Means): 81.55401438608376
       - Silhouette Score (Hierarchical) : 17.925024032592773

        <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/d650e0f7-2f2f-454b-96bc-1526257e9023"/>
      </p>
  
     + TF-IDF Technique
       - Silhouette Score (K-Means): 0.7683612431807604
       - Silhouette Score (Hierarchical) : 17.966507375240326
         
        <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/0db92a80-8c87-402b-9e59-1add9d408174"/>
      </p>

     + LDA Technique
       - Silhouette Score (K-Means): 81.55401438608376
       - Silhouette Score (Hierarchical) 16.194509
        <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/0f654069-0e71-44d9-81de-a398a3dddace" />
      </p>

7. **Evaluations**
   
    <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/765fcbda-d8d0-4883-8731-73d5e78304c5" />
      </p>
    <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/36ab4c20-eff9-43c8-94cf-e6e562a5c850" />
      </p>

8. **Champion Model**

   <p align="center">
          <img src="https://github.com/RimTouny/Sentiment-Analysis-NLP/assets/48333870/8e038b8c-ea10-45d8-bc83-823a80b463b9" />
      </p>




