# Natural Language Processing

Interaction between computers and languages, NLP applies ML models to text and language.

Uses:

- sentiment analysis: identifying mood, average sentiment, opinion mining
- predict genre of book
- question answering
- translation
- document summarisation

Many libraries:

- NLTK
- SpaCy
- Stanford NLP
- OpenNLP

Bag of words models:

- preprocess text to classify before fitting classification algorithm
- Involves
  - vocabulary of known words
  - measure of presence of known words

# Example

- csv files can cause issues, as commas are used in normal language
  - a solution is to use " at beginning and end, but some text might contain ""
- can use tsv (tab separated values), usually there are no tabs in text


```python
# Importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
# quoting = 3 to ignore double quote ""

# Cleaning the texts
# - eliminate not useful words (the, on)
# - eliminate numbers
# - stemming, use stem of words (loved, loving.. lov), not to have too many words
# - all lower cap
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    
    review = ' '.join(review)
    
    corpus.append(review)

# Creating the bag of words model 
#sparse matrix through tokenization, each word has a column, 
#with 0 or 1 in each row if it appears in the review
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #max_features can limit how many words
X = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:, 1]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_test, y_test)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

```