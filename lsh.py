from datasketch import MinHashLSH, MinHash
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

# Sample dataset
dataset = ["This is document one", "This is document two", "This is document three"]

# Text preprocessing
stop_words = set(stopwords.words('english'))
preprocessed_docs = []
for doc in dataset:
    tokens = word_tokenize(doc.lower())  # Tokenize and lowercase
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]  # Remove punctuation and stopwords
    preprocessed_docs.append(" ".join(tokens))

# Vectorization (using simple word counts)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(preprocessed_docs)

# Convert document vectors into MinHash signatures
minhashes = []
for i in range(len(preprocessed_docs)):
    minhash = MinHash(num_perm=128)  # num_perm is the number of permutation functions to use
    for word in preprocessed_docs[i].split():
        minhash.update(word.encode('utf8'))
    minhashes.append(minhash)

# Create LSH index
lsh = MinHashLSH(threshold=0.5, num_perm=128)
for i, minhash in enumerate(minhashes):
    lsh.insert(str(i), minhash)

# Query similar documents
query = "This is document four"
query_minhash = MinHash(num_perm=128)
for word in query.lower().split():
    if word in vectorizer.vocabulary_:  # Check if the word is in the vocabulary
        query_minhash.update(word.encode('utf8'))

result = lsh.query(query_minhash)
print("Approximate nearest neighbors:", result)
