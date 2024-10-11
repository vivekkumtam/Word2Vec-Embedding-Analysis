import nltk
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk.stem import WordNetLemmatizer

sentences = movie_reviews.sents()
print(len(sentences))

def reduce_dimensions(wv):
    num_dimensions = 2
    vectors = np.asarray(wv.vectors)
    labels = wv.index_to_key
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

def plot_with_matplotlib(x_vals, y_vals, labels, words_to_plot):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)
    for w in words_to_plot:
        if w in labels:
            i = labels.index(w)
            print("Plotting", w, "at", x_vals[i], y_vals[i])
            plt.annotate(w, (x_vals[i], y_vals[i]))
        else:
            print(w, "cannot be plotted because its word embedding is not given.")
    plt.show()

def preprocess_text(text):
    words = word_tokenize(text)

    words = [word.lower() for word in words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

def preprocess_corpus(corpus):
    preprocessed_corpus = []
    for file_id in corpus.fileids():
        document = ' '.join(movie_reviews.words(file_id))
        preprocessed_document = preprocess_text(document)
        preprocessed_corpus.append(preprocessed_document)
    
    return preprocessed_corpus

movie_reviews_corpus = movie_reviews

preprocessed_corpus = preprocess_corpus(movie_reviews_corpus)

print("Original Text:")
print(' '.join(movie_reviews.words(movie_reviews.fileids()[0])))
print("\nPreprocessed Text:")
print(preprocessed_corpus[0])

word_pairs = [("happy", "joyful", 0.9),
              ("angry", "sad", 0.2),
              ("movie", "film", 0.8),
              ("good", "excellent", 0.9),
              ("bad", "horrible", 0.1),
              ("love", "hate", 0.3),
              ("beautiful", "ugly", 0.2),
              ("fun", "boring", 0.4),
              ("actor", "actress", 0.7),
              ("script", "plot", 0.6)]

with open("word_pairs.txt", "w") as file:
    for pair in word_pairs:
        file.write("\t".join(map(str, pair)) + "\n")

sentences = movie_reviews.sents()
model_sg = Word2Vec(sentences, vector_size=100, window=5, sg=1, epochs=10)
model_cbow = Word2Vec(sentences, vector_size=100, window=5, sg=0, epochs=10)

model_sg.wv.save_word2vec_format("wordvectors_sg.txt")
model_cbow.wv.save_word2vec_format("wordvectors_cbow.txt")

wv = api.load("word2vec-google-news-300")

words_to_plot = ["happy", "joyful", "angry", "sad", "movie", "film", "good", "excellent", "bad", "horrible",
                 "love", "hate", "beautiful", "ugly", "fun", "boring", "actor", "actress", "script", "plot"]

x_vals_sg, y_vals_sg, labels_sg = reduce_dimensions(model_sg.wv)
plot_with_matplotlib(x_vals_sg, y_vals_sg, labels_sg, words_to_plot)

x_vals_cbow, y_vals_cbow, labels_cbow = reduce_dimensions(model_cbow.wv)
plot_with_matplotlib(x_vals_cbow, y_vals_cbow, labels_cbow, words_to_plot)

def evaluate_embeddings(embeddings):
    with open("word_pairs.txt", "r") as file:
        lines = file.readlines()
        pairs = [line.strip().split("\t") for line in lines]
        
    embeddings_dict = {word: embeddings[word] for word in embeddings.index_to_key}
    
    embeddings_scores = []
    human_scores = []
    
    for pair in pairs:
        word1, word2, human_score = pair
        human_score = float(human_score)
        
        if word1 in embeddings_dict and word2 in embeddings_dict:
            embedding_score = cosine_similarity([embeddings_dict[word1]], [embeddings_dict[word2]])[0][0]
            
            embeddings_scores.append(embedding_score)
            human_scores.append(human_score)
    
    return np.array(embeddings_scores), np.array(human_scores)

sg_scores, sg_human_scores = evaluate_embeddings(model_sg.wv)
cbow_scores, cbow_human_scores = evaluate_embeddings(model_cbow.wv)
google_news_scores, google_news_human_scores = evaluate_embeddings(wv)

print("Pearson correlation for SkipGram:", pearsonr(sg_scores, sg_human_scores)[0])
print("Pearson correlation for CBOW:", pearsonr(cbow_scores, cbow_human_scores)[0])
print("Pearson correlation for Google News embeddings:", pearsonr(google_news_scores, google_news_human_scores)[0])

words_to_find_similar = ["happy", "movie", "love", "actor", "script"]
for word in words_to_find_similar:
    print(f"\nMost similar words to '{word}' using SkipGram:")
    print(model_sg.wv.most_similar(word))

    print(f"\nMost similar words to '{word}' using CBOW:")
    print(model_cbow.wv.most_similar(word))

    print(f"\nMost similar words to '{word}' using Google News embeddings:")
    print(wv.most_similar(word))

