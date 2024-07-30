import os
import cv2
import requests
import joblib
import tempfile
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
import pandas as pd
import joblib
from gensim.models import Word2Vec
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import scipy
from nltk.tokenize.toktok import ToktokTokenizer
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'png', 'jpeg'])
CORS(app)

class PlantData:
    kelembapan: float
    intensitasCahaya: int
    suhu: float
    ph: float
    jenisTanah: str

class Soil:
    SoilType: str
    Description: str

class PlantData:
    def __init__(self, kelembapan, intensitasCahaya, ph, jenisTanah, suhu):
        self.kelembapan = kelembapan
        self.intensitasCahaya = intensitasCahaya
        self.ph = ph
        self.jenisTanah = jenisTanah
        self.suhu = suhu

class Articles:
    def __init__(self, articleID, soilType, title, content, imageURL):
        self.articleID = articleID
        self.soilType = soilType
        self.title = title
        self.content = content
        self.imageURL = imageURL
        
    def to_dict(self):
        return {
            "articleID": self.articleID,
            "soilType": self.soilType,
            "title": self.title,
            "content": self.content,
            "imageURL": self.imageURL,
        }

class ArticleData:
    def __init__(self, query, articles):
        self.query = query
        self.articles = articles

def allowed_extension(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def check_server_availability(destination_url, timeout=30):
    try:
        response = requests.get(destination_url, timeout=timeout)
        if response.status_code == 400:
            return True
        else:
            return False
    except requests.exceptions.Timeout:
        return False

# -----------------------------------------------NLP Word2Vec-----------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

stopword_list = set(stopwords.words('indonesian'))
tokenizer = ToktokTokenizer()
factory = StemmerFactory()
stemmer = factory.create_stemmer()
    
# def loadWord2vecModel():
#     model = gensim.models.KeyedVectors.load_word2vec_format('asset/model/word2vec/word2vec_nlp.bin', binary=True)
#     return model

# def remove_stopwords(text, is_lower_case=False):
#     pattern = r'[^a-zA-z0-9\s]'
#     text = re.sub(pattern, "", "".join(text))
#     tokens = tokenizer.tokenize(text)
#     tokens = [token.strip() for token in tokens]
#     if is_lower_case:
#         filtered_tokens = [token for token in tokens if token not in stopword_list]
#     else:
#         filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
#     filtered_text = ' '.join(filtered_tokens)
#     return filtered_text

# Function to get the embedding vector for n dimension, we have used "300"
# def get_embedding(word, model):
#     if word in model.key_to_index:
#         return model[word]
#     else:
#         return np.zeros(300)

# # Getting average vector for each document
# out_dict = {}
# def process_article(articles, model):
#     for article in articles:
#         average_vector = (np.mean(np.array([get_embedding(x, model) for x in nltk.word_tokenize(remove_stopwords(article))]), axis=0))
#         dict = { article : (average_vector) }
#         out_dict.update(dict)

# # Function to calculate the similarity between the query vector and document vector
# def get_sim(query_embedding, average_vector_doc):
#     sim = [(1 - scipy.spatial.distance.cosine(query_embedding, average_vector_doc))]
#     return sim

# # Rank all the documents based on the similarity to get relevant docs
# def Ranked_documents(query):
#     query_words = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(query.lower())],dtype=float), axis=0))
#     rank = []
#     for k,v in out_dict.items():
#         rank.append((k, get_sim(query_words, v)))
#     rank = sorted(rank,key=lambda t: t[1], reverse=True)
#     print('Ranked Documents :')
#     return rank

# --------------------------------- V2.0 ------------------------------------------

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def get_embedding(word, model):
    if word in model:
        return model[word]
    else:
        return np.zeros(300)

# Getting average vector for each document
out_dict = {}
def process_article(documents, model):
    for sen in documents:
        try:
            # Calculate average vector for the sentence
            embeddings = [get_embedding(x, model) for x in nltk.word_tokenize(remove_stopwords(sen))]
            if embeddings:
                average_vector = np.mean(np.array(embeddings), axis=0)
                if average_vector.shape == (300,):
                    out_dict[sen] = average_vector
                else:
                    print(f"Ignoring sentence '{sen}' due to inconsistent vector shape: {average_vector.shape}")
                    print()
            else:
                print(f"No valid embeddings for sentence: {sen}")
        except ValueError as e:
            print(f"Error processing sentence '{sen}': {e}")

# Function to calculate the similarity between the query vector and document vector
def get_sim(query_embedding, average_vector_doc):
    sim = 1 - scipy.spatial.distance.cosine(query_embedding, average_vector_doc)
    return sim

# Rank all the documents based on the similarity to get relevant docs
def Ranked_documents(query, model):
    query = re.sub(r'[^a-zA-Z\s]', '', query)
    query_embedding = np.mean(np.array([get_embedding(x, model) for x in nltk.word_tokenize(remove_stopwords(query.lower))], dtype=float), axis=0)
    rank = [(k, get_sim(query_embedding, v)) for k, v in out_dict.items() if v is not None and not np.isnan(v).any()]
    rank = sorted(rank, key=lambda t: t[1], reverse=True)
    return rank

def loadWord2vecModel():
    model = Word2Vec.load('asset/model/word2vec/word2vec_model_two_dim.bin').wv
    return model

@app.route("/api/articles/matched", methods=['POST'])
def information_retrieval():
    try:
        input_data = request.get_json()
        if not input_data:
            raise ValueError("No input data provided")

        query = input_data.get('query', None)
        if not query:
            raise ValueError("No query provided")

        articles_data = input_data.get('articles', None)
        if not articles_data:
            raise ValueError("No articles provided")

        articles = [Articles(**item) for item in articles_data]

        article_contents = [article.content for article in articles]
        model = loadWord2vecModel()
        print("articles :",article_contents)

        process_article(article_contents, model)

        ranked = Ranked_documents(query, model)

        ranked_articles = []
        for content, similarity in ranked:
            if similarity > 0.9:
                matched_article = next((article for article in articles if article.content == content), None)
                if matched_article:
                    ranked_articles.append({
                        "articleID": matched_article.articleID,
                        "soilType": matched_article.soilType,
                        "title": matched_article.title,
                        "content": matched_article.content,
                        "imageURL": matched_article.imageURL,
                    })

        return jsonify({"articles": ranked_articles})

    except ValueError as e:
        print("error : ", e)
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print("error : ", e)
        return jsonify({"error": str(e)}), 500

# ------------------------------------------------NLP TF-IDF------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stopword_list]
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

@app.route("/api/articles/tfidf", methods=['POST'])
def information_retrieval_tf_idf():
    try:
        input_data = request.get_json()
        if not input_data:
            raise ValueError("No input data provided")

        # Extract query from input data
        query = input_data.get('query', None)
        if not query:
            raise ValueError("No query provided")

        # Extract articles from input data
        articles_data = input_data.get('articles', None)
        if not articles_data:
            raise ValueError("No articles provided")

        # Proses setiap artikel dalam array input
        articles = []
        for item in articles_data:
            article = Articles(
                articleID=item.get('articleID'),
                soilType=item.get('soilType'),
                title=item.get('title'),
                content=item.get('content'),
                imageURL=item.get('imageURL'),
            )
            articles.append(article)
        
        # Preprocessing pada query dan artikel
        preprocessed_query = preprocess_text(query)
        preprocessed_articles = [preprocess_text(article.content) for article in articles]
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(
            max_features=1000,         
            ngram_range=(1, 4),        
            min_df=1,               
            use_idf=True,
            smooth_idf=True,      
            norm='l2'                  
        )
        tfidf_matrix = vectorizer.fit_transform(preprocessed_articles + [preprocessed_query])
        cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

        # Membuat daftar hasil yang relevan
        result = []
        for idx, similarity in enumerate(cosine_similarities):
            if 0.1 < similarity < 1:
                print("sim : ")
                print(similarity)
                result.append(articles[idx].to_dict())

        return jsonify({"articles": result})

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------

def loadmodelSVM():
    model = joblib.load('asset/model/svm/svm93.sav')
    return model

def processImage(image, target_size=(224, 224)):
    img = cv2.imread(image)
    img_resized = cv2.resize(img, target_size)
    # img_normalized = img_resized.astype('float32') / 255
    img_flat = img_resized.flatten()
    img_2d = img_flat.reshape(1, -1)

    return img_2d

def predict_class(image):
    model = loadmodelSVM()
    predictions = model.predict(image)

    return predictions

def loadModelDT():
    model = joblib.load('asset/model/decision_tree/model_dt.sav')
    column_transformer = joblib.load('asset/model/decision_tree/column_transformer_model_dt.sav')
    return model, column_transformer

def preprocess_input(input_data, column_transformer):
    # Convert input_data (dictionary) to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Transform the input data using the loaded ColumnTransformer
    input_encoded = column_transformer.transform(input_df)
    print("encoded data : ", input_encoded)
    return input_encoded

def make_predictions(input_encoded, model):
    print("encoded input: ", input_encoded)
    predictions = model.predict(input_encoded)
    return predictions.tolist()

class PlantData:
    def __init__(self, kelembapan, intensitasCahaya, ph, jenisTanah, suhu):
        self.kelembapan = kelembapan
        self.intensitasCahaya = intensitasCahaya
        self.ph = ph
        self.jenisTanah = jenisTanah
        self.suhu = suhu

@app.before_request
def remove_trailing_slash():
    if request.path != '/' and request.path.endswith('/'):
        return redirect(request.path[:-1])

@app.route("/", methods=['GET'])
def homepage():
    try:
        # Membuka file HTML
        with open("static/index.html", "r") as file:
            return file.read()
    except IOError as e:
        print("Error:", e)
        return "Error: File not found", 500
    

@app.route("/api/recommendation", methods=['POST'])
def plant_recommendation():
    try:
        input_data = request.get_json()
        if not input_data:
            raise ValueError("No input data provided")

        plant = PlantData(**input_data)
        data = {
            "kelembapan": plant.kelembapan,
            "intensitas cahaya": plant.intensitasCahaya,
            "ph": plant.ph,
            "jenis tanah": plant.jenisTanah
        }
        model, column_transformer = loadModelDT()
        encoded_data = preprocess_input(data, column_transformer)
        prediction = make_predictions(encoded_data, model)

        return jsonify({
            "data":{
                "plantRecommendation": prediction[0],
                "suhu":plant.suhu,
                "kelembapan":plant.kelembapan,
                "ph":plant.ph,
                "intensitasCahaya":plant.intensitasCahaya
                },
            "status":{
                    "code":200,
                    "message":"successfully recommending plant"
                }}
            ), 200
    
    except Exception as err:
        app.logger.error(f"handler: bind input error: {err}")
        return jsonify({"error": f"cannot embed data: {err}"}), 400

@app.route("/api/predict", methods = ['POST'])
def soil_prediction():
        image = request.files["image"]
        if image:
            # Membuat file sementara untuk menyimpan file gambar
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, 'temp_image.jpg')
            image.save(temp_path)

            processed_image = processImage(temp_path)
            print(processed_image)
            predicted_class = predict_class(processed_image)
            print("predicted : ", predicted_class[0])

            os.remove(temp_path)
            os.rmdir(temp_dir)
            return jsonify({
                "data": {
                    "jenis_tanah": predicted_class[0]
                }, 
                "status": {
                    "code": 200,
                    "message": "successfully predicted soil type"
                },
            }), 200
        else:
            return jsonify({
                "error": "image file needed"
                }), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8081)))
