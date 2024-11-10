import re
import requests
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from pymongo import MongoClient
from flask import Flask, jsonify, request
import logging
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import torch
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
app = Flask(__name__)
try:
    stop_words_russian = set(stopwords.words('russian'))
except LookupError:
    nltk.download('stopwords')
    stop_words_russian = set(stopwords.words('russian'))
try:
    stop_words_english = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words_english = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stop_words = stop_words_russian.union(stop_words_english)

client = MongoClient('mongodb://localhost:27017/')
db = client['ml_training_data_db']
productionVacancies = db['productionVacancies']

SPRING_BACKEND_URL = 'http://localhost:8081/api/candidates'
model_transformer = SentenceTransformer('paraphrase-xlm-r-multilingual-v1').to('cuda')
def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'\b([А-Яа-яA-Za-z]{1,3})\.', r'\1', text)  
    text = re.sub(r'[^А-Яа-яЁёa-zA-Z\s\-\.]', ' ', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)
def getAllVacanciesFromProduction():
    try:
        logging.info(f"Отправка GET-запроса на {SPRING_BACKEND_URL}")
        response = requests.get(SPRING_BACKEND_URL)
        if response.status_code != 200:
            logging.error(f"Ошибка при получении данных из бэкенда: {response.status_code}")
            return f"Ошибка: {response.status_code}" 
        candidates_data = response.json() 
        if not isinstance(candidates_data, list):
            logging.error("Некорректный формат данных от бэкенда, ожидался список")
            return "Ошибка формата данных"
        logging.info(f"Получено кандидатов: {len(candidates_data)}")
        count = 0
        for candidate in candidates_data:
            candidate_id = candidate.get('id')
            full_name = candidate.get('fullName', '')
            contact_info = candidate.get('contactInfo', '')
            position = candidate.get('position', '')  # Добавляем позицию
            work_experience = candidate.get('workExperience', [])
            education = candidate.get('education', [])
            skills = candidate.get('skills', [])
            resume_full_text = candidate.get('resumeFullText', '')
            _class = "spring.candidatematchingsystem.model.Candidate"
            candidate_data = {
                "_id": candidate_id,
                "fullName": full_name,
                "contactInfo": contact_info,
                "position": position, 
                "workExperience": work_experience,
                "education": education,
                "skills": skills,
                "resumeFullText": resume_full_text,
                "_class": _class
            }
            existing = productionVacancies.find_one({"_id": candidate_id})
            if existing:
                logging.info(f"Кандидат с id {candidate_id} уже существует. Пропуск.")
                continue
            try:
                productionVacancies.insert_one(candidate_data)
                logging.info(f"Кандидат с id {candidate_id} добавлен в productionVacancies")
                count += 1
            except Exception as insert_error:
                logging.error(f"Ошибка при вставке кандидата с id {candidate_id}: {insert_error}")    
        logging.info(f"Синхронизировано {count} новых кандидатов.")
        return f"Синхронизировано {count} новых кандидатов." 
    except Exception as e:
        logging.error(f"Ошибка при синхронизации кандидатов: {e}")
        return f"Ошибка: {e}"
def extract_position_keywords(query_text):
    position_keywords = [
        'Full-Stack', 'Product', 'Backend', 'Frontend', 'Data Analyst', 
        'Project Manager', 'Machine Learning', 'DevOps', 'Software', 
        '.NET', 'Android', 'Data Scientist', 'Data Engineer', 
        'Java-разработчик', 'Java Developer', 'Python-разработчик', 
        'Unity', 'DevOps', 'ios'
    ]
    query_text_lower = query_text.lower()
    first_part = re.split(r'[,\s]', query_text_lower)[0] 
    matched_keywords = []
    for keyword in position_keywords:
        keyword_lower = keyword.lower()
        if re.search(r'\b' + re.escape(keyword_lower) + r'\b', first_part.strip()):
            matched_keywords.append(keyword)
    return matched_keywords

def extract_skills_from_sentence(text):
    skills_list = [
    'java', 'python', 'kotlin', 'android', 'react', 'django', 'ios' , 'jira', 'MySQL', 'UML' , 'Agile' , 'Scrum', 'BABOK' , 'CI/CD' , 'Docker', 'Kubernetes', 'Bamboo', 'Kafka', 'Scikit', 'Tableau', 'KPI', 'OKR', 'NoSQL', 'Node.js', 'django', 'MobX', 'Vuex', 'Enzyme', 'Webpack', 'Lodash', 'Solid', 'OOP', 'Algorithms',
    'machine learning', 'deep learning', 'c++', 'javascript', 'unity', 'dijkstra', 'Lua', 'AWS', 'GCP', 'Microsoft Azure', 'Google Cloud', 'Fire Base', 'Firewall', 'SSL', 'Git', 'Hyper-V', 'Power BI', 'NPS', 'Figma', 'GitLab', 'React', 'Angular', 'Vue.js', 'Cypress', 'Gulp', 'jQuery', 'LESS','NGINX', 'DTO', 'gRPC', 
    'sql', 'nosql', 'angular', 'html', 'css', 'swift', 'flutter', 'go', 'golang', 'c#', 'OpenGL', 'Vulkan', 'DirectX', 'Blender', 'TensorFlow', 'Keras', 'JSON', 'XML', 'CSV','REST API', 'NPS', 'Excel' 'User Stories', 'Acceptance Criteria', '.net', 'Redux', 'Jasmine', 'Rollup', 'PostCSS', 'compose', 'Swagger', 'POSTMAN',
    'http','Matplotlib', 'Seaborn', 'Plotly', 'Linear Regression', 'Decision Trees', 'Random Forest', 'K-Means', 'Talend', 'Hive', 'Presto', 'Pandas', 'NumPy', 'SciPy', 'Spark', 'SWOT',  'Waterfall', 'Typeform', 'SurveyMonkey', 'GitHub', 'NLP', 'ML', 'deep', 'Mocha', 'Babel', 'Svelte', 'RESTful', 'MVC', 'MVVM', 'OAuth', 'JWT',
    ]
    text = text.lower()
    extracted_skills = []
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text):
            extracted_skills.append(skill)
    return extracted_skills

def calculate_skills_similarity(query_skills, candidate_skills):
    vectorizer = CountVectorizer().fit_transform([query_skills, candidate_skills])
    return cosine_similarity(vectorizer)[0][1]

def calculate_cosine_similarity(text1, text2):
    text1 = preprocess_text(text1)
    text2 = preprocess_text(text2)
    embedding1 = model_transformer.encode([text1], convert_to_tensor=True).to('cuda')
    embedding2 = model_transformer.encode([text2], convert_to_tensor=True).to('cuda')
    return util.pytorch_cos_sim(embedding1, embedding2).item()
def find_best_candidates_for_query(query_text, k=3):
    processed_query = preprocess_text(query_text)
    position_keywords = extract_position_keywords(processed_query)
    results = []  
    for candidate in productionVacancies.find():
        candidate_id = candidate.get('_id')
        position = candidate.get('position', '')  
        work_experience = " ".join(candidate.get('workExperience', []))
        skills = " ".join(candidate.get('skills', []))
        position_similarity = calculate_cosine_similarity(processed_query, position)
        if position_keywords:
            max_position_similarity = max(
                [calculate_cosine_similarity(" ".join(position_keywords), position)]
            )
            position_similarity = min(max_position_similarity, 1.0)
            if position_similarity > 0.6 :
                position_similarity = 1
            else: 
                position_similarity = 0
        work_experience_similarity = calculate_cosine_similarity(processed_query, work_experience)
        query_skills = ' '.join(extract_skills_from_sentence(query_text))
        skills_similarity = calculate_skills_similarity(query_skills, skills)
        final_score = (position_similarity + work_experience_similarity + skills_similarity) / 3
        if position_similarity < 0.2 and work_experience_similarity < 0.24 and skills_similarity < 0.24:
            final_score = final_score * 0.7
        results.append({
            'candidateId': candidate_id,
            'score': final_score,
            'position': position_similarity,
            'keys': position_keywords
        }) 
    results.sort(key=lambda x: x['score'], reverse=True)
    logging.info(f"Найдено лучших кандидатов: {len(results[:k])}")
    return results[:k]
@app.route('/search_candidates', methods=['GET'])
def search_candidates():
    try:
        k = int(request.args.get('k', 3))
        if k <= 0:
            raise ValueError("Invalid 'k' parameter. It should be a positive integer.")
        query_text = request.args.get('query_text')
        if not query_text:
            raise ValueError("Missing 'query_text' parameter.")
    except ValueError as e:
        logging.warning(f"Некорректные параметры запроса: {e}")
        return jsonify({"error": str(e)}), 400
    best_candidates = find_best_candidates_for_query(query_text, k)
    return jsonify(best_candidates), 200
if __name__ == '__main__':
    getAllVacanciesFromProduction()
    app.run(port=5000)
