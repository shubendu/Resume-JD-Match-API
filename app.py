from flask import Flask, request, render_template, jsonify
import PyPDF2
import io
from werkzeug.utils import secure_filename
import re
from gensim.models.doc2vec import Doc2Vec
import numpy as np
from numpy.linalg import norm

def preprocess_text(text):
    text = text.lower()
    
    # Remove punctuation
    text = re.sub('[^a-z]', ' ', text)
    
    # Remove numerical values 
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text


def model_prediction(input_CV,input_JD):
    print(input_CV)
    model = Doc2Vec.load('cv_job_maching.model')
    v1 = model.infer_vector(input_CV.split())
    v2 = model.infer_vector(input_JD.split())
    similarity = 100*(np.dot(np.array(v1), np.array(v2))) / (norm(np.array(v1)) * norm(np.array(v2)))
    return round(similarity, 2)



app = Flask(__name__)

def extract_text_from_pdf(pdf_file):
    pdf_content = io.BytesIO(pdf_file.read())
    pdf_reader = PyPDF2.PdfReader(pdf_content)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text()
    return text_content

def extracting_similarity_score(text,pdf_files):
    first_line_txt = text
    first_lines = []
    for pdf_file in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        score = model_prediction(preprocess_text(first_line_txt),preprocess_text(pdf_text))
        pdf_file_name = secure_filename(pdf_file.filename)
        first_lines.append({'score': score, 'pdf_file_name': pdf_file_name})
    return first_lines


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/process_data', methods=['POST'])
def process_data_api():
    try:
        # Assuming extracting_similarity_score returns a dictionary
        text_data = request.form.get('text_data')
        pdf_files = request.files.getlist('pdf_file')
        results = extracting_similarity_score(text_data, pdf_files)

        return jsonify(results)
    except Exception as e:
        # Handle exceptions appropriately
        return jsonify({'error': str(e)})

@app.route('/process_data_html', methods=['POST'])
def process_data_html():
    text_data = request.form.get('text_data')
    pdf_files = request.files.getlist('pdf_file')
    results = extracting_similarity_score(text_data,pdf_files)

    return render_template('result_multiple.html', results=results)


if __name__ == '__main__':
    app.run(debug=True)
