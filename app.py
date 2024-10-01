import os  
import tempfile  
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify  
import pandas as pd  
from sentence_transformers import SentenceTransformer, util  
import numpy as np  
import logging  
from io import BytesIO  
import json  
  
# Initialize Flask app  
app = Flask(__name__)  
app.secret_key = 'your_secret_key'  # You should replace 'your_secret_key' with a real secret key for production  
  
# Set up logging  
logging.basicConfig(level=logging.DEBUG)  
  
# Load pre-trained BERT model from sentence-transformers  
model = SentenceTransformer('all-MiniLM-L6-v2')  
  
# Function to calculate the similarity score  
def calculate_similarity(text1, text2):  
    embeddings1 = model.encode(text1, convert_to_tensor=True)  
    embeddings2 = model.encode(text2, convert_to_tensor=True)  
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)  
    return cosine_scores.item()  
  
# Function to match controls  
def match_controls(df1, df2, text_columns1, text_columns2):  
    matched = []  
    not_matched = []  
  
    for i, row1 in df1.iterrows():  
        best_score = 0  
        best_match = None  
        combined_text1 = ' '.join([str(row1[col]) for col in text_columns1])  
        for j, row2 in df2.iterrows():  
            combined_text2 = ' '.join([str(row2[col]) for col in text_columns2])  
            score = calculate_similarity(combined_text1, combined_text2)  
            if score > best_score:  
                best_score = score  
                best_match = row2  
  
        if best_score >= 0.5:  
            match_data = row1.to_dict()  
            match_data.update(row2.to_dict())  
            match_data['Confidence Score'] = best_score * 100  
            matched.append(match_data)  
        else:  
            not_match_data = row1.to_dict()  
            if best_match is not None:  
                not_match_data.update(best_match.to_dict())  
            not_match_data['Confidence Score'] = best_score * 100  
            not_matched.append(not_match_data)  
  
    # Sort by highest confidence score  
    matched.sort(key=lambda x: x['Confidence Score'], reverse=True)  
    not_matched.sort(key=lambda x: x['Confidence Score'], reverse=True)  
  
    return matched, not_matched  
  
@app.route('/')  
def index():  
    error = request.args.get('error')  
    return render_template('index.html', error=error)  
  
@app.route('/upload', methods=['POST'])  
def upload():  
    try:  
        file1 = request.files['file1']  
        file2 = request.files['file2']  
  
        # Read the first sheet of the excel files  
        df1 = pd.read_excel(file1, sheet_name=0)  
        df2 = pd.read_excel(file2, sheet_name=0)  
  
        columns1 = df1.columns.tolist()  
        columns2 = df2.columns.tolist()  
  
        # Serialize DataFrames to JSON  
        df1_json = df1.to_json()  
        df2_json = df2.to_json()  
  
        return jsonify({  
            'columns1': columns1,  
            'columns2': columns2,  
            'df1': df1_json,  
            'df2': df2_json  
        })  
  
    except Exception as e:  
        logging.error(f"An error occurred during upload: {e}")  
        return jsonify({'error': str(e)}), 500  
  
@app.route('/match', methods=['POST'])  
def match():  
    try:  
        columns1 = request.form.getlist('columns1')  
        columns2 = request.form.getlist('columns2')  
        df1_json = request.form['df1']  
        df2_json = request.form['df2']  
  
        logging.info(f"Columns1: {columns1}")  
        logging.info(f"Columns2: {columns2}")  
  
        if not columns1 or not columns2 or not df1_json or not df2_json:  
            raise ValueError("Missing required form data")  
  
        # Deserialize DataFrames from JSON  
        df1 = pd.read_json(df1_json)  
        df2 = pd.read_json(df2_json)  
  
        matched, not_matched = match_controls(df1, df2, columns1, columns2)  
  
        # Create DataFrames  
        matched_df = pd.DataFrame(matched)  
        not_matched_df = pd.DataFrame(not_matched)  
  
        # Save the DataFrames to an Excel file with multiple sheets  
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')  
        with pd.ExcelWriter(temp_file.name, engine='openpyxl') as writer:  
            matched_df.to_excel(writer, index=False, sheet_name='Matched_Entries')  
            not_matched_df.to_excel(writer, index=False, sheet_name='Not_Matched_Low_Probability_Entries')  
  
        temp_file.close()  
        filename = os.path.basename(temp_file.name)  
  
        # Flash logs to the UI  
        flash('Matching process completed successfully.', 'success')  
        flash(f'Matched Entries: {len(matched)}', 'info')  
        flash(f'Not Matched/Low Probability Entries: {len(not_matched)}', 'info')  
  
        return render_template('result.html', matched=matched, not_matched=not_matched, filename=filename, columns=matched_df.columns)  
  
    except Exception as e:  
        logging.error(f"An error occurred during matching: {e}")  
        flash(f"An error occurred: {e}", 'danger')  
        return redirect(url_for('index', error=str(e)))  
  
@app.route('/download/<filename>', methods=['GET'])  
def download_file(filename):  
    return send_file(os.path.join(tempfile.gettempdir(), filename), as_attachment=True)  
  
if __name__ == '__main__':  
    app.run(debug=True)  

