from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import werkzeug.utils
import requests 
import json
import numpy as np

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)

# --- Configuration ---
UPLOAD_FOLDER = 'sales_uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Helper Functions ---
def get_safe_filepath(filename):
    """Securely gets the path for a file in the upload folder."""
    filename = werkzeug.utils.secure_filename(filename)
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def load_df_from_file(filename):
    """Loads a DataFrame from a given file in the upload folder."""
    filepath = get_safe_filepath(filename)
    if not os.path.exists(filepath):
        return None
        
    if filename.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        return None
    
    df.columns = df.columns.str.strip().str.lower()
    return df

# --- Routes ---
@app.route('/')
def index():
    """Serves the main frontend file."""
    # This route is a fallback for serving the frontend if not done by a dedicated web server.
    # In many production setups, a web server like Nginx would handle this.
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/sales-files', methods=['GET'])
def get_sales_files():
    """Lists all available sales data files."""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        return jsonify([])
    files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith(('.csv', '.xlsx'))]
    return jsonify(files)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handles file uploads for sales data."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
        
    if file:
        filepath = get_safe_filepath(file.filename)
        file.save(filepath)
        return jsonify({"success": True, "message": f"File '{file.filename}' uploaded successfully."})

@app.route('/api/sales-data', methods=['GET'])
def get_sales_data():
    """Gets all sales data from a specified file."""
    filename = request.args.get('file')
    if not filename:
        return jsonify({"error": "No file specified"}), 400
    
    df = load_df_from_file(filename)
    if df is None:
        return jsonify({"error": "File not found or unsupported format"}), 404
        
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/sales-suggestions', methods=['GET'])
def get_sales_suggestions():
    """Generates sales suggestions using the Gemini API."""
    filename = request.args.get('file')
    if not filename:
        return jsonify({"suggestions": ["Select a file to get suggestions."]})
        
    df = load_df_from_file(filename)
    if df is None or 'cost' not in df.columns:
        return jsonify({"suggestions": ["Selected file must contain a 'cost' column for AI-powered suggestions."]})

    api_key = os.getenv("API_KEY", "AIzaSyAOfe_P7HQs8jdh4Kjn_Dukr6cKE1GS4ds")
    if not api_key:
        return jsonify({"suggestions": ["API_KEY environment variable not set. Could not generate AI suggestions."]})

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    df['profit'] = (df['price'] - df['cost']) * df['quantity']
    data_summary = df.head().to_json(orient='records')

    prompt = f"""
    You are an expert sales analyst. Based on the following sales data sample, provide three actionable and concise suggestions to improve sales and profitability.
    Data Sample (JSON format):
    {data_summary}
    """
    
    response_schema = {
        "type": "OBJECT",
        "properties": { "suggestions": { "type": "ARRAY", "items": {"type": "STRING"} } }
    }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": { "responseMimeType": "application/json", "responseSchema": response_schema }
    }

    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        json_text = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify(json.loads(json_text))
    except Exception as e:
        print(f"API Error: {e}")
        return jsonify({"suggestions": ["Error communicating with the AI service."]})


@app.route('/api/advanced-analytics', methods=['GET'])
def get_advanced_analytics():
    """Provides advanced analytics including profitability analysis."""
    filename = request.args.get('file')
    if not filename:
        return jsonify({"error": "No file specified"}), 400

    df = load_df_from_file(filename)
    if df is None or 'cost' not in df.columns:
        return jsonify({ "total_profit": 0, "average_profit_margin": 0, "mom_revenue_growth": 0, "top_profitable_products": [], "least_profitable_products": [] })

    df['date'] = pd.to_datetime(df['date'])
    df['revenue'] = df['price'] * df['quantity']
    df['profit'] = (df['price'] - df['cost']) * df['quantity']

    total_revenue = df['revenue'].sum()
    total_profit = df['profit'].sum()
    average_profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0

    monthly_revenue = df.set_index('date').resample('M')['revenue'].sum()
    mom_growth = monthly_revenue.pct_change().iloc[-1] * 100 if len(monthly_revenue) > 1 else 0

    product_profit = df.groupby('product')['profit'].sum().sort_values()
    
    top_profitable = product_profit.tail(5).iloc[::-1]
    least_profitable = product_profit.head(5)

    return jsonify({
        "total_profit": float(total_profit),
        "average_profit_margin": float(average_profit_margin),
        "mom_revenue_growth": float(mom_growth) if pd.notna(mom_growth) else 0,
        "top_profitable_products": [{"product": k, "profit": float(v)} for k, v in top_profitable.items()],
        "least_profitable_products": [{"product": k, "profit": float(v)} for k, v in least_profitable.items()]
    })


if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
