from flask import Flask, request, jsonify
import whisper
import os
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
import google.generativeai as genai
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
import time

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load Whisper model once at startup
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("✓ Whisper model loaded")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=gemini_api_key)

# Optimized Gemini configuration for faster response
model = genai.GenerativeModel(
    model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), 
    generation_config={
        "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        "top_p": 0.9,
        "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "800")),  # Reduced for faster response
        "candidate_count": 1,  # Single candidate for faster processing
    }
)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=3)

SCORING_RUBRIC = {
    "Advanced": {"min": 25, "max": 30},
    "High-Intermediate": {"min": 20, "max": 24},
    "Low-Intermediate": {"min": 16, "max": 19},
    "Basic": {"min": 10, "max": 15},
    "Below Basic": {"min": 0, "max": 9}
}

WRITING_SCORING_RUBRIC = {
    "Advanced": {"min": 24, "max": 30},
    "High-Intermediate": {"min": 17, "max": 23},
    "Low-Intermediate": {"min": 13, "max": 16},
    "Basic": {"min": 7, "max": 12},
    "Below Basic": {"min": 0, "max": 6}
}

# Optimized prompt templates for faster processing
SPEAKING_PROMPT_TEMPLATE = """You are a TOEFL IBT Speaking examiner. Evaluate this response:

Question: {question}
Answer: {answer_text}
Target: {target_level}

Criteria: Content, Language, Fluency, Organization
Scale: Advanced (25-30), High-Intermediate (20-24), Low-Intermediate (16-19), Basic (10-15), Below Basic (0-9)

Use personal tone with "you/your". Return JSON only:
{{
    "score": [0-30],
    "level": "[level name]",
    "target_level": "{target_level}",
    "meets_target": [true/false],
    "feedback": "[brief personal feedback using 'you/your']",
    "strengths": ["strength1", "strength2"]
}}"""

WRITING_PROMPT_TEMPLATE = """You are a TOEFL IBT Writing examiner. Evaluate this response:

Question: {question}
Answer: {answer_text}
Target: {target_level}

Criteria: Content, Organization, Language, Vocabulary, Task Response
Scale: Advanced (24-30), High-Intermediate (17-23), Low-Intermediate (13-16), Basic (7-12), Below Basic (0-6)

Use personal tone with "you/your". Return JSON only:
{{
    "score": [0-30],
    "level": "[level name]",
    "target_level": "{target_level}",
    "meets_target": [true/false],
    "feedback": "[brief personal feedback using 'you/your']",
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"]
}}"""

def get_gemini_assessment_fast(question, answer_text, target_level):
    """
    Optimized Gemini assessment for speaking
    """
    prompt = SPEAKING_PROMPT_TEMPLATE.format(
        question=question,
        answer_text=answer_text,
        target_level=target_level
    )

    try:
        # Use optimized generation with timeout
        start_time = time.time()
        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}  # 15 second timeout
        )
        
        processing_time = time.time() - start_time
        print(f"Gemini response time: {processing_time:.2f}s")
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        assessment['processing_time'] = round(processing_time, 2)
        return assessment
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return {
            "error": "Failed to parse assessment response",
            "details": f"JSON parsing error: {str(e)}",
            "fallback_score": 15,  # Fallback score
            "fallback_level": "Basic"
        }
    except Exception as e:
        print(f"Error getting Gemini assessment: {str(e)}")
        return {
            "error": "Failed to get assessment from Gemini",
            "details": str(e),
            "fallback_score": 15,
            "fallback_level": "Basic"
        }

def get_gemini_writing_assessment_fast(question, answer_text, target_level):
    """
    Optimized Gemini assessment for writing
    """
    prompt = WRITING_PROMPT_TEMPLATE.format(
        question=question,
        answer_text=answer_text,
        target_level=target_level
    )

    try:
        start_time = time.time()
        response = model.generate_content(
            prompt,
            request_options={"timeout": 15}
        )
        
        processing_time = time.time() - start_time
        print(f"Gemini writing response time: {processing_time:.2f}s")
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        assessment['processing_time'] = round(processing_time, 2)
        return assessment
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        return {
            "error": "Failed to parse assessment response",
            "details": f"JSON parsing error: {str(e)}",
            "fallback_score": 15,
            "fallback_level": "Basic"
        }
    except Exception as e:
        print(f"Error getting Gemini writing assessment: {str(e)}")
        return {
            "error": "Failed to get assessment from Gemini",
            "details": str(e),
            "fallback_score": 15,
            "fallback_level": "Basic"
        }

@app.route('/assess-speaking', methods=['POST'])
def assess_speaking():
    """
    Optimized endpoint for TOEFL speaking assessment
    """
    start_time = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    if 'question' not in request.form:
        return jsonify({'error': 'Question is required'}), 400
        
    if 'type' not in request.form:
        return jsonify({'error': 'Target level is required'}), 400

    question = request.form['question']
    target_level = request.form['type']  
    audio_file = request.files['audio']

    valid_levels = ["Advanced", "High-Intermediate", "Low-Intermediate", "Basic", "Below Basic"]
    if target_level not in valid_levels:
        return jsonify({
            'error': 'Invalid target level',
            'valid_levels': valid_levels
        }), 400

    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        # Save file
        audio_file.save(filepath)
        
        # Transcribe audio (this is usually the bottleneck)
        print("Transcribing audio...")
        transcription_start = time.time()
        
        # Use smaller segments for faster processing if file is large
        transcription_result = whisper_model.transcribe(
            filepath,
            fp16=False,  # Disable fp16 if causing issues
            verbose=False  # Reduce logging overhead
        )
        answer_text = transcription_result["text"].strip()
        
        transcription_time = time.time() - transcription_start
        print(f"Transcription time: {transcription_time:.2f}s")
        
        min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "5"))
        if not answer_text or len(answer_text.split()) < min_words:
            return jsonify({
                'error': f'Audio transcription is too short or empty. Please provide a longer response (minimum {min_words} words).',
                'transcription': answer_text,
                'transcription_time': round(transcription_time, 2)
            }), 400
        
        # Get assessment in parallel if needed
        print("Getting Gemini assessment...")
        assessment = get_gemini_assessment_fast(question, answer_text, target_level)
        
        total_time = time.time() - start_time
        
        response_data = {
            'transcription': answer_text,
            'question': question,
            'target_level': target_level,
            'assessment': assessment,
            'status': 'success',
            'performance': {
                'total_time': round(total_time, 2),
                'transcription_time': round(transcription_time, 2),
                'assessment_time': assessment.get('processing_time', 0)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e),
            'processing_time': round(time.time() - start_time, 2)
        }), 500
        
    finally:
        # Clean up file
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/assess-writing', methods=['POST'])
def assess_writing():
    """
    Optimized endpoint for TOEFL writing assessment
    """
    start_time = time.time()
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must contain JSON data'}), 400
    
    if 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
        
    if 'answer' not in data:
        return jsonify({'error': 'Answer is required'}), 400
        
    if 'type' not in data:
        return jsonify({'error': 'Target level (type) is required'}), 400

    question = data['question'].strip()
    answer_text = data['answer'].strip()
    target_level = data['type']

    valid_levels = ["Advanced", "High-Intermediate", "Low-Intermediate", "Basic", "Below Basic"]
    if target_level not in valid_levels:
        return jsonify({
            'error': 'Invalid target level',
            'valid_levels': valid_levels
        }), 400

    min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "10"))
    word_count = len(answer_text.split()) if answer_text else 0
    
    if not answer_text or word_count < min_words:
        return jsonify({
            'error': f'Answer is too short or empty. Please provide a longer response (minimum {min_words} words).',
            'word_count': word_count
        }), 400

    try:
        print("Getting Gemini writing assessment...")
        assessment = get_gemini_writing_assessment_fast(question, answer_text, target_level)
        
        total_time = time.time() - start_time
        
        response_data = {
            'question': question,
            'answer': answer_text,
            'target_level': target_level,
            'word_count': word_count,
            'assessment': assessment,
            'status': 'success',
            'performance': {
                'total_time': round(total_time, 2),
                'assessment_time': assessment.get('processing_time', 0)
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e),
            'processing_time': round(time.time() - start_time, 2)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'whisper_model': 'loaded',
        'gemini_configured': bool(os.getenv("GEMINI_API_KEY")),
        'env_loaded': True,
        'optimization': 'enabled'
    })

@app.route('/scoring-rubric', methods=['GET'])
def get_scoring_rubric():
    """Get TOEFL speaking scoring rubric"""
    return jsonify({
        'speaking_rubric': SCORING_RUBRIC,
        'writing_rubric': WRITING_SCORING_RUBRIC,
        'valid_levels': ["Advanced", "High-Intermediate", "Low-Intermediate", "Basic", "Below Basic"]
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration (without sensitive data)"""
    return jsonify({
        'whisper_model': 'base',
        'gemini_model': os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        'temperature': float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        'max_tokens': int(os.getenv("GEMINI_MAX_TOKENS", "800")),
        'min_words_threshold': int(os.getenv("MIN_WORDS_THRESHOLD", "5")),
        'upload_folder': UPLOAD_FOLDER,
        'optimizations': {
            'reduced_max_tokens': True,
            'timeout_enabled': True,
            'performance_tracking': True,
            'optimized_prompts': True
        }
    })

if __name__ == '__main__':
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in .env file!")
        print("Please create a .env file with your Gemini API key.")
        print("Get your free API key at: https://aistudio.google.com/app/apikey")
        exit(1)
    else:
        print("✓ Gemini API key loaded from .env file")
    
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"✓ Starting optimized Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)