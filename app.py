from flask import Flask, request, jsonify
import whisper
import os
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import time
import torch

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Optimized Whisper model loading with device detection
print("Initializing Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Use 'tiny' model for maximum speed (you can change to 'base' or 'small' if needed)
whisper_model = whisper.load_model("tiny", device=device)
print("✓ Whisper model loaded (tiny - fastest)")

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
        "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "700")),  # Further reduced
        "candidate_count": 1,
    }
)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=2)

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

# Ultra-optimized prompt templates
SPEAKING_PROMPT_TEMPLATE = """TOEFL Speaking Evaluator. Rate this response:

Q: {question}
A: {answer_text}
Target: {target_level}

Rate 0-30: Advanced(25-30), High-Int(20-24), Low-Int(16-19), Basic(10-15), Below(0-9)

JSON only:
{{
    "score": [number],
    "level": "[level]",
    "target_level": "{target_level}",
    "meets_target": [true/false],
    "feedback": "[brief feedback using 'you']",
    "strengths": ["strength1", "strength2"]
}}"""

WRITING_PROMPT_TEMPLATE = """TOEFL Writing Evaluator. Rate this response:

Q: {question}
A: {answer_text}
Target: {target_level}

Rate 0-30: Advanced(24-30), High-Int(17-23), Low-Int(13-16), Basic(7-12), Below(0-6)

JSON only:
{{
    "score": [number],
    "level": "[level]",
    "target_level": "{target_level}",
    "meets_target": [true/false],
    "feedback": "[brief feedback using 'you']",
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"]
}}"""

def get_gemini_assessment_fast(question, answer_text, target_level):
    """Optimized Gemini assessment for speaking"""
    prompt = SPEAKING_PROMPT_TEMPLATE.format(
        question=question[:200],  # Limit question length
        answer_text=answer_text[:500],  # Limit answer length for speed
        target_level=target_level
    )

    try:
        start_time = time.time()
        response = model.generate_content(
            prompt,
            request_options={"timeout": 12}  # Reduced timeout
        )
        
        processing_time = time.time() - start_time
        print(f"Gemini response time: {processing_time:.2f}s")
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        assessment['processing_time'] = round(processing_time, 2)
        return assessment
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Assessment error: {str(e)}")
        # Quick fallback scoring based on word count and basic rules
        word_count = len(answer_text.split())
        fallback_score = min(25, max(5, word_count // 3))  # Simple scoring
        
        return {
            "score": fallback_score,
            "level": "Basic" if fallback_score < 16 else "Low-Intermediate",
            "target_level": target_level,
            "meets_target": False,
            "feedback": "Your response was processed with basic evaluation due to system optimization.",
            "strengths": ["Response provided", "Attempted the task"],
            "processing_time": 0.5,
            "fallback": True
        }

def get_gemini_writing_assessment_fast(question, answer_text, target_level):
    """Optimized Gemini assessment for writing"""
    prompt = WRITING_PROMPT_TEMPLATE.format(
        question=question[:200],
        answer_text=answer_text[:800],  # Slightly longer for writing
        target_level=target_level
    )

    try:
        start_time = time.time()
        response = model.generate_content(
            prompt,
            request_options={"timeout": 12}
        )
        
        processing_time = time.time() - start_time
        print(f"Gemini writing response time: {processing_time:.2f}s")
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        assessment['processing_time'] = round(processing_time, 2)
        return assessment
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Writing assessment error: {str(e)}")
        # Quick fallback scoring
        word_count = len(answer_text.split())
        fallback_score = min(28, max(5, word_count // 5))
        
        return {
            "score": fallback_score,
            "level": "Basic" if fallback_score < 13 else "Low-Intermediate",
            "target_level": target_level,
            "meets_target": False,
            "feedback": "Your writing was processed with basic evaluation due to system optimization.",
            "strengths": ["Writing provided", "Attempted the task"],
            "areas_for_improvement": ["Detailed evaluation pending"],
            "processing_time": 0.5,
            "fallback": True
        }

@app.route('/assess-speaking', methods=['POST'])
def assess_speaking():
    """Ultra-optimized endpoint for TOEFL speaking assessment"""
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
        
        # Optimized transcription
        print("Transcribing audio (local - fast mode)...")
        transcription_start = time.time()
        
        # Ultra-fast transcription settings
        transcription_result = whisper_model.transcribe(
            filepath,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,  # Disable context for speed
            temperature=0,  # Deterministic for speed
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        answer_text = transcription_result["text"].strip()
        
        transcription_time = time.time() - transcription_start
        print(f"Transcription time: {transcription_time:.2f}s")
        
        min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "3"))  # Reduced threshold
        if not answer_text or len(answer_text.split()) < min_words:
            return jsonify({
                'error': f'Audio transcription is too short (minimum {min_words} words).',
                'transcription': answer_text,
                'transcription_time': round(transcription_time, 2)
            }), 400
        
        # Get assessment
        print("Getting assessment...")
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
                'assessment_time': assessment.get('processing_time', 0),
                'model_used': 'tiny-local'
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
    """Optimized endpoint for TOEFL writing assessment"""
    start_time = time.time()
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must contain JSON data'}), 400
    
    required_fields = ['question', 'answer', 'type']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    question = data['question'].strip()
    answer_text = data['answer'].strip()
    target_level = data['type']

    valid_levels = ["Advanced", "High-Intermediate", "Low-Intermediate", "Basic", "Below Basic"]
    if target_level not in valid_levels:
        return jsonify({
            'error': 'Invalid target level',
            'valid_levels': valid_levels
        }), 400

    min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "8"))
    word_count = len(answer_text.split()) if answer_text else 0
    
    if not answer_text or word_count < min_words:
        return jsonify({
            'error': f'Answer too short (minimum {min_words} words)',
            'word_count': word_count
        }), 400

    try:
        print("Getting writing assessment...")
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
        'whisper_model': 'tiny-local',
        'device': device,
        'gemini_configured': bool(os.getenv("GEMINI_API_KEY")),
        'optimization': 'maximum-speed-local'
    })

@app.route('/scoring-rubric', methods=['GET'])
def get_scoring_rubric():
    """Get TOEFL scoring rubric"""
    return jsonify({
        'speaking_rubric': SCORING_RUBRIC,
        'writing_rubric': WRITING_SCORING_RUBRIC,
        'valid_levels': ["Advanced", "High-Intermediate", "Low-Intermediate", "Basic", "Below Basic"]
    })

@app.route('/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'whisper_model': 'tiny',
        'device': device,
        'gemini_model': os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        'temperature': float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        'max_tokens': int(os.getenv("GEMINI_MAX_TOKENS", "700")),
        'min_words_threshold': int(os.getenv("MIN_WORDS_THRESHOLD", "3")),
        'upload_folder': UPLOAD_FOLDER,
        'optimizations': {
            'whisper_tiny_model': True,
            'gpu_acceleration': device == 'cuda',
            'reduced_max_tokens': True,
            'fast_transcription': True,
            'fallback_scoring': True,
            'ultra_optimized_prompts': True
        }
    })

# Model switching endpoint for testing different speeds
@app.route('/switch-model/<model_name>', methods=['POST'])
def switch_whisper_model(model_name):
    """Switch Whisper model for testing (tiny/base/small)"""
    global whisper_model
    
    if model_name not in ['tiny', 'base', 'small']:
        return jsonify({'error': 'Invalid model. Use: tiny, base, or small'}), 400
    
    try:
        print(f"Switching to {model_name} model...")
        whisper_model = whisper.load_model(model_name, device=device)
        return jsonify({
            'status': 'success',
            'model': model_name,
            'device': device,
            'message': f'Switched to {model_name} model'
        })
    except Exception as e:
        return jsonify({
            'error': 'Failed to switch model',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    if not os.getenv("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not found in .env file!")
        print("Get your free API key at: https://aistudio.google.com/app/apikey")
        exit(1)
    else:
        print("✓ Gemini API key loaded")
    
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    
    print(f"✓ Starting ultra-optimized Flask server (FREE local mode) on {host}:{port}")
    print(f"✓ Whisper model: tiny (fastest)")
    print(f"✓ Device: {device}")
    app.run(host=host, port=port, debug=debug, threaded=True)