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

print("Initializing Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

whisper_model = whisper.load_model("tiny", device=device)
print("✓ Whisper model loaded (tiny - fastest)")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(
    model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), 
    generation_config={
        "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        "top_p": 0.9,
        "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "700")),  
        "candidate_count": 1,
    }
)

executor = ThreadPoolExecutor(max_workers=2)


SPEAKING_PROMPT_TEMPLATE = """TOEFL IBT Speaking Evaluator. Rate this speaking response on a scale of 0-7.5:

Question: {question}
Response: {answer_text}

Evaluate based on delivery, language use, and topic development.
Give a score from 0 to 7.5 (decimals allowed).

Return JSON only:
{{
    "score": [decimal number 0-7.5],
    "feedback": "[brief constructive feedback using 'you']",
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"]
}}"""

WRITING_PROMPT_TEMPLATE = """TOEFL IBT Writing Evaluator. Rate this writing response on a scale of 0-30:

Question: {question}
Response: {answer_text}

Evaluate based on organization, language use, vocabulary, and coherence.
Give a score from 0 to 30 (whole numbers only).

Return JSON only:
{{
    "score": [number 0-30],
    "feedback": "[brief constructive feedback using 'you']",
    "strengths": ["strength1", "strength2"],
    "areas_for_improvement": ["area1", "area2"]
}}"""

def get_gemini_assessment_fast(question, answer_text):
    """Optimized Gemini assessment for speaking (0-7.5 scale)"""
    prompt = SPEAKING_PROMPT_TEMPLATE.format(
        question=question[:500],  
        answer_text=answer_text[:500]  
    )

    try:
        start_time = time.time()
        response = model.generate_content(
            prompt,
            request_options={"timeout": 12} 
        )
        
        processing_time = time.time() - start_time
        print(f"Gemini response time: {processing_time:.2f}s")
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        assessment['processing_time'] = round(processing_time, 2)
        
        if 'score' in assessment:
            assessment['score'] = max(0, min(7.5, float(assessment['score'])))
        
        return assessment
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Assessment error: {str(e)}")
        word_count = len(answer_text.split())
        fallback_score = min(7.5, max(1.0, word_count / 20))  # Simple scoring for 0-7.5
        
        return {
            "score": round(fallback_score, 1),
            "feedback": "Your response was processed with basic evaluation due to system optimization.",
            "strengths": ["Response provided", "Attempted the task"],
            "areas_for_improvement": ["Detailed evaluation pending"],
            "processing_time": 0.5,
            "fallback": True
        }

def get_gemini_writing_assessment_fast(question, answer_text):
    """Optimized Gemini assessment for writing (0-30 scale)"""
    prompt = WRITING_PROMPT_TEMPLATE.format(
        question=question[:500],
        answer_text=answer_text[:800]  
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
        
        if 'score' in assessment:
            assessment['score'] = max(0, min(30, int(assessment['score'])))
        
        return assessment
        
    except (json.JSONDecodeError, Exception) as e:
        print(f"Writing assessment error: {str(e)}")
        word_count = len(answer_text.split())
        fallback_score = min(28, max(5, word_count // 5))
        
        return {
            "score": fallback_score,
            "feedback": "Your writing was processed with basic evaluation due to system optimization.",
            "strengths": ["Writing provided", "Attempted the task"],
            "areas_for_improvement": ["Detailed evaluation pending"],
            "processing_time": 0.5,
            "fallback": True
        }

@app.route('/assess-speaking', methods=['POST'])
def assess_speaking():
    """Ultra-optimized endpoint for TOEFL speaking assessment (0-7.5 scale)"""
    start_time = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400
    
    if 'question' not in request.form:
        return jsonify({'error': 'Question is required'}), 400

    question = request.form['question']
    audio_file = request.files['audio']

    filename = secure_filename(audio_file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    
    try:
        audio_file.save(filepath)
        
        print("Transcribing audio (local - fast mode)...")
        transcription_start = time.time()
        
        transcription_result = whisper_model.transcribe(
            filepath,
            fp16=False,
            verbose=False,
            condition_on_previous_text=False,  
            temperature=0,  
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6
        )
        answer_text = transcription_result["text"].strip()
        
        transcription_time = time.time() - transcription_start
        print(f"Transcription time: {transcription_time:.2f}s")
        
        min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "3"))  
        if not answer_text or len(answer_text.split()) < min_words:
            return jsonify({
                'error': f'Audio transcription is too short (minimum {min_words} words).',
                'transcription': answer_text,
                'transcription_time': round(transcription_time, 2)
            }), 400
        
        print("Getting assessment...")
        assessment = get_gemini_assessment_fast(question, answer_text)
        
        total_time = time.time() - start_time
        
        response_data = {
            'transcription': answer_text,
            'question': question,
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
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except:
            pass

@app.route('/assess-writing', methods=['POST'])
def assess_writing():
    """Optimized endpoint for TOEFL writing assessment (0-30 scale)"""
    start_time = time.time()
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request must contain JSON data'}), 400
    
    required_fields = ['question', 'answer']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} is required'}), 400

    question = data['question'].strip()
    answer_text = data['answer'].strip()

    min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "8"))
    word_count = len(answer_text.split()) if answer_text else 0
    
    if not answer_text or word_count < min_words:
        return jsonify({
            'error': f'Answer too short (minimum {min_words} words)',
            'word_count': word_count
        }), 400

    try:
        print("Getting writing assessment...")
        assessment = get_gemini_writing_assessment_fast(question, answer_text)
        
        total_time = time.time() - start_time
        
        response_data = {
            'question': question,
            'answer': answer_text,
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
        'optimization': 'maximum-speed-local',
        'scoring_system': {
            'speaking': '0-7.5 scale (direct scoring)',
            'writing': '0-30 scale (direct scoring)'
        }
    })

@app.route('/scoring-info', methods=['GET'])
def get_scoring_info():
    """Get TOEFL scoring information"""
    return jsonify({
        'speaking_scale': '0-7.5 (decimals allowed)',
        'writing_scale': '0-30 (whole numbers)',
        'description': 'Direct scoring without rubric categories'
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
        'scoring_system': {
            'speaking_range': '0-7.5 (decimals)',
            'writing_range': '0-30 (integers)',
            'no_rubric_categories': True
        },
        'optimizations': {
            'whisper_tiny_model': True,
            'gpu_acceleration': device == 'cuda',
            'reduced_max_tokens': True,
            'fast_transcription': True,
            'fallback_scoring': True,
            'ultra_optimized_prompts': True,
            'no_target_level_required': True
        }
    })

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
    print(f"✓ Scoring: Speaking (0-7.5), Writing (0-30)")
    app.run(host=host, port=port, debug=debug, threaded=True)