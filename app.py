from flask import Flask, request, jsonify
import whisper
import os
from werkzeug.utils import secure_filename
import json
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

whisper_model = whisper.load_model("base")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=gemini_api_key)

model = genai.GenerativeModel(
    model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"), 
    generation_config={
        "temperature": float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
        "top_p": 0.9,
        "max_output_tokens": int(os.getenv("GEMINI_MAX_TOKENS", "1000")),
    }
)

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

def get_gemini_assessment(question, answer_text, target_level):
    """
    Get assessment from Gemini for TOEFL speaking response
    """
    prompt = f"""
    You are a TOEFL Speaking examiner. Please evaluate the following speaking response based on TOEFL speaking criteria.

    Question: {question}
    Student's Answer: {answer_text}
    Target Level: {target_level}

    Please evaluate based on these criteria:
    1. Content Relevance and Development
    2. Language Use and Grammar
    3. Pronunciation and Fluency (based on transcription quality)
    4. Organization and Coherence

    Scoring Scale:
    - Advanced (25-30): Excellent command of English, clear and coherent response
    - High-Intermediate (20-24): Good command with minor issues
    - Low-Intermediate (16-19): Adequate command with some limitations
    - Basic (10-15): Limited command, basic communication
    - Below Basic (0-9): Very limited command, difficult to understand

    Consider the target level "{target_level}" as a reference point for evaluation.

    IMPORTANT: When giving feedback, use a personal tone as if speaking directly to the student. Use "you" and "your" instead of "the student" or "the speaker". For example:
    - Instead of "The response is irrelevant" say "Your response is not relevant to the question"
    - Instead of "The speaker needs to..." say "You need to..."
    - Instead of "The student should..." say "You should..."

    Please provide your response in the following JSON format only (no additional text):
    {{
        "score": [numerical score 0-30],
        "level": "[actual achieved level name]",
        "target_level": "{target_level}",
        "meets_target": [true/false - whether response meets target level],
        "feedback": "[comprehensive personal feedback combining content relevance and overall suggestions using 'you/your']",
        "strengths": ["strength1", "strength2"]
    }}
    """

    try:
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        return assessment
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Raw response: {response.text}")
        return {
            "error": "Failed to parse assessment response",
            "details": f"JSON parsing error: {str(e)}",
            "raw_response": response.text
        }
    except Exception as e:
        print(f"Error getting Gemini assessment: {str(e)}")
        return {
            "error": "Failed to get assessment from Gemini",
            "details": str(e)
        }

def get_gemini_writing_assessment(question, answer_text, target_level):
    """
    Get assessment from Gemini for TOEFL writing response
    """
    prompt = f"""
    You are a TOEFL Writing examiner. Please evaluate the following writing response based on TOEFL writing criteria.

    Question: {question}
    Student's Answer: {answer_text}
    Target Level: {target_level}

    Please evaluate based on these criteria:
    1. Content Relevance and Development of Ideas
    2. Organization and Structure
    3. Language Use and Grammar
    4. Vocabulary Range and Accuracy
    5. Task Response and Coherence

    Scoring Scale:
    - Advanced (24-30): Excellent writing skills, well-developed ideas, sophisticated language use
    - High-Intermediate (17-23): Generally well-organized, adequate development, minor language errors
    - Low-Intermediate (13-16): Adequate organization, some development, noticeable language limitations
    - Basic (7-12): Limited development of ideas, frequent language errors, basic vocabulary
    - Below Basic (0-6): Very limited writing ability, difficult to understand, severe language problems

    Consider the target level "{target_level}" as a reference point for evaluation.

    IMPORTANT: When giving feedback, use a personal tone as if speaking directly to the student. Use "you" and "your" instead of "the student" or "the writer". For example:
    - Instead of "The essay lacks organization" say "Your essay needs better organization"
    - Instead of "The writer should..." say "You should..."
    - Instead of "The response demonstrates..." say "Your response demonstrates..."

    Check specifically for:
    - Relevance of the answer to the question asked
    - Logical flow and structure of ideas
    - Grammar and syntax accuracy
    - Vocabulary appropriateness and variety
    - Overall coherence and clarity

    Please provide your response in the following JSON format only (no additional text):
    {{
        "score": [numerical score 0-30],
        "level": "[actual achieved level name]",
        "target_level": "{target_level}",
        "meets_target": [true/false - whether response meets target level],
        "feedback": "[comprehensive personal feedback focusing on content relevance, organization, and language use using 'you/your']",
        "strengths": ["strength1", "strength2"],
        "areas_for_improvement": ["area1", "area2"]
    }}
    """

    try:
        response = model.generate_content(prompt)
        
        response_text = response.text.strip()
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        assessment = json.loads(response_text)
        return assessment
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {str(e)}")
        print(f"Raw response: {response.text}")
        return {
            "error": "Failed to parse assessment response",
            "details": f"JSON parsing error: {str(e)}",
            "raw_response": response.text
        }
    except Exception as e:
        print(f"Error getting Gemini writing assessment: {str(e)}")
        return {
            "error": "Failed to get assessment from Gemini",
            "details": str(e)
        }

@app.route('/assess-speaking', methods=['POST'])
def assess_speaking():
    """
    Main endpoint for TOEFL speaking assessment
    Expects: question, audio file, and type (target level)
    Returns: transcription, feedback, and score
    """
    
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
    audio_file.save(filepath)

    try:
        print("Transcribing audio...")
        transcription_result = whisper_model.transcribe(filepath)
        answer_text = transcription_result["text"].strip()
        
        min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "5"))
        if not answer_text or len(answer_text.split()) < min_words:
            return jsonify({
                'error': f'Audio transcription is too short or empty. Please provide a longer response (minimum {min_words} words).',
                'transcription': answer_text
            }), 400
        
        print("Getting Gemini assessment...")
        assessment = get_gemini_assessment(question, answer_text, target_level)
        
        if "error" in assessment:
            return jsonify({
                'transcription': answer_text,
                'assessment_error': assessment
            }), 500
        
        response_data = {
            'transcription': answer_text,
            'question': question,
            'target_level': target_level,
            'assessment': assessment,
            'status': 'success'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e)
        }), 500
        
    finally:
        try:
            os.remove(filepath)
        except:
            pass

@app.route('/assess-writing', methods=['POST'])
def assess_writing():
    """
    Main endpoint for TOEFL writing assessment
    Expects: question, answer, and type (target level) as text
    Returns: feedback and score
    """
    
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

    min_words = int(os.getenv("MIN_WORDS_THRESHOLD", "10"))  # Writing might need more words
    if not answer_text or len(answer_text.split()) < min_words:
        return jsonify({
            'error': f'Answer is too short or empty. Please provide a longer response (minimum {min_words} words).',
            'word_count': len(answer_text.split()) if answer_text else 0
        }), 400

    try:
        print("Getting Gemini writing assessment...")
        assessment = get_gemini_writing_assessment(question, answer_text, target_level)
        
        if "error" in assessment:
            return jsonify({
                'question': question,
                'answer': answer_text,
                'assessment_error': assessment
            }), 500
        
        # Return complete response
        response_data = {
            'question': question,
            'answer': answer_text,
            'target_level': target_level,
            'word_count': len(answer_text.split()),
            'assessment': assessment,
            'status': 'success'
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'whisper_model': 'loaded',
        'gemini_configured': bool(os.getenv("GEMINI_API_KEY")),
        'env_loaded': True
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
        'max_tokens': int(os.getenv("GEMINI_MAX_TOKENS", "1000")),
        'min_words_threshold': int(os.getenv("MIN_WORDS_THRESHOLD", "5")),
        'upload_folder': UPLOAD_FOLDER
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
    
    print(f"✓ Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)