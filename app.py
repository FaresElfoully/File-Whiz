from flask import Flask, request, jsonify, session, after_this_request
from flask_cors import CORS
from merged_app import process_and_load_data, query_faiss, clear_database
import os
import shutil
import traceback
import sys
import io
from werkzeug.utils import secure_filename
import uuid
import urllib.parse
import speech_recognition as sr
from io import BytesIO
from pydub import AudioSegment
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tempfile
import wave
import contextlib

# Set up UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

app = Flask(__name__)

# Configure CORS to allow all origins during development
CORS(app, 
     resources={
         r"/api/*": {
             "origins": ["http://localhost:5173"],
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type"],
             "supports_credentials": True,
             "max_age": 3600
         }
     })

# Configure session
app.config.update(
    SECRET_KEY='your-secret-key-here',  # Change this in production
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_COOKIE_SECURE=False,  # Set to True in production with HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    PERMANENT_SESSION_LIFETIME=1800,  # 30 minutes
    UPLOAD_FOLDER='uploads',
    MAX_CONTENT_LENGTH=16 * 1024 * 1024  # 16MB max file size
)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'pptx', 'txt', 'md', 'wav', 'mp3', 'ogg', 'webm'}
FAISS_PATH = 'faiss_index'

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_PATH, exist_ok=True)

# Add rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.after_request
def after_request(response):
    """Ensure proper CORS headers are set"""
    origin = request.headers.get('Origin')
    if origin and origin.startswith('http://localhost:'):
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.before_request
def before_request():
    """Initialize session and handle preflight requests"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        response = app.make_default_options_response()
        return response
        
    # Initialize session if needed
    if 'initialized' not in session:
        session['initialized'] = False
        session['processed_files'] = []
    session.modified = True
    
    # Debug logging
    print("\n=== Request Info ===")
    print(f"Endpoint: {request.endpoint}")
    print(f"Session ID: {session.get('_id', None)}")
    print(f"Session Data: {dict(session)}")
    print(f"Method: {request.method}")
    print(f"Headers: {dict(request.headers)}")

@app.before_request
def check_session():
    """Initialize or check session before each request"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        clear_folders()

@app.route('/')
def index():
    # Return API status instead of template
    return jsonify({
        "status": "ok",
        "message": "FileWhiz API is running"
    })

@app.route('/api/upload-files', methods=['POST'])
@limiter.limit("20 per minute")  # Add specific limit for file uploads
def upload_files():
    """Handle file upload and processing"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files[]')
        if not files:
            return jsonify({'error': 'No files selected'}), 400

        # Clear existing files
        if os.path.exists(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)

        uploaded_files = []
        for file in files:
            if file.filename == '':
                continue

            if not allowed_file(file.filename):
                return jsonify({'error': f'File type not allowed: {file.filename}'}), 400

            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            print(f"Saving file to: {filepath}")
            file.save(filepath)
            uploaded_files.append(filename)

        if not uploaded_files:
            return jsonify({'error': 'No valid files uploaded'}), 400

        print(f"Processing files: {uploaded_files}")
        
        try:
            # Process the files
            process_and_load_data(UPLOAD_FOLDER)
            
            # Update session
            session['initialized'] = True
            session['processed_files'] = uploaded_files
            session.modified = True
            
            print(f"Files processed successfully. Session: {dict(session)}")
            
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded and processed {len(uploaded_files)} files',
                'files': uploaded_files
            })
            
        except Exception as e:
            print(f"Error processing files: {str(e)}")
            traceback.print_exc()
            
            # Clean up on error
            shutil.rmtree(UPLOAD_FOLDER)
            os.makedirs(UPLOAD_FOLDER)
            
            return jsonify({
                'error': f'Error processing files: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"Error in upload_files: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-processed-files', methods=['GET'])
def get_processed_files():
    """Get list of processed files"""
    try:
        if not session.get('initialized', False):
            return jsonify({
                'files': [],
                'message': 'No documents have been processed yet.'
            })
        
        processed_files = session.get('processed_files', [])
        print(f"Processed files in session: {processed_files}")
        
        return jsonify({
            'files': processed_files,
            'message': f'Found {len(processed_files)} processed files.'
        })
    except Exception as e:
        print(f"Error in get_processed_files: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'files': [],
            'error': str(e)
        }), 500

@app.route('/api/clear-session', methods=['POST'])
def clear_session():
    """Clear session and all temporary files"""
    try:
        print("Clearing session...")
        
        # Clear all folders
        clear_folders()
        print("Folders cleared")
        
        # Clear and reinitialize session
        session.clear()
        session['session_id'] = str(uuid.uuid4())
        session['initialized'] = False
        session['processed_files'] = []
        session.modified = True
        
        print(f"New session initialized: {session['session_id']}")
        
        return jsonify({
            'success': True,
            'message': 'New session started successfully'
        })
    except Exception as e:
        print(f"Error clearing session: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/files')
def get_files():
    files = session.get('files', [])
    return jsonify({'files': files})

@app.route('/api/query', methods=['POST'])
def query():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    if 'question' not in request.json:
        return jsonify({'error': 'No question provided'}), 400

    # Check if any files have been processed
    if not session.get('initialized', False) or not session.get('processed_files', []):
        return jsonify({'error': 'No files have been processed yet. Please upload and process some files first.'}), 400

    try:
        result = query_faiss(request.json['question'])
        if 'error' in result:
            return jsonify({'error': result['error']}), 400
            
        return jsonify({
            'response': result['response'],
            'sources': result['sources']
        })
    except Exception as e:
        print(f"Error in query: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_session():
    try:
        # Clear session data
        session.clear()
        
        # Clear upload folder
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/remove-file', methods=['POST'])
def remove_file():
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            return jsonify({
                'success': False,
                'error': 'No filename provided'
            }), 400

        filename = secure_filename(data['filename'])
        
        # Remove from uploads folder
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        # Remove from outputs folder (if exists)
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join('outputs', f"{base_name}.md")
        if os.path.exists(output_path):
            os.remove(output_path)
        
        # Update session
        if 'processed_files' in session:
            session['processed_files'] = [
                f for f in session.get('processed_files', [])
                if f != filename
            ]
            session.modified = True
        
        return jsonify({
            'success': True,
            'message': f'Successfully removed {filename}'
        })
            
    except Exception as e:
        print(f"Error removing file: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/voice-to-text', methods=['POST'])
@limiter.limit("10 per minute")
def voice_to_text():
    """Convert voice recording to text using speech recognition"""
    try:
        print("Processing voice to text request...")
        print(f"Request files: {request.files.keys()}")
        print(f"Request form: {request.form}")
        
        # Get language from request, default to Arabic
        language = request.form.get('language', 'ar-AR')
        print(f"Using language: {language}")
        
        if 'audio' not in request.files:
            print("No 'audio' file in request")
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400
            
        audio_file = request.files['audio']
        print(f"Audio file received: {audio_file.filename}")
        
        if not audio_file:
            print("Audio file is empty")
            return jsonify({
                'success': False,
                'error': 'Empty audio file'
            }), 400

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio:
            print(f"Created temporary file: {temp_audio.name}")
            audio_file.save(temp_audio.name)
            print(f"Saved audio file to temporary location")
            
            try:
                # Initialize recognizer
                print("Initializing speech recognizer...")
                recognizer = sr.Recognizer()
                
                # Adjust recognition settings for Arabic
                recognizer.energy_threshold = 300  # Increase sensitivity
                recognizer.dynamic_energy_threshold = True
                recognizer.pause_threshold = 1.0  # Slightly longer pause for Arabic
                
                # Load the audio file
                print("Loading audio file into recognizer...")
                with sr.AudioFile(temp_audio.name) as source:
                    print("Recording audio from file...")
                    # Record the audio file with adjusted duration
                    audio = recognizer.record(source)
                    
                    print(f"Starting speech recognition in {language}...")
                    # Perform the recognition with Arabic language
                    text = recognizer.recognize_google(audio, language=language)
                    print(f"Recognition successful. Text: {text}")
                    
                    return jsonify({
                        'success': True,
                        'text': text,
                        'language': language
                    })
                    
            except sr.UnknownValueError as e:
                print(f"Speech recognition failed - could not understand audio: {str(e)}")
                error_msg = 'لم نتمكن من فهم الصوت. يرجى التحدث بوضوح والمحاولة مرة أخرى' if language.startswith('ar') else 'Could not understand audio. Please speak clearly and try again.'
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'language': language
                }), 400
            except sr.RequestError as e:
                print(f"Speech recognition service error: {str(e)}")
                error_msg = 'حدث خطأ في خدمة التعرف على الكلام' if language.startswith('ar') else 'Error with the speech recognition service'
                return jsonify({
                    'success': False,
                    'error': f'{error_msg}: {str(e)}',
                    'language': language
                }), 500
            except Exception as e:
                print(f"Unexpected error during recognition: {str(e)}")
                traceback.print_exc()
                error_msg = 'حدث خطأ أثناء معالجة الصوت' if language.startswith('ar') else 'Error processing audio'
                return jsonify({
                    'success': False,
                    'error': f'{error_msg}: {str(e)}',
                    'language': language
                }), 500
            finally:
                # Clean up the temporary file
                try:
                    print(f"Cleaning up temporary file: {temp_audio.name}")
                    os.unlink(temp_audio.name)
                    print("Temporary file cleaned up successfully")
                except Exception as e:
                    print(f"Error removing temporary file: {str(e)}")
                    
    except Exception as e:
        print(f"Error in voice_to_text: {str(e)}")
        traceback.print_exc()
        error_msg = 'حدث خطأ في الخادم' if language.startswith('ar') else 'Server error'
        return jsonify({
            'success': False,
            'error': f'{error_msg}: {str(e)}',
            'language': language
        }), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folders():
    """Clear all files from uploads and outputs folders"""
    try:
        folders = [UPLOAD_FOLDER, FAISS_PATH]
        for folder in folders:
            if os.path.exists(folder):
                print(f"Clearing folder: {folder}")
                shutil.rmtree(folder)
            os.makedirs(folder, exist_ok=True)
            print(f"Recreated folder: {folder}")
        return True
    except Exception as e:
        print(f"Error clearing folders: {str(e)}")
        traceback.print_exc()
        # Create folders if they don't exist
        for folder in [UPLOAD_FOLDER, FAISS_PATH]:
            os.makedirs(folder, exist_ok=True)
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)