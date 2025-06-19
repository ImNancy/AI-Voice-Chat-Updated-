import os
import logging
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from groq import Groq
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")
CORS(app)

# Initialize Groq client
groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
if not groq_api_key:
    # Use the hardcoded key as fallback
    groq_api_key = "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"

try:
    groq_client = Groq(api_key=groq_api_key)
    app.logger.info("Groq client initialized successfully")
except Exception as e:
    app.logger.error(f"Failed to initialize Groq client: {str(e)}")
    groq_client = None

@app.route('/')
def index():
    """Serve the main chat interface"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests and return AI responses"""
    app.logger.info("Chat endpoint called")
    
    try:
        # Get and validate request data
        if not request.is_json:
            app.logger.error("Request is not JSON")
            return jsonify({'error': 'Request must be JSON', 'success': False}), 400
            
        data = request.get_json()
        app.logger.info(f"Received data: {data}")
        
        if not data:
            app.logger.error("No JSON data received")
            return jsonify({'error': 'No data received', 'success': False}), 400
            
        user_message = data.get('message', '').strip()
        app.logger.info(f"User message: {user_message}")

        if not user_message:
            app.logger.error("Empty message received")
            return jsonify({'error': 'Message is required', 'success': False}), 400

        # Check if Groq client is available
        if not groq_client:
            app.logger.error("Groq client is not available")
            return jsonify({'error': 'Groq API service not available', 'success': False}), 500

        app.logger.info("Calling Groq API...")
        
        # Create chat completion with Groq
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant. Provide concise, friendly responses suitable for voice conversation. Keep responses conversational and not too long."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=150
            )

            ai_response = chat_completion.choices[0].message.content
            app.logger.info(f"AI response generated successfully: {ai_response[:100]}...")

            response_data = {
                'response': ai_response,
                'success': True
            }
            app.logger.info(f"Returning response: {response_data}")
            
            return jsonify(response_data)
            
        except Exception as groq_error:
            app.logger.error(f"Groq API error: {str(groq_error)}", exc_info=True)
            return jsonify({
                'error': f'AI service error: {str(groq_error)}',
                'success': False
            }), 500

    except Exception as e:
        app.logger.error(f"Unexpected error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Server error: {str(e)}',
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'groq_configured': bool(groq_client)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)