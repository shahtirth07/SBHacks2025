from flask import Flask, render_template, request, jsonify
import os
from chatbot import ask_claude

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        if not user_message:
            return jsonify({'response': 'No message provided.'}), 400

        # Call the `ask_claude` function from chatbot.py
        bot_response = ask_claude(user_message)
        return jsonify({'response': bot_response})
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}'}), 500

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save file (optional)
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Generate summary (replace with actual logic)
    summary = f"Summary of {file.filename}: This is a placeholder summary."

    # Generate example images/graphs (replace with actual logic)
    images = ["/static/images/example1.png", "/static/images/example2.png"]

    return jsonify({'summary': summary, 'images': images})



if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)  # Create uploads directory
    app.run(debug=True)
