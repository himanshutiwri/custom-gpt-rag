# app/routes.py

from flask import Blueprint, render_template, request, jsonify
import re
from app.utils import format_response
from app.chat import ChatManager

# Create a Blueprint for the main application
main = Blueprint('main', __name__)

# Initialize the ChatManager
chat_manager = ChatManager()  # Ensure this class is defined in app/chat.py

@main.route('/')
def index():
    """Render the chat interface."""
    return render_template('chat.html')  # Renders the chat.html template

@main.route('/test')
def test():
    return render_template('base.html')  # Assuming base.html exists in templates/

@main.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages from users."""
    user_message = request.json.get('message')
    
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    # Get AI response using updated ChatManager logic
    ai_response = chat_manager.get_response(user_message)

    # Format the response if necessary (e.g., replace newlines)
    formatted_response = format_response(ai_response)

    return jsonify({"response": formatted_response})  # Return the formatted AI's response as JSON

def format_response(response):
    # Replace newlines with <br> for line breaks
    response = response.replace('\n', '<br>')
    
    # Convert bold text (e.g., **text**) to <strong> tags
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    
    # Convert italic text (e.g., *text*) to <em> tags
    response = re.sub(r'\*(.*?)\*', r'<em>\1</em>', response)

    # Detect headings (e.g., "## Heading" or "# Heading")
    response = re.sub(r'(?m)^(#+)\s*(.*)', lambda m: f'<h{len(m.group(1))}>{m.group(2)}</h{len(m.group(1))}>', response)

    # Handle lists (e.g., "1. Item" or "- Item")
    response = re.sub(r'(?m)^\d+\.\s*(.*)', r'<li>\1</li>', response)
    response = re.sub(r'(?m)^\-\s*(.*)', r'<li>\1</li>', response)
    
    # Wrap list items in <ul>
    if '<li>' in response:
        response = '<ul>' + response + '</ul>'

    # Handle code blocks (using triple backticks)
    if "```" in response:
        parts = response.split("```")
        formatted_response = ""
        for i, part in enumerate(parts):
            if i % 2 == 1:  # Odd index means it's code
                formatted_response += f'<pre class="code-block"><code>{part}</code></pre>'
            else:
                formatted_response += part.replace('\n', '<br>')  # Replace newlines with <br>
        return formatted_response

    return response