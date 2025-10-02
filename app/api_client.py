# app/api_client.py
from dotenv import load_dotenv
import os
from groq import Groq
from logger import CustomLogger  # Import your custom logger
# Load environment variable from .env file
load_dotenv()

class GroqClient:
    """Class to interact with the Groq API."""

    def __init__(self):
        self.api_key = os.getenv('API_KEY')  # Get the actual API key from environment
        self.client = Groq(api_key=self.api_key)
        self.logger = CustomLogger().get_logger()  # Initialize your custom logger

    def get_response(self, messages):
        """
        Send messages to the Groq API and return the response.

        :param messages: List of messages for the conversation.
        :return: AI response as a string.
        """
        try:
            self.logger.info("Sending messages to Groq API...")
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama-3.1-8b-instant"# Use the appropriate model for general use
            )
            response = chat_completion.choices[0].message.content
            self.logger.info("Received response from Groq API.")
            return response
        except Exception as e:
            self.logger.error(f"Error communicating with Groq API: {e}")
            return "Sorry, I couldn't get a response at this time."