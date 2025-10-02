# app/__init__.py

from flask import Flask
import os
import yaml
from dotenv import load_dotenv
from logger import CustomLogger  # Import your custom logger

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """Class to handle application configuration."""
    
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from config.yaml."""
        with open('config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        
        # Replace API key placeholder with actual value from environment variables
        if 'api' in config and 'key' in config['api']:
            config['api']['key'] = os.getenv('API_KEY')  # Get the actual API key from environment
        
        return config

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder='templates')  # Initialize the Flask app

    # Load configuration
    app_config = AppConfig()
    app.config.update(app_config.config)

    # Set up logging
    logger = CustomLogger().get_logger()  # Initialize your custom logger
    logger.info("Flask application starting...")

    # Import and register routes
    from .routes import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app