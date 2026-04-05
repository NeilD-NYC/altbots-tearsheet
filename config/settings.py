# config/settings.py
# Application configuration — loads from environment variables

import os

DEBUG = os.getenv("DEBUG", "false").lower() == "true"
