import os
from dotenv.main import load_dotenv
from flask import Flask

load_dotenv()

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), '../templates'),
            static_folder=os.path.join(os.path.dirname(__file__), '../static'))
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']
app.config['PERMANENT_SESSION_LIFETIME'] = 3600
app.config['UPLOAD_FOLDER'] = os.path.abspath('static/uploads')
app.config['OUTPUT_FOLDER'] = os.path.abspath('static/outputs')
app.config['ALLOWED_EXTENSIONS'] = {'.png', '.jpg', '.jpeg', '.gif'}
app.config['DEBUG'] = bool(int(os.environ['DEBUG']))
