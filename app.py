import os
from flask import render_template, request, redirect, url_for, make_response, Response
from werkzeug.utils import secure_filename
from config import logger, app
from PIL import Image
import shutil
import uuid


def _is_allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def _set_session_id(response: Response):
    current_session_id = request.cookies.get('session_id')

    if not current_session_id:
        response.set_cookie('session_id', str(uuid.uuid4()))


def _get_session_id() -> str:
    return request.cookies.get('session_id', '')


def _get_uploads_dir():
    """Get absolute path of uploads directory for current session."""
    session_id = _get_session_id()
    return os.path.join(app.config.get('UPLOAD_FOLDER'), session_id)


def _get_outputs_dir():
    """Get absolute path of outputs directory for current session."""
    session_id = _get_session_id()
    return os.path.join(app.config.get('OUTPUT_FOLDER'), session_id)


def _get_img_filename():
    return request.cookies.get('img_filename')


def _create_dirs():
    """Prepare the session before each request."""
    os.makedirs(_get_uploads_dir(), exist_ok=True)
    os.makedirs(_get_outputs_dir(), exist_ok=True)


@app.route('/clear_data', methods=['POST'])
def clear_data():
    """Clear all files in the uploads and outputs directories for the current session."""
    session_id = _get_session_id()

    uploads_dir = _get_uploads_dir()
    outputs_dir = _get_outputs_dir()

    if os.path.exists(uploads_dir):
        shutil.rmtree(uploads_dir)

    if os.path.exists(outputs_dir):
        shutil.rmtree(outputs_dir)

    response = make_response(redirect(url_for('home')))
    response.set_cookie('img_filename', '', expires=0)

    return response


@app.route('/', methods=['GET', 'POST'])
def home():
    """Handle image upload and serve the upload page."""
    if request.method == 'GET':
        response = make_response(render_template('upload.html'))
        _set_session_id(response)
        return response

    if request.method == 'POST':
        _create_dirs()
        uploaded_img = request.files.get('uploaded_img')

        if uploaded_img and _is_allowed_file(uploaded_img.filename):
            uploaded_img_filename = secure_filename(uploaded_img.filename)
            uploads_dir = _get_uploads_dir()
            logger.debug(f'uploads_dir = {uploads_dir}')
            if not uploads_dir:
                logger.error(f"Uploads directory wasn't created.")
                return redirect(url_for('home'))

            img_filepath = os.path.join(uploads_dir, uploaded_img_filename)

            try:
                uploaded_img.save(img_filepath)
                response = make_response(render_template('upload.html',
                                                         session_id=_get_session_id(),
                                                         img_filename=uploaded_img_filename))
                response.set_cookie('img_filename', uploaded_img_filename)
                return response

            except Exception as e:
                logger.error(f"Error saving uploaded image: {e}")
                return redirect(url_for('home'))

        logger.warning("Invalid file uploaded.")
        return redirect(url_for('home'))

    return "Wrong HTTP method"


@app.route('/editor/', methods=['GET'])
def edit_img():
    """Serve the image editing page."""
    return render_template('editor.html',
                           session_id=_get_session_id(),
                           img_filename=_get_img_filename())


if __name__ == '__main__':
    app.run(debug=True)
