import os
from flask import render_template, request, redirect, url_for, make_response, Response, send_from_directory
from werkzeug.utils import secure_filename
from config import logger, app
import shutil
import uuid
from filters import apply_blur, apply_grey, apply_sharpen, apply_emboss, apply_sobel, apply_canny, \
                    apply_gaussian_blur, apply_invert, apply_sepia, apply_vignette, apply_posterize, \
                    adjust_saturation

filters = {'blur': 'blur.jpg',
           'grey': 'grey.jpg',
           'sharpen': 'sharpen.jpg',
           'emboss': 'emboss.jpg',
           'sobel': 'sobel.jpg',
           'canny': 'canny.jpg',
           'gaussian blur': 'gaussian_blur.jpg',
           'invert': 'invert.jpg',
           'sepia': 'sepia.jpg',
           'vignette': 'vignette.jpg',
           'posterize': 'posterize.jpg',
           'adjust saturation': 'adjust_saturation.jpg',
          }

filters_methods = {'blur': apply_blur,
                   'grey': apply_grey,
                   'sharpen': apply_sharpen,
                   'emboss': apply_emboss,
                   'sobel': apply_sobel,
                   'canny': apply_canny,
                   'gaussian_blur': apply_gaussian_blur,
                   'invert': apply_invert,
                   'sepia': apply_sepia,
                   'vignette': apply_vignette,
                   'posterize': apply_posterize,
                   'adjust_saturation': adjust_saturation,
                  }

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
                           img_filename=_get_img_filename(),
                           filters=filters)


@app.route('/editor/apply-filter/<filter_name>', methods=['GET'])
def apply_filter(filter_name):
    if filter_name in filters:
        session_id = _get_session_id()
        img_filename = _get_img_filename()
        uploads_dir = _get_uploads_dir()
        outputs_dir = _get_outputs_dir()

        original_img_path = os.path.join(uploads_dir, img_filename)
        filtered_img_filename = f'filtered_{img_filename}'
        filtered_img_path = os.path.join(outputs_dir, filtered_img_filename)

        if filter_name in filters_methods:
            is_filter_applied = filters_methods[filter_name](original_img_path, filtered_img_path)

            if not is_filter_applied:
                return redirect(url_for('edit_img'))

        return render_template('apply_filter.html',
                               session_id=session_id,
                               img_filename=img_filename,
                               filter_name=filter_name,
                               filtered_img_filename=filtered_img_filename,
                               filters=filters)

    return redirect(url_for('edit_img'))


@app.route('/save-filtered/<filtered_img_filename>/')
def save_filtered_image(filtered_img_filename):
    outputs_dir = _get_outputs_dir()
    return send_from_directory(outputs_dir, filtered_img_filename, as_attachment=True)


@app.route('/save-current/<img_filename>/')
def save_current_image(img_filename):
    uploads_dir = _get_uploads_dir()
    return send_from_directory(uploads_dir, img_filename, as_attachment=True)


@app.route('/make-current-image/<filtered_img_filename>', methods=['GET'])
def switch_img_to_current(filtered_img_filename):
    """Make the filtered image the current image for further filtering."""
    previous_img_filename = _get_img_filename()
    uploads_dir = _get_uploads_dir()
    outputs_dir = _get_outputs_dir()

    filtered_img_filepath = os.path.join(outputs_dir, filtered_img_filename)
    new_img_filepath = os.path.join(uploads_dir, previous_img_filename)
    new_img_filename = os.path.basename(new_img_filepath)

    if os.path.exists(new_img_filepath):
        os.remove(new_img_filepath)

    if os.path.exists(filtered_img_filepath):
        os.rename(filtered_img_filepath, new_img_filepath)

    response = make_response(redirect(url_for('edit_img')))
    response.set_cookie('img_filename', new_img_filename)

    return response


if __name__ == '__main__':
    app.run(debug=True)
