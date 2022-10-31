import os
from flask import Flask, request, json, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
import make_model_detect

uploads = './uploads/temp_image'

app = Flask(__name__)
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = uploads

ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Check file type
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def response_config(data, status_code, mime_type):
    return app.response_class(
        response=json.dumps(data),
        status=status_code,
        mimetype=mime_type
    )
@app.route('/')
def home():
    result = ""
    return render_template('index.html')

@app.route('/verify-vehicle', methods=['POST'])
def get_vehicle_identity():
    print("Inside Backend")
    print(request.files)
    if 'file' not in request.files:
        data = {
            'Message': 'Image not found!'
        }
        return response_config(data, 404, 'application/json')

    for image in request.files.getlist('file'):
        print(image)
        if image.filename == '':
            data = {
                'Message': 'No selected image!'
            }
            return response_config(data, 404, 'application/json')

        if image and allowed_file(image.filename):
            try:
                filename = secure_filename(image.filename)
                temp_image = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                image.save(temp_image)


            except Exception as e:
                app.logger.error('Error : ', str(e))

                data = {
                    'Message': str(e)
                }
                return response_config(data, 505, 'application/json')

        else:
            data = {
                'Message': 'Invalid image type!'
            }
            return response_config(data, 404, 'application/json')

    res = make_model_detect.engine()

    return response_config(res, 202, 'application/json')


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=8080, debug=True)
    app.run(debug=True)
