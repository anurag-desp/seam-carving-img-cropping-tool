import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
from src.seam_carve import SeamCarve

UPLOAD_FOLDER = './static/img/input/'
OUTPUT_FOLDER = './static/img/output/'
ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

img_paths = {
    'upload_path' : None,
    'output_path' : None
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            
            img_paths['upload_path'] = upload_path
            img_paths['output_path'] = output_path
            
            file.save(upload_path)
            return render_template('/html/home.html', filename=filename)
        

            
    return render_template('/html/home.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='/img/input/' + filename), code=301)

@app.route('/crop', methods=["GET", "POST"])
def crop_image():
    if img_paths['upload_path'] is None:
        return render_template('/html/home.html')
    axis_to_crop = request.form['crop-axis']
    scale = request.form['scale']
    seam_carve = SeamCarve(
        image_axis_to_crop=axis_to_crop, 
        crop_scale=float(scale),
        src_path=img_paths['upload_path'],
        dest_path=img_paths['output_path'])
    seam_carve.crop()
    return send_file(img_paths['output_path'], mimetype='png', as_attachment=True, download_name='cropped-' + img_paths['output_path'].split('/')[-1])
    
# @app.route('/download', methods=["GET"])
# def download():
#     return send_file(img_paths['output_path'], as_attachment=True)