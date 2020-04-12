from flask import Flask, render_template, request
import random
import requests
from werkzeug.utils import secure_filename

app = Flask(__name__)

# serve image
# images = [
#     "https://www.ocregister.com/wp-content/uploads/2017/12/spo_ldn-l-bosa-061.jpg"
# ]

# @app.route('/')
# def index():
#     url = random.choice(images)
#     return render_template('index.html', url=url)

@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")  
 
@app.route('/success', methods = ['POST'])  
def success():  
    if request.method == 'POST':  
        #URL = 'http://127.0.0.1:5000/predict'
        #URL = 'http://0.0.0.0:5000/predict'
        URL = 'http://backend:5000/predict'
        f = request.files['file'] # <class 'werkzeug.datastructures.FileStorage'>
        #print(type(f))  
        # f.save(f.filename)  
        filename = {'file': f.stream.read()}
        # filename = {'file': bytearray(f)}
        # print(filename)
        result = requests.post(URL, files=filename)
        # print(result.text)
        return render_template("success.html", name = result.text, image = f)  
        #return render_template("success.html", name = f.filename)  

if __name__ == "__main__":
    app.run(host="0.0.0.0")