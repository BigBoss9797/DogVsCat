import os 
from flask import Flask, request, render_template

from Predict import derive_by_web

app = Flask(__name__, static_url_path = "")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/", methods=["GET"])
def upload_page():
	return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
	root = os.path.join(APP_ROOT, 'static/')
	print(root)
	if not os.path.isdir(root):
		os.mkdir(root)
	print(request.files.getlist("file"))
	for upload in request.files.getlist("file"):
		print(upload)
		print("{} is the file name".format(upload.filename))
		filename = upload.filename
		imgfn = filename
		destination = "/".join([root, imgfn])
		upload.save(destination)
		result = predict_image(destination)
		print("Prediction: ", result)
	return render_template("result.html", image_name = imgfn, result = result)

def predict_image(path):
	print(path)
	return derive_by_web(path)

if __name__ == "__main__":
	app.run(port = 4555, debug =True)