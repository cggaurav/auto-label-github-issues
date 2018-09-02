import os
from flask import Flask
from flask import request
app = Flask(__name__)

# TODO: Connect this
@app.route('/')
def hello_world():
    return 'Hello, Auto Label Github issues!'

@app.route('/predict')
def predict():
	# TODO: Connect to the model
	# /predict?issue=x
	issue = request.args.get('issue')
	return issue

if __name__ == "__main__":
    app.run()