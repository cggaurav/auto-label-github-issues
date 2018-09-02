import os
from flask import Flask
from flask import request
api = Flask(__name__)

# TODO: Connect this
@api.route('/')
def hello_world():
    return 'Hello, Auto Label Github issues!'

@api.route('/predict')
def predict():
	# TODO: Connect to the model
	# /predict?issue=x
	issue = request.args.get('issue')
	return issue

if __name__ == "__main__":
    api.run()