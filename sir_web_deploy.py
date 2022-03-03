import flask
import os
from flask import send_from_directory

app = flask.Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return "Deploying minus 20 hours on Max"

if __name__ == "__main__":
    app.secret_key = 'Epidemic_Enjoyers'
    app.debug = True
    app.run()