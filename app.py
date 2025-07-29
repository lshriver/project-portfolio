from flask import Flask, render_template
import os

app = Flask(__name__)

# Ensure directories exist
def ensure_directories():
    directories = [
        os.path.join('static', 'css'),
        os.path.join('static', 'js'),
        os.path.join('static', 'images'),
        os.path.join('static', 'css', 'tech_bubbles'),
        os.path.join('static', 'css', 'buttons'),
        os.path.join('templates', 'projects')
    ]

    for directory in directories: 
        if not os.path.exists(directory):
            os.makedirs(directory)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/projects/neural-bifurcations')
def neural_bifurcations():
    return render_template('projects/neural_bifurcations.html')

if __name__ == '__main__':
    ensure_directories()
    app.run(host='0.0.0.0', port=5000, debug=True)