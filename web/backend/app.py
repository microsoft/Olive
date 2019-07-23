from flask import Flask, jsonify, request
from flask_cors import CORS
import uuid, sys, json
sys.path.append('../../notebook')
import onnxpipeline
from werkzeug.utils import secure_filename
import netron

BOOKS = [
    {
        'id': uuid.uuid4().hex,
        'title': 'On the Road',
        'author': 'Jack Kerouac',
        'read': True
    },
    {
        'id': uuid.uuid4().hex,
        'title': 'Harry Potter and the Philosopher\'s Stone',
        'author': 'J. K. Rowling',
        'read': False
    },
    {
        'id': uuid.uuid4().hex,
        'title': 'Green Eggs and Ham',
        'author': 'Dr. Seuss',
        'read': True
    }
]


# configuration
DEBUG = True

# instantiate the app
app = Flask(__name__)
app.config.from_object(__name__)

# enable CORS
CORS(app)

@app.route('/visualize', methods=['POST'])
def visualize():
    response_object = {'status': 'failure'}
    if request.method == 'POST':
        response_object = {'status': 'success'}
        temp_model = request.files['file']
        model_name = temp_model.filename

        request.files['file'].save(model_name)

        netron.start(model_name)

    return jsonify(response_object)

@app.route('/convert', methods=['GET', 'POST'])
def convert():
    response_object = {'status': 'success'}
    if request.method == 'POST':

        temp_model = request.files['file']

        model_name = temp_model.filename
        temp_json = 'temp.json'

        request.files['file'].save(model_name)
        request.files['metadata'].save(temp_json)
        
        with open(temp_json, 'r') as f:
            json_data = json.load(f)

        json_data['model'] = model_name

        with open(temp_json, 'w') as f:
            json.dump(json_data, f)

        pipeline = onnxpipeline.Pipeline()
        model, output = pipeline.convert_model(model=model_name, input_json=temp_json)

        response_object['logs'] = output

    return jsonify(response_object)

@app.route('/books/<book_id>', methods=['PUT', 'DELETE'])
def single_book(book_id):
    response_object = {'status': 'success'}
    if request.method == 'PUT':
        post_data = request.get_json()
        remove_book(book_id)
        BOOKS.append({
            'id': uuid.uuid4().hex,
            'title': post_data.get('title'),
            'author': post_data.get('author'),
            'read': post_data.get('read')
        })
        response_object['message'] = 'Book updated!'
    if request.method == 'DELETE':
        remove_book(book_id)
        response_object['message'] = 'Book removed!'
    return jsonify(response_object)

def remove_book(book_id):
    for book in BOOKS:
        if book['id'] == book_id:
            BOOKS.remove(book)
            return True
    return False




if __name__ == '__main__':
    app.run()