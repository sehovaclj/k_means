"""Module that contains the server and endpoint functions."""
import json
from flask import Flask, request
from waitress import serve
from k_means.resources.server import HOST, PORT
from k_means.utils.server import run_simulation_util


def rest_serving():
    app = Flask(__name__)

    @app.route('/kMeansEngine/runSimulation', methods=['POST'])
    def run_simulation():
        return run_simulation_util(json.loads(request.data))

    # Run the application
    print(f'\n[Rest API Server] RUNNING on {HOST}:{PORT}\n')
    serve(app, host=HOST, port=PORT)
