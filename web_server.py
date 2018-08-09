import os
import sys, traceback
import argparse
import json
import pprint
#WEB SERVER: FLASK
from flask import Flask, jsonify, abort, make_response, request, render_template, flash, redirect
from flask_cors import CORS, cross_origin
#WEB SERVER: TORNADO
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop
#WEB SERVER: FALCON
#import falcon
#WEB SERVER: WAITRESS
#if os.name == 'nt':
#    from waitress import serve
#GEVENT
from gevent.pywsgi import WSGIServer
from gevent import monkey
# need to patch sockets to make requests async
monkey.patch_all()
import requests
from multiprocessing import cpu_count


can_fork = hasattr(os, "fork")

#FLASK
app = Flask(__name__)
CORS(app)

import __main__
REST_config = __main__.REST_config
current_version = "1.0"
instances={}


default_headers = {'Content-Type': 'application/json', 
        "Access-Control-Allow-Origin" :"*",
        "Access-Control-Allow-Methods":"GET,PUT,POST,DELETE" ,
        "Access-Control-Allow-Headers":"Content-Type"}  

#FLASK & TORNADO-compatible
class LoggingMiddleware(object):
    def __init__(self, app):
        self._app = app

    def __call__(self, environ, resp):
        errorlog = environ['wsgi.errors']
        pprint.pprint(('REQUEST', environ["HTTP_HOST"], environ["REMOTE_ADDR"], environ["REQUEST_METHOD"], environ["PATH_INFO"], environ["QUERY_STRING"]), stream=errorlog)

        def log_response(status, headers, *args):
            pprint.pprint(('RESPONSE', status #, headers
                           ), stream=errorlog)
            return resp(status, headers, *args)

        return self._app(environ, log_response)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

#####################
##      alive      ##
#####################
@app.route(REST_config["route"]+current_version+'/alive', methods=['OPTIONS','GET'])
def alive(): 
    if request.method == 'OPTIONS':
        r = make_response("")
        for key, value in default_headers.items():
            r.headers.add(key, value)
        return r, 200
    if request.method == 'GET':
        r = make_response("alive")
        for key, value in default_headers.items():
            r.headers.add(key, value)
        return r, 200

#####################
##    instances    ##
#####################
@app.route(REST_config["route"]+current_version+'/instances', methods=['OPTIONS','GET'])
def valid_instances(): 
    if request.method == 'OPTIONS':
        r = make_response("")
        for key, value in default_headers.items():
            r.headers.add(key, value)
        return r, 200
    if request.method == 'GET':
        r = make_response(jsonify({"instances":instances}))
        for key, value in default_headers.items():
            r.headers.add(key, value)
        return r, 200

#####################
##    inference    ##
#####################
@app.route(REST_config["route"]+current_version+'/inference', methods=['OPTIONS','POST'])
def inference():       
    if request.method == 'OPTIONS':
        r = make_response("")
        for key, value in default_headers.items():
            r.headers.add(key, value)
        return r, 200
    if request.method == 'POST':
        response_dict={}
        REST_instance = request.args.get('instance',default=None)       
        if REST_instance == None or REST_instance == "":
            return make_response(jsonify({'error': 'Missing (or empty) arg called "instance"'}), 400)

        if REST_instance not in instances:
            return make_response(jsonify({'error': 'Provided "instance" is not a recognized one.'}), 400)


        ##################
        ## missing code ##
        ##################
        return make_response(jsonify({'error': 'Not implemented yet.'}), 400)  

        response = jsonify(response_dict)
        for key, value in default_headers.items():
            response.headers.add(key, value)
        return response, 200

#####################
## batch_inference ##
#####################
@app.route(REST_config["route"]+current_version+'/batch_inference', methods=['OPTIONS','POST'])
def batch_inference(): 
    if request.method == 'OPTIONS':
        r = make_response("")
        for key, value in default_headers.items():
            r.headers.add(key, value)
        return r, 200
    if request.method == 'POST':
        response_dict={}
        ##################
        ## missing code ##
        ##################
        return make_response(jsonify({'error': 'Not implemented yet.'}), 400)   

        response = jsonify(response_dict)
        for key, value in default_headers.items():
            response.headers.add(key, value)
        return response, 200


def run_server():
        for instance in REST_config["REST_instances"]:
            instances[instance["name"]]=instance

        app.wsgi_app = LoggingMiddleware(app.wsgi_app)    
        #app.run(debug=True,
        #        host='0.0.0.0',
        #        port=configurations["port"],
        #        threaded=True,
        #        #threaded= (True if can_fork == False else False),processes =
        #        #(cpu_count() if can_fork else 1),
        #        use_reloader=False)

        ###TORNADO
        ##http_server = HTTPServer(WSGIContainer(app))
        ##http_server.listen(configurations["port"])
        ##IOLoop.current().start()

        #GEVENT
        print("Start gevent WSGI server")
        # use gevent WSGI server instead of the Flask
        http = WSGIServer(('', REST_config["port"]), app.wsgi_app)
        # TODO gracefully handle shutdown
        http.serve_forever()