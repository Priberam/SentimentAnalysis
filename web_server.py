import os
import sys, traceback
import argparse
import json
import pprint
#DEBUG# #WEB SERVER: FLASK
#DEBUG# from flask import Flask, jsonify, abort, make_response, request, render_template, flash, redirect
#DEBUG# from flask_cors import CORS, cross_origin
#DEBUG# #WEB SERVER: TORNADO
#DEBUG# from tornado.wsgi import WSGIContainer
#DEBUG# from tornado.httpserver import HTTPServer
#DEBUG# from tornado.ioloop import IOLoop
#DEBUG# #WEB SERVER: FALCON
#DEBUG# #import falcon
#DEBUG# #WEB SERVER: WAITRESS
#DEBUG# #if os.name == 'nt':
#DEBUG# #    from waitress import serve
#DEBUG# #GEVENT
#DEBUG# from gevent.pywsgi import WSGIServer
#DEBUG# from gevent import monkey
#DEBUG# # need to patch sockets to make requests async
#DEBUG# monkey.patch_all()
#DEBUG# import requests
#DEBUG# from multiprocessing import cpu_count
#DEBUG# 
#DEBUG# 
#DEBUG# can_fork = hasattr(os, "fork")
#DEBUG# 
#DEBUG# #FLASK
#DEBUG# app = Flask(__name__)
#DEBUG# CORS(app)

import core
import __main__
REST_config = __main__.REST_config
current_version = "1.0"
instances={}

config={}
config["port"]=REST_config["port"]
config["route"]=REST_config["route"]
if "<version>" in config["route"]:
    config["route"] = config["route"].replace("<version>", current_version)
print("Port: ", config["port"], "\tRoute: ", config["route"] )



#DEBUG#  default_headers = {'Content-Type': 'application/json', 
#DEBUG#          "Access-Control-Allow-Origin" :"*",
#DEBUG#          "Access-Control-Allow-Methods":"GET,PUT,POST,DELETE" ,
#DEBUG#          "Access-Control-Allow-Headers":"Content-Type"}  
#DEBUG#  
#DEBUG#  #FLASK & TORNADO-compatible
#DEBUG#  class LoggingMiddleware(object):
#DEBUG#      def __init__(self, app):
#DEBUG#          self._app = app
#DEBUG#  
#DEBUG#      def __call__(self, environ, resp):
#DEBUG#          errorlog = environ['wsgi.errors']
#DEBUG#          pprint.pprint(('REQUEST', \
#DEBUG#                         environ["HTTP_HOST"], \
#DEBUG#                          environ["REMOTE_ADDR"], \
#DEBUG#                          environ["REQUEST_METHOD"], \
#DEBUG#                          environ["PATH_INFO"], \
#DEBUG#                          environ["QUERY_STRING"]), \
#DEBUG#                          stream=errorlog)
#DEBUG#  
#DEBUG#          def log_response(status, headers, *args):
#DEBUG#              pprint.pprint(('RESPONSE', status #, headers
#DEBUG#                             ), stream=errorlog)
#DEBUG#              return resp(status, headers, *args)
#DEBUG#  
#DEBUG#          return self._app(environ, log_response)
#DEBUG#  
#DEBUG#  
#DEBUG#  @app.errorhandler(404)
#DEBUG#  def not_found(error):
#DEBUG#      return make_response(jsonify({'error': 'Not found'}), 404)
#DEBUG#  
#DEBUG#  #####################
#DEBUG#  ##      alive      ##
#DEBUG#  #####################
#DEBUG#  @app.route(config["route"]+'/alive', methods=['OPTIONS','GET'])
#DEBUG#  def alive(): 
#DEBUG#      if request.method == 'OPTIONS':
#DEBUG#          r = make_response("")
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              r.headers.add(key, value)
#DEBUG#          return r, 200
#DEBUG#      if request.method == 'GET':
#DEBUG#          r = make_response("alive")
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              r.headers.add(key, value)
#DEBUG#          return r, 200
#DEBUG#  
#DEBUG#  #####################
#DEBUG#  ##    instances    ##
#DEBUG#  #####################
#DEBUG#  @app.route(config["route"]+'/instances', methods=['OPTIONS','GET'])
#DEBUG#  def valid_instances(): 
#DEBUG#      if request.method == 'OPTIONS':
#DEBUG#          r = make_response("")
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              r.headers.add(key, value)
#DEBUG#          return r, 200
#DEBUG#      if request.method == 'GET':
#DEBUG#          r = make_response(jsonify({"instances":instances}))
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              r.headers.add(key, value)
#DEBUG#          return r, 200
#DEBUG#  
#DEBUG#  #####################
#DEBUG#  ##    inference    ##
#DEBUG#  #####################
#DEBUG#  @app.route(config["route"]+'/inference', methods=['OPTIONS','POST'])
#DEBUG#  def inference():       
#DEBUG#      if request.method == 'OPTIONS':
#DEBUG#          r = make_response("")
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              r.headers.add(key, value)
#DEBUG#          return r, 200
#DEBUG#      if request.method == 'POST':
#DEBUG#          response_dict={}
#DEBUG#          REST_instance = request.args.get('instance',default=None)       
#DEBUG#          if REST_instance == None or REST_instance == "":
#DEBUG#              return make_response(jsonify({'error': 'Missing (or empty) arg called "instance"'}), 400)
#DEBUG#  
#DEBUG#          if REST_instance not in instances:
#DEBUG#              return make_response(jsonify({'error': 'Provided "instance" is not a recognized one.'}), 400)
#DEBUG#  
#DEBUG#          text = request.data
#DEBUG#          if text==None or text=="":
#DEBUG#              return make_response(jsonify({'error': 'Data field is empty: no text to process.'}), 400)
#DEBUG#  
#DEBUG#          instance = instances[REST_instance]
#DEBUG#          response_dict["prediction"]= core.predict(instance, text)
#DEBUG#  
#DEBUG#          response = jsonify(response_dict)
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              response.headers.add(key, value)
#DEBUG#          return response, 200
#DEBUG#  
#DEBUG#  #####################
#DEBUG#  ## batch_inference ##
#DEBUG#  #####################
#DEBUG#  @app.route(config["route"]+'/batch_inference', methods=['OPTIONS','POST'])
#DEBUG#  def batch_inference(): 
#DEBUG#      if request.method == 'OPTIONS':
#DEBUG#          r = make_response("")
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              r.headers.add(key, value)
#DEBUG#          return r, 200
#DEBUG#      if request.method == 'POST':
#DEBUG#          response_dict={}
#DEBUG#  
#DEBUG#          ##################
#DEBUG#          ## missing code ##
#DEBUG#          ##################
#DEBUG#  
#DEBUG#          return make_response(jsonify({'error': 'Not implemented yet.'}), 400)   
#DEBUG#  
#DEBUG#          response = jsonify(response_dict)
#DEBUG#          for key, value in default_headers.items():
#DEBUG#              response.headers.add(key, value)
#DEBUG#          return response, 200


def run_server():
        core.load_instances(REST_config, instances)

        instance = instances["EN200Twitter"]
        print( core.predict(instance, "why does tom cruise take so many times to figure things out in the movie edge of tomorrow , but gets it right 1 st time in mission impossible ?"))

#DEBUG#    app.wsgi_app = LoggingMiddleware(app.wsgi_app)    
#DEBUG#    #app.run(debug=True,
#DEBUG#    #        host='0.0.0.0',
#DEBUG#    #        port=configurations["port"],
#DEBUG#    #        threaded=True,
#DEBUG#    #        #threaded= (True if can_fork == False else False),processes =
#DEBUG#    #        #(cpu_count() if can_fork else 1),
#DEBUG#    #        use_reloader=False)
#DEBUG#
#DEBUG#    ###TORNADO
#DEBUG#    ##http_server = HTTPServer(WSGIContainer(app))
#DEBUG#    ##http_server.listen(configurations["port"])
#DEBUG#    ##IOLoop.current().start()
#DEBUG#
#DEBUG#    #GEVENT
#DEBUG#    print("Start gevent WSGI server")
#DEBUG#    # use gevent WSGI server instead of the Flask
#DEBUG#    http = WSGIServer(('', config["port"]), app.wsgi_app)
#DEBUG#    # TODO gracefully handle shutdown
#DEBUG#    http.serve_forever()