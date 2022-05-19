# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import pickle
import signal
import sys
import traceback

import flask
import pandas as pd
from model_handler import LDAWrapper

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route("/ping", methods=["GET"])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    try:
        LDAWrapper.initialize()
        health = LDAWrapper.initialized
    except:
        print("error occurred loading LDA")
        health = False

    status = 200 if health else 404
    return flask.Response(response="\n", status=status, mimetype="application/json")


@app.route("/invocations", methods=["POST"])
def transformation():
    """
    Do an inference on a single batch of data. Text to JSON.
    """
    data = None

    if not LDAWrapper.initialized:
        # break fast if there's an issue....
        LDAWrapper.initialize()

    if flask.request.content_type == "text/plain":
        data = flask.request.data.decode("utf-8")
    else:
        return flask.Response(
            response="This predictor only supports regular text data", status=415, mimetype="text/plain"
        )

    if data == "":
        return flask.Response(
            response="This predictor does not support empty data", status=415, mimetype="text/plain"
        )

    print("Prediction on:\n", data)
    # Do the prediction
    prediction = LDAWrapper.handle(data)
    result = flask.jsonify(prediction)
    return result
