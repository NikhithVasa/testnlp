#!flask/bin/python
import json
from flask import Flask, Response, request
import optparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

application = Flask(__name__)

@application.route('/', methods=['GET'])
def get():
    return Response(json.dumps({'Output': 'Hello World transformers working!!'}), mimetype='application/json', status=200)

@application.route('/', methods=['POST'])
def post():
    return Response(json.dumps({'Output': 'Hello World1'}), mimetype='application/json', status=200)


@application.route('/analyzeTranscript', methods=['GET'])
def fetchAnswer():
    question = request.args.get('question')
    prediction = email_classifier(question)
    if prediction == "LABEL_1":
        print("This is a Question")
        return "This is a Question"
    else:
        print("This is not a Question")
        return "This is not a Question"

def email_classifier(text):
    """
    Tokenizes a given sentence and returns the predicted class. 
    
    Returns:
    LABEL_0 --> sentence is predicted as a statement
    LABEL_1 --> sentence is predicted as a question
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")
    model = AutoModelForSequenceClassification.from_pretrained(
        "shahrukhx01/question-vs-statement-classifier")

    inputs = tokenizer(f"{text}", return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    return model.config.id2label[predicted_class_id]

if __name__ == '__main__':
    default_port = "80"
    default_host = "0.0.0.0"
    parser = optparse.OptionParser()
    parser.add_option("-H", "--host",
                      help=f"Hostname of Flask app {default_host}.",
                      default=default_host)

    parser.add_option("-P", "--port",
                      help=f"Port for Flask app {default_port}.",
                      default=default_port)

    parser.add_option("-d", "--debug",
                      action="store_true", dest="debug",
                      help=optparse.SUPPRESS_HELP)

    options, _ = parser.parse_args()

    application.run(
        debug=options.debug,
        host=options.host,
        port=int(options.port)
    )
