"""
This module handles the page routing and rendering.
It also handles the REST API for the text normalization model.
"""
import os
import tensorflow as tf
from flask import (Flask, render_template, jsonify, Response,
                   stream_with_context, Markup, request)
from normalizer import serve
from normalizer.utils import simplediff

APP = Flask(__name__)

APP.jinja_env.add_extension('jinja2.ext.loopcontrols')

@APP.errorhandler(404)
def url_error(error):
    """Returns the error message for wrong URL.
    Decorators:
        APP
    Arguments:
        error: error message
    Returns:
        string: HTML formatted error message.
    """

    return """
    Wrong URL!
    <pre>{}</pre>""".format(error), 404


@APP.errorhandler(500)
def server_error(error):
    """Returns the error message for Internal server error.
    Decorators:
        APP
    Arguments:
        error: error message
    Returns:
        string: HTML formatted error message.
    """

    return """
    An internal error occured: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(error), 500


@APP.route('/normalize/api', methods=['POST'])
def normalize():
    """The route for REST API which returns the normalized input.
    Decorators:
        APP
    Returns:
        json: contains the input and the normalized output
    """
    src = request.form['src']
    output = NORMALIZER.model_api(src)
    return jsonify({'src': src, 'tgt': output})


@APP.context_processor
def find_errors():
    """Returns highlighted text if doesn't match"""
    def _compare(enc, dec, res, tagged_words):
        diff, test_statistics, tagged_enc, tags_count = simplediff.check_errors(
            enc=enc, dec=dec, res=res, tagged_words=tagged_words)
        return {'highlighted': Markup(diff),
                'test_stats': test_statistics,
                'tags_count': tags_count,
                'tagged_enc': Markup(tagged_enc)}
    return dict(highlight_incorrect=_compare)


@APP.context_processor
def safe_division():
    """Jinja filter to do division with ZeroDivisionError handling."""
    def _divide(x, y):
        try:
            return x / y
        except ZeroDivisionError:
            return 0
    return dict(safe_division=_divide)


@APP.context_processor
def detokenize():
    def _detokenize(tokens):
        return NORMALIZER.detokenizer.detokenize(tokens,
                                                 return_str=True)
    return dict(detokenize=_detokenize)


@APP.route('/')
def index():
    """Returns the index.html"""
    return render_template('index.html')


@APP.route('/normalize/test', methods=['POST'])
def accuracy_test():
    """Normalizing the test files on the fly.
    The trick is to have an inner function that uses a
    generator to generate data and to invoke that function
    and pass it to a Response object.
    """
    enc = request.form['enc-data']
    dec = request.form['dec-data']

    def stream_template(template_name, **context):
        """Returns stream object from stream()
        instead of string from render()

        Args:
            template_name (str): filename of the template
            **context (tuple): keyword arguments

        Returns:
            StreamObject: stream object of the template
        """
        APP.update_template_context(context)
        template = APP.jinja_env.get_template(template_name)
        stream = template.stream(context)
        stream.enable_buffering(5)
        return stream

    def generate():
        """Returns a generator for Lazy-loading of table rows
        Yields:
            dict: The input, expected, and system output
        """
        enc_content = enc.splitlines()
        dec_content = dec.splitlines()

        for i, e in enumerate(enc_content[:500]):
            if e:
                result = {'enc': e.strip().strip('\n').lower(),
                          'dec': dec_content[i].strip().strip('\n').lower(),
                          'res': NORMALIZER.model_api(e.strip().strip('\n').lower())}
                yield result
    return Response(stream_with_context(
        stream_template('accuracy_testing.html', rows=generate(),
                                                 tagged_words=tagged_words)))


def readlines(filename):
    with open(filename, 'r') as infile:
        rows = infile.read().splitlines()
    return rows


def set_tag(tag, words):
    return {word.lower(): tag for word in words}


if __name__ == '__main__':
    ARGS = serve.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    tagged_words = {}

    if ARGS.twitter: 
        testing_path = os.path.join('normalizer', 'testing')
    else:
        testing_path = os.path.join('normalizer', 'training', 'data', 'dataset', 'd_1pass')
    tagged_words.update(set_tag('phonetic_styles', readlines(os.path.join(
        testing_path, 'phonetic_style.dic'))))

    tagged_words.update(set_tag('accent_styles', readlines(os.path.join(
        testing_path, 'accent_style.dic'))))

    tagged_words.update(set_tag('contractions', readlines(os.path.join(
        testing_path, 'contractions.dic'))))

    tagged_words.update(set_tag('misspellings', readlines(os.path.join(
        testing_path, 'misspelling.dic'))))
    
    tagged_words.update(set_tag('repeating_characters', readlines(os.path.join(
        testing_path, 'repeating_characters.dic'))))

    tagged_words.update(set_tag('repeating_units', readlines(os.path.join(
        testing_path, 'repeating_units.dic'))))

    with tf.Session() as sess:
        NORMALIZER = serve.Serve(sess=sess,
                                 model_name=ARGS.model_name,
                                 dataset_name=ARGS.dataset_name,
                                 checkpoint=ARGS.checkpoint,
                                 char_emb=ARGS.char_emb)
        APP.run(threaded=True, debug=True, use_reloader=True)
