"Plot a PNG using matplotlib in a web request, using Flask."

# Install dependencies, preferably in a virtualenv:
#
#     pip install flask matplotlib
#
# Run the development server:
#
#     python app.py
#
# Go to http://localhost:5000/plot.png and see a plot of random data.

import random
from io import BytesIO

from flask import Flask, make_response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


app = Flask(__name__)


@app.route('/plot')
def plot():
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)

    xs = range(100)
    ys = [random.randint(1, 50) for x in xs]

    axis.plot(xs, ys)
    canvas = FigureCanvas(fig)
    output = BytesIO()
    canvas.print_png(output)
    # response = {}
    # response["result"] = {"precision": 0.8, "recall": 0.7, "f1": 0.75}
    response = make_response(output.getvalue())
    response.mimetype = 'image/png'
    return response


if __name__ == '__main__':
    app.run(debug=True)