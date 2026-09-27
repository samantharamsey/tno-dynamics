'''compatibility entry point for the render deployment'''

from apps.neptune_zvs.app import app, server


if __name__ == '__main__':
    app.run(debug=True)
