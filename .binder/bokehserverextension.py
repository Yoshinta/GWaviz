from subprocess import Popen


def load_jupyter_server_extension(nbapp):
    """serve the main-app directory with bokeh server"""
    Popen(["bokeh", "serve", "main-app", "--allow-websocket-origin=*"])
