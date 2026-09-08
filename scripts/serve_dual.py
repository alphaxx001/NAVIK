import http.server
import ssl
import threading
import os
import sys

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web'))
os.chdir(base_dir)

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress excessive request logging

def run_http():
    try:
        httpd = http.server.ThreadingHTTPServer(('0.0.0.0', 8088), QuietHandler)
        print("Serving HTTP on 0.0.0.0:8088...")
        httpd.serve_forever()
    except Exception as e:
        print("HTTP server error:", e)

def run_https():
    try:
        httpd = http.server.ThreadingHTTPServer(('0.0.0.0', 8443), QuietHandler)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        print("Serving HTTPS on 0.0.0.0:8443...")
        httpd.serve_forever()
    except Exception as e:
        print("HTTPS server error:", e)

if __name__ == "__main__":
    t_http = threading.Thread(target=run_http, daemon=True)
    t_http.start()
    run_https()
