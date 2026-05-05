"""Serve the frontend on port 3000."""
import http.server
import os

os.chdir(os.path.join(os.path.dirname(__file__), "frontend"))
server = http.server.HTTPServer(("0.0.0.0", 3000), http.server.SimpleHTTPRequestHandler)
print("Frontend running at http://localhost:3000/demo.html")
server.serve_forever()
