from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse
import os
from sauti_rag import SautiEngine

engine = SautiEngine()

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
    <title>SautiNLP Web UI</title>
    <style>
        body { font-family: Arial, sans-serif; background: #faf9f6; color: #2b2b2b; max-width: 700px; margin: 40px auto; padding: 20px; }
        h1 { color: #1A2B4C; }
        .box { background: #fff; padding: 20px; border-radius: 6px; border: 1px solid #E2E8F0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        input[type="text"] { width: 75%; padding: 10px; font-size: 14px; border: 1px solid #cbd5e1; border-radius: 4px; }
        button { padding: 10px 20px; background: #3B71CA; color: white; border: none; border-radius: 4px; font-size: 14px; cursor: pointer; }
        button:hover { background: #1A2B4C; }
        pre { background: #1E293B; color: #E2E8F0; padding: 15px; border-radius: 4px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body>
    <div class="box">
        <h1>SautiNLP Web UI</h1>
        <p>Test Swahili &amp; Sheng regional slang translations and RAG retrieval live.</p>
        <form method="GET" action="/">
            <input type="text" name="query" value="{query}" placeholder="Try: noma, luku, mbogi, form...">
            <button type="submit">Translate / Query</button>
        </form>
        <h3>Result:</h3>
        <pre>{result}</pre>
    </div>
</body>
</html>
"""

class SautiWebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        query_params = urllib.parse.parse_qs(parsed_path.query)
        query_str = query_params.get("query", [""])[0]

        result = "Enter a query above or use phrases like 'noma', 'luku', 'mbogi', 'form'."
        if query_str:
            try:
                context = engine.retrieve_context(query_str)
                res = engine.generate_rag_response(query_str, context)
                result = f"Query: {query_str}\n\nRetrieved Context:\n{context}\n\nResponse:\n{res}"
            except Exception as e:
                result = f"Query: {query_str}\nError: {e}"

        page = HTML_PAGE.replace("{query}", query_str).replace("{result}", result)
        
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(page.encode("utf-8"))

def run_server(port=8080):
    server_address = ('', port)
    httpd = HTTPServer(server_address, SautiWebHandler)
    print(f"Starting SautiNLP Web UI server at http://127.0.0.1:{port} ...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
