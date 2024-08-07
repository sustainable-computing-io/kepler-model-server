# import  external src
import  http.server
import  socketserver
import  atexit
import  threading

import  os
import  sys

#################################################################
# import  internal src 
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.append(src_path)
#################################################################

from util.config import  model_toppath

os.chdir(model_toppath)

def cleanup_task(server):
    print("Shutdown server...")
    server.shutdown()

def get_server(file_server_port):
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", file_server_port), Handler)

    # Register the cleanup task to be executed on program exit
    atexit.register(cleanup_task, httpd)

    print("Http File Serve Serving at Port", file_server_port, " for ", model_toppath)
    return httpd

def http_file_server(file_server_port):
    try: 
        httpd = get_server(file_server_port)
        # Start the server in a separate thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
    except Exception as err:
        print("File server is running: {}".format(err))

if __name__ == "__main__":
    httpd = get_server(8110)
    httpd.serve_forever()