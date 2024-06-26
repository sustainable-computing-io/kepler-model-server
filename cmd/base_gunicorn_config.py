import os


workers = int(os.environ.get('GUNICORN_PROCESSES', '2'))

threads = int(os.environ.get('GUNICORN_THREADS', '4'))

port = os.environ.get('GUNICORN_PORT', '8100')

bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:' + port)


forwarded_allow_ips = '*'

secure_scheme_headers = { 'X-Forwarded-Proto': 'https' }