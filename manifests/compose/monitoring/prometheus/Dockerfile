FROM quay.io/prometheus/prometheus:main

COPY /prometheus.yml /etc/prometheus/prometheus.yml

CMD [\
	"--config.file=/etc/prometheus/prometheus.yml",\
	"--storage.tsdb.path=/prometheus", \
	"--web.enable-admin-api" \
]
