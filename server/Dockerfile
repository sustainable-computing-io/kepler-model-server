FROM registry.access.redhat.com/ubi8/ubi
RUN yum install -y python3 && yum clean all
COPY . .
RUN python3 -m pip install -r requirements.txt
EXPOSE 9100
CMD [ "python3", "server.py" ]
