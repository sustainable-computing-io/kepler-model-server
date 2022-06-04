FROM registry.access.redhat.com/ubi8/ubi
RUN  yum install -y python3.8 python3-devel gcc-c++  && yum clean all
COPY . .
RUN python3.8 -m pip install -r requirements.txt
EXPOSE 8100
CMD [ "python3", "server.py" ]
