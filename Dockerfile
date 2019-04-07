FROM sordonia120446/vision:latest

ADD . /app
WORKDIR /app
RUN pip install matplotlib
