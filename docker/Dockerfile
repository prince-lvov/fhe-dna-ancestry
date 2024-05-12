FROM python:3.8-slim
ENV PYTHONUNBUFFERED 1
ENV TZ "Europe/Moscow"
RUN apt update
RUN apt full-upgrade -y
RUN apt install -y build-essential
RUN mkdir /var/run/sshd

RUN SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install matplotlib==3.3.4 numpy==1.20.3 pandas==1.3.5 PyYAML==5.1.2 scikit-allel==1.3.1 scikit-learn==1.0.1 scipy==1.5.3 seaborn==0.11.2 sklearn-crfsuite==0.3.6 tqdm==4.62.3 uncertainty-calibration==0.0.7 xgboost==1.1.1

RUN groupadd --gid 1000 user \
  && useradd --uid 1000 --gid user --shell /bin/bash --create-home user

RUN apt install -y openssh-server wget curl rsync git htop mc bcftools

RUN pip install jupyter gdown concrete-ml

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*

RUN echo "user:12345678" | chpasswd

WORKDIR /home/user

CMD ["/usr/sbin/sshd", "-D"]
