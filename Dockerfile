FROM python:3.8-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libc-dev \
    libssl-dev \
    libjpeg-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    openslide-tools \
    && rm -rf /var/lib/apt/lists/*

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m piptools sync requirements.txt

COPY --chown=user:user model1.pth /opt/app

COPY --chown=user:user model2.pth /opt/app

COPY --chown=user:user model3.pth /opt/app

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
