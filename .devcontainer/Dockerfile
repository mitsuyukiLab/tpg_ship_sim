FROM python:3.12

USER root
WORKDIR /tpg-ship-sim

RUN apt-get update && apt-get install -y \
    sudo \
    git \
    curl \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN python -m pip install numpy==1.26.0 \
    distance \
    geopy \
    folium \
    cartopy \
    matplotlib \
    tz \
    datetime timedelta  \
    polars==0.19.2 \
    selenium \
    SciencePlots \
    opencv-python \
    jupyterlab \
    plotly \
    scikit-learn \
    kaleido


RUN pip install tqdm

# Linter and Formatter
RUN pip install flake8 black isort

# hydra-core
RUN pip install hydra-core --upgrade \
    hydra-core optuna