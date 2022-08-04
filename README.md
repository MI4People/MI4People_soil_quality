# MI4People_soil_quality

## Overview

Soil Quality Evaluation System developed by MI4People gGmbH
T

This project is based on `Kedro 0.18.2`. Take a look at the [Kedro documentation](https://kedro.readthedocs.io) for further details.


## How to setup Kedro for the project

Create a new Python virtual environment, called kedro-environment, using conda:

`conda create --name mi4people_soil_quality python=3.7 -y`

This will create an isolated Python 3.7 environment. To activate it:

`conda activate mi4people_soil_quality`

Install Kedro using

`pip install kedro`

Install the dependencies declared in src/requirements.txt:

`pip install -r src/requirements.txt`

## How to run the pipeline
You can run all pipelines using `kedro run` or only specific pipelines using `kedro run <pipeline>`

## How to use Jupyter Notebooks
After installing Jupyter, you can start a local notebook server:
`kedro jupyter notebook`