# Study Hours Estimator
Task: **Classification**

The purpose of this project is to predict the success probabilty of a startup project using determining factors such as target amount, backer count, project member count, and etc. This is to help startup teams to reassess their strategies especially in crowdfunding as well as take care of the important variables that may define the project's success.

## About the Dataset:

Sourced from: https://archive.ics.uci.edu/dataset/1025/turkish+crowdfunding+startups
Provenance: This dataset contains data on crowdfunding campaigns in Turkey. The dataset includes various characteristics such as crowdfunding projects, project descriptions, targeted and raised funds, campaign durations, and number of backers. Collected in 2022, this dataset provides a valuable resource for researchers who want to understand and analyze the crowdfunding ecosystem in Turkey. In total, there are data from more than 1500 projects on 6 different platforms. The dataset is particularly useful for training natural language processing (NLP) and machine learning models. This dataset is an important reference point for studies on the characteristics of successful crowdfunding campaigns and provides comprehensive information for entrepreneurs, investors and researchers in Turkey.

## About the project

**Package Manager:** uv: Python packaging in Rust\
**Virtual Environment:** uv venv\
**Web Deployment:** FastAPI\
**Container:** Docker\
**Cloud Service:** N/A

## How to use the project

### Clone the repository on your device
First, clone this repository by executing this prompt on the CLI:
```
git clone https://github.com/kabsmeiou/startup-success-probability.git
```
Then go to the terminal and make sure to navigate to where the *startup-success-probability* folder is at
```
cd startup-success-probability
```

### Creating the environment
Create a virtual environment with:
```
python -m venv .venv
```

Upon creating the environment, activate it with

**Windows:**
```
.venv/Scripts/activate
```
if the command above doesn't work in Windows, try a different approach:
```
cd .venv/Scripts
activate
```

or\
**Unix-based systems**
```
source .venv/bin/Activate
```

Install the dependencies by
```
pip install -r requirements
```

### Alternative: Using UV

You may install **uv package manager** on your system and simply run
```
uv sync --frozen
```
read the docs:https://docs.astral.sh/uv/getting-started/installation/ for details about installation.

### Testing the service

Now that the dependencies are set, you can start running the scripts. First, go to the *src* directory
```
cd src
```
At this point, you can choose to train the model yourself by running *train.py*
```
python3 train.py
```

However, you can simply run *test.py* as the model is already available in */src* as *startup-success-predictor.bin*.
Do this by first running the app using
```
python3 main.py
```
Then running *test.py* that sends a **POST** request to the service deployed
```
python3 test.py
```
Feel free to modify the dictionary in the *test.py* file to see how the probability of success changes.

### Running the app with Docker
For this to work, make sure you have Docker installed on your system and build the project with the following command
```
docker build -t startup-predict .
```

Then, on your terminal, run this command:
```
docker run -it --rm -p 8000:8000 startup-predict
```
In case of the error ***docker: permission denied while trying to connect to the Docker daemon socket at unix:///var/run/docker.sock: Head "http://%2Fvar%2Frun%2Fdocker.sock/_ping": dial unix /var/run/docker.sock: connect: permission denied.***, add *sudo* to the commands above with *docker build* and *docker run*.

You should see something like this:\
![alt text](https://i.imgur.com/hCM8H2u.png)

Finally, you may access 
```
http://0.0.0.0:8000
```
and start filling out the forms to get predictions!

