from typing import Union, Optional

from enum import Enum
from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware

import numpy as np
import pickle

app = FastAPI()

# Load model with pickle
model_file=f'startup-success-predictor.bin'

with open(model_file, 'rb') as f_in:
    model, dv = pickle.load(f_in)

# Tailwindcss
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(GZipMiddleware)

# Load templates from the "templates" directory
templates = Jinja2Templates(directory="templates")

class project_category(str, Enum):
    FilmVideoPhotography = 'film-video-photography'
    Technology = 'technology'
    CultureArt = 'culture-art'
    Education = 'education'
    Environment = 'environment'
    Music = 'music'
    HealthBeauty = 'health-beauty'
    Design = 'design'
    Publishing = 'publishing'
    FoodEatingDrinking = 'food-eating-drinking'
    Sports = 'sports'
    Animals = 'animals'
    Fashio = 'fashion'
    SocialResponsibility = 'social_responsibility'
    DancePerformance = 'dance-performance'
    Tourism = 'tourism'
    Other = 'other'


class binary_option(str, Enum):
    Yes = 'yes'
    No = 'no'


class funding_type(str, Enum):
    Prize = 'prize'
    Donation = 'donation'


class funding_method(str, Enum):
    all_or_nothing = 'all-or-nothing'
    keep_it_all = 'keep-it-all'


class Project(BaseModel):
    name: Optional[str]
    mass_funding_type: funding_type
    project_category: project_category
    funding_method: funding_method
    project_supported: int
    number_of_projects_owned: int
    number_of_teams: int
    project_duration: int
    promo_video: binary_option
    promo_video_length: int
    image_count: int
    faq: binary_option
    updates: int
    comments: int
    reward_count: int
    project_member_count: int
    website: binary_option
    social_media: binary_option
    social_media_count: int
    social_media_followers: int
    total_tags: int
    target_amount: int
    backer_count: int


def predict_success(project: Project):
    '''
    This function preprocesses the object passed and predicts the success rate of a startup using
    the imported XGB Classifier model.
    ''' 
    # Convert to dictionary
    data = dict(project)
    # Transform data
    data['log_backer_count'] = np.log1p(data['backer_count'])
    del data['backer_count']
    del data['name']
    data['project_category'] = data['project_category'].value
    data['website'] = data['website'].value
    data['mass_funding_type'] = data['mass_funding_type'].value
    data['funding_method'] = data['funding_method'].value
    data['promo_video'] = data['promo_video'].value
    data['faq'] = data['faq'].value
    data['social_media'] = data['social_media'].value

    mapping = {'yes': 1, 'no': 0}
    data['faq'] = mapping[data['faq']]
    data['social_media'] = mapping[data['social_media']]
    data['promo_video'] = mapping[data['promo_video']]
    data['website'] = mapping[data['website']]
    X = dv.transform(data)
    print(data)

    # Predict and return probability
    probability_of_success = model.predict_proba(X)[:,1].round(4)[0] * 100
    return probability_of_success

@app.post("/test/")
def predict_test(project: dict):
    data = project

    # Transform data
    data['log_backer_count'] = np.log1p(data['backer_count'])
    del data['backer_count']
    data['project_category'] = data['project_category']
    data['website'] = data['website']
    data['mass_funding_type'] = data['mass_funding_type']
    data['funding_method'] = data['funding_method']
    data['promo_video'] = data['promo_video']
    data['faq'] = data['faq']
    data['social_media'] = data['social_media']

    X = dv.transform(data)
    print(data)

    # Predict and return probability
    probability_of_success = model.predict_proba(X)[:,1].round(4)[0] * 100
    # convert directly to float from numpy.float as numpy.float cannot be serialized by fastapi json encoder
    return {"probability_of_success": float(probability_of_success)} 

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request, message: str = None, form_data: dict = None):
    return templates.TemplateResponse("index.html", {"request": request, "message": message, "form_data": form_data})


@app.post("/predict/", response_class=HTMLResponse)
async def create_project(
    request: Request,
    project_data: Project = Form(...),
):
    # Validate and process the data using the Project model
    project = dict(project_data)
    prediction = predict_success(project)
    message = f"Success Rate of {project['name']}: {prediction:.2f}%"
    print(message)
    redirect_url = f"/?message={message}&form_data={project}"
    return RedirectResponse(redirect_url, status_code=303)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)