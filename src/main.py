from typing import Union

from enum import Enum
from fastapi import FastAPI, Form, Request
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware

import os

import arel

app = FastAPI()

# tailwindcss
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
    name: str
    category: project_category
    has_website: binary_option
    funding_type: funding_type
    funding_method: funding_method
    project_supported: int
    project_owned: int
    number_of_teams: int
    project_duration: int
    has_promo_video: binary_option
    promo_video_length: int
    image_count: int
    has_faq: binary_option
    updates: int
    comments: int
    reward_count: int
    project_member_count: int
    has_social_media: binary_option
    social_media_count: int
    social_media_followers: int
    total_tags: int
    target_amount: int
    backer_count: int



def predict_success(project: Project):
    # Implement your prediction logic here
    pass


@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/", response_class=HTMLResponse)
async def create_project(
    request: Request,
    name: str = Form(...),
    category: project_category = Form(...),
    has_website: binary_option = Form(...),
    funding_type: funding_type = Form(...),
    funding_method: funding_method = Form(...),
    project_supported: int = Form(...),
    project_owned: int = Form(...),
    number_of_teams: int = Form(...),
    project_duration: int = Form(...),
    has_promo_video: binary_option = Form(...),
    promo_video_length: int = Form(...),
    image_count: int = Form(...),
    has_faq: binary_option = Form(...),
    updates: int = Form(...),
    comments: int = Form(...),
    reward_count: int = Form(...),
    project_member_count: int = Form(...),
    has_social_media: binary_option = Form(...),
    social_media_count: int = Form(...),
    social_media_followers: int = Form(...),
    total_tags: int = Form(...),
    target_amount: int = Form(...),
    backer_count: int = Form(...),
):
    # Validate and process the data using the Project model
    project = Project(name=name, category=category, has_website=has_website)
    prediction = predict_success(project)
    message = f"Prediction: {69.90}"
    return templates.TemplateResponse("index.html", {"request": request, "message": message})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)