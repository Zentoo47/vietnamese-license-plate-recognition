"""Application entry point for ITS License Plate Detection."""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.api.routes import router as api_router
from app.database import init_db

app = FastAPI(
    title="ITS License Plate Detection",
    description="Vietnamese license plate detection and OCR web application",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(api_router)


@app.on_event("startup")
async def startup_event():
    init_db()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
