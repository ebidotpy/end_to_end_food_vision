import os
from fastapi import FastAPI, Request, responses, Response
from fastapi.middleware.cors import CORSMiddleware
from cnnClassifier.pipeline.predict import PredictionPipeline
from cnnClassifier.utils.common import decodeImage


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"])

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

@app.get("/")
async def home():
    file_path = "templates/index.html"
    with open(file_path, "r") as file:
        html_content = file.read()

    return Response(content=html_content, media_type="text/html")

@app.route("/train", methods=['GET','POST'])
async def trainRoute(request):
    os.system("python main.py")
    return Response(content="Training done successfully!", media_type="text/plain")

@app.post("/predict")
async def predictRoute(request: Request):
    image = await request.json()
    decodeImage(image['image'], clApp.filename)
    result = clApp.classifier.predict()
    return responses.JSONResponse(content=result, media_type="application/json")

clApp = ClientApp()