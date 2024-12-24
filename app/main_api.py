from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from modul.inference import pred_optimal_route
from logs import logger



app = FastAPI(title="Route Optimization API",
    description="author: Fiorentika Devasha",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReqPredict(BaseModel):
    id_kurir: str


@app.post("/api/v1/predict-route")
async def create_item(request: ReqPredict):
    try:
        logger.info(request)
        data = pred_optimal_route(request.id_kurir)
        logger.info(data)
        return {"resp_msg": "predict route berhasil",
                     "resp_data":  data
                     }
    except Exception as e:
        logger.error(str(e))
        return JSONResponse(
            status_code= status.HTTP_401_UNAUTHORIZED,
            content={"resp_msg": str(e),
                     "resp_data":  None
                     },
        )