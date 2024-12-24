import pandas as pd
import time

from modul.modul_tsp_geopy import predict_optimal_route
from logs import logger


df = pd.read_csv('sampel2.csv', dtype=str)

def pred_optimal_route(id_kurir: str):
    data = {
        "nama_kantor": [],
        "latitude": [],
        "longitude": []
    }
    df_kantor = pd.DataFrame(data)
    new_data = {
        "nama_kantor": "DC CIPEDES",
        "latitude": -6.894104,
        "longitude": 107.590074
    }

    df_kantor = pd.concat([df_kantor, pd.DataFrame([new_data])], ignore_index=True)
    df['id_kurir'] = df['pod__photo'].str.extract(r'camera\.photoDeliveryProcessImage\.(\d+)')
    df['pod__timereceive'] = pd.to_datetime(df['pod__timereceive'])
    df['date'] = df['pod__timereceive'].dt.date
    df['date'] = df['date'].astype(str)
    df_cipedes = df[df["connote__connote_receiver_address_detail"].str.contains("cipedes", case=False, na=False)]

    kurir = [id_kurir]
    df_coba_cipedes = df_cipedes[df_cipedes['id_kurir'].isin(kurir)]
    df_coba_cipedes1 = df_coba_cipedes[(df_coba_cipedes['id_kurir']==id_kurir) & (df_coba_cipedes['date']=='2024-10-11')]
    df_coba_cipedes1 = df_coba_cipedes1.drop(columns=['id_kurir','date'])

    # predict
    logger.info('start predicting optimal route')
    start_time = time.time()  
    result_geopy = predict_optimal_route(df_coba_cipedes1, df_kantor)
    end_time = time.time()  
    elapsed_time = end_time - start_time  
    logger.info(f'end predicting optimal route. Elapsed time: {elapsed_time:.2f} seconds')

    result_geopy = result_geopy.to_dict(orient='records') # convert df into dict format (json)
    logger.info(result_geopy)
    return result_geopy
