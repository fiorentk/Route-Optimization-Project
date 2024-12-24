import pandas as pd
from modul.modul_tsp_geopy import predict_optimal_route


print('Start Script!')
df = pd.read_csv('sampel2.csv', dtype=str)

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

# Hitung jumlah total per 'id_kurir' dan 'date'
grouped_df_cipedes = (
    df_cipedes.groupby(['id_kurir', 'date'], as_index=False)
    .agg(count=('date', 'size'))
)

kurir = ['560001282']
df_coba_cipedes = df_cipedes[df_cipedes['id_kurir'].isin(kurir)]
df_coba_cipedess = df_coba_cipedes.drop(columns=['id_kurir','date'])
df_coba_cipedes1 = df_coba_cipedes[(df_coba_cipedes['id_kurir']=='560001282') & (df_coba_cipedes['date']=='2024-10-11')]
df_coba_cipedes1 = df_coba_cipedes1.drop(columns=['id_kurir','date'])
result_geopy = predict_optimal_route(df_coba_cipedes1, df_kantor)

result_geopy

print(f"result: {result_geopy.to_dict(orient='records')}")
