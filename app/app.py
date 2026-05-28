import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Estimador de Precio de Propiedades", page_icon="🏠", layout="wide")
st.title("🏠 Estimador de Precio de Propiedades — Thane, India")
st.markdown("Ingresa las características de la propiedad y obtén un precio estimado usando el modelo entrenado.")

@st.cache_resource
def load_models():
    model = joblib.load('model_rf.pkl')
    feature_names = joblib.load('feature_names.pkl')
    scaler = joblib.load('scaler_kmeans.pkl')
    kmeans = joblib.load('model_kmeans.pkl')
    return model, feature_names, scaler, kmeans

try:
    model, feature_names, scaler, kmeans = load_models()
except FileNotFoundError:
    st.error('No se encontraron los modelos. Ejecuta los notebooks 03 y 04 para generar los archivos en app/')
    st.stop()

st.sidebar.header('📋 Características de la Propiedad')
bhk = st.sidebar.selectbox('Habitaciones (BHK)', [1, 2, 3, 4, 5])
area = st.sidebar.slider('Área en sqft', 100, 2000, 750)
floor = st.sidebar.slider('Piso', 0, 30, 5)
total_floors = st.sidebar.slider('Total de pisos del edificio', 1, 50, 10)
bathrooms = st.sidebar.selectbox('Número de baños', [1, 2, 3, 4, 5])
balconies = st.sidebar.selectbox('Número de balcones', [0, 1, 2, 3])
furnishing = st.sidebar.selectbox('Amueblado', ['Unfurnished', 'Semi-Furnished', 'Furnished'])
facing = st.sidebar.selectbox('Orientación', ['East', 'West', 'North', 'South'])
has_parking = st.sidebar.checkbox('¿Tiene estacionamiento?', value=True)
is_luxury = st.sidebar.checkbox('¿Propiedad de lujo?', value=False)
has_garden = st.sidebar.checkbox('¿Vista a jardín/parque?', value=False)
city_avg_price = st.sidebar.number_input('Precio mediano de la ciudad (Rupias)', min_value=100000, max_value=100000000, value=5000000, step=100000)

furnishing_map = {'Unfurnished': 0, 'Semi-Furnished': 1, 'Furnished': 2}

if st.sidebar.button('🔍 Estimar Precio'):
    floor_ratio = floor / total_floors if total_floors > 0 else 0
    input_data = {
        'bhk_count': bhk,
        'carpet_area_num': area,
        'floor_number': floor,
        'total_floors': total_floors,
        'floor_ratio': floor_ratio,
        'Bathroom': bathrooms,
        'Balcony': balconies,
        'furnishing_score': furnishing_map[furnishing],
        'has_parking': int(has_parking),
        'is_luxury': int(is_luxury),
        'has_garden_view': int(has_garden),
        'has_society': 1,
        'city_avg_price': city_avg_price,
    }
    row = pd.DataFrame([input_data])
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_names]

    precio = model.predict(row)[0]
    mae_placeholder = 1_500_000

    col1, col2, col3 = st.columns(3)
    col1.metric('💰 Precio Estimado', f'₹ {precio:,.0f}')
    col2.metric('📉 Rango mínimo', f'₹ {max(0, precio - mae_placeholder):,.0f}')
    col3.metric('📈 Rango máximo', f'₹ {precio + mae_placeholder:,.0f}')

    if precio >= 10_000_000:
        st.success(f'💎 Equivalente a {precio/10_000_000:.2f} Cr')
    else:
        st.success(f'💵 Equivalente a {precio/100_000:.1f} Lac')

    cluster_features = np.array([[
        bhk,
        area,
        precio / area if area > 0 else 0,
        furnishing_map[furnishing],
        int(has_parking),
        int(is_luxury),
        bathrooms,
        floor_ratio,
        city_avg_price,
    ]])
    scaled = scaler.transform(cluster_features)
    cluster = kmeans.predict(scaled)[0]
    perfiles = {
        0: '💼 Inversión Económica',
        1: '👨‍👩‍👧 Perfil Familiar',
        2: '🏢 Perfil Ejecutivo',
        3: '✨ Lujo Premium',
    }
    st.info(f'**Perfil estimado:** {perfiles.get(cluster, "No identificado")}')
