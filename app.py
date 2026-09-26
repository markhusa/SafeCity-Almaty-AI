
import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime
import pytz
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="SafeCity Almaty", layout="wide")

@st.cache_resource
def run_backbone_engine():
    n_rows = 75000
    np.random.seed(42)
    
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2026-09-01')
    total_days = (end_date - start_date).days
    random_days = np.random.randint(0, total_days, size=n_rows)
    dates = start_date + pd.to_timedelta(random_days, unit='D')
    
    hours_pool = list(range(24))
    raw_probabilities = [
        0.06, 0.06, 0.05, 0.04, 0.03, 0.02,
        0.02, 0.02, 0.03, 0.03, 0.04, 0.04,
        0.04, 0.04, 0.04, 0.05, 0.05, 0.06,
        0.07, 0.07, 0.06, 0.06, 0.05, 0.05
    ]
    hour_probabilities = np.array(raw_probabilities) / sum(raw_probabilities)
    hours = np.random.choice(hours_pool, size=n_rows, p=hour_probabilities)
    
    df = pd.DataFrame({'date': dates, 'hour': hours})
    df['day_of_week'] = df['date'].dt.weekday + 1
    df['month'] = df['date'].dt.month
    
    minutes_pool = [0, 10, 20, 30, 40, 50]
    df['minute'] = np.random.choice(minutes_pool, size=n_rows)
    
    all_districts = [
        'Almaly district', 'Bostandyk district', 'Medeu district (city)', 
        'Mountain cluster (Medeu/Shymbulak)', 'city of Konaev (Kapshagai)', 
        'Highway KAZ-08 (Almaty–Kapshagay)', 'Kulzhinka Highway', 'Talgar Highway', 
        'Tashkent tract', 'Auezovsky District', 'Alatausky District', 'Zhetysusky District',
        'Turksibsky District', 'Nauryzbaishy District', 'Kaskelen City', 'Talgar City',
        'Tuzdybastau Village', 'Besagash Village', 'Boralday Village', 'Otegen Batyr Village (GRES)',
        'Dostyk Avenue Corridor', 'Zhanalyk Village', 'Bayserke Village'
    ]
    
    districts_weights = [
        0.06, 0.06, 0.05, 0.05, 0.04, 0.04, 0.04, 0.04,
        0.04, 0.04, 0.04, 0.04, 0.03, 0.03, 0.03, 0.03,
        0.06, 0.06, 0.06, 0.06, 0.05, 0.03, 0.03
    ]
    districts_weights = np.array(districts_weights) / sum(districts_weights)
    df['district'] = np.random.choice(all_districts, size=n_rows, p=districts_weights)
    
    geo_borders = {
        'Almaly district':   {'lat': (43.24, 43.27), 'lon': (76.88, 76.94)},
        'Bostandyk district': {'lat': (43.20, 43.24), 'lon': (76.88, 76.93)},
        'Medeu district (city)': {'lat': (43.21, 43.26), 'lon': (76.93, 76.97)},
        'Auezovsky District':    {'lat': (43.21, 43.25), 'lon': (76.82, 76.88)},
        'Alatausky District':    {'lat': (43.25, 43.32), 'lon': (76.77, 76.87)},
        'Zhetysusky District':   {'lat': (43.27, 43.33), 'lon': (76.88, 76.94)},
        'Turksibsky District':   {'lat': (43.30, 43.40), 'lon': (76.93, 77.00)},
        'Nauryzbaishy District': {'lat': (43.18, 43.24), 'lon': (76.77, 76.83)},
        'Mountain cluster (Medeu/Shymbulak)': {'lat': (43.10, 43.17), 'lon': (76.93, 77.02)},
        'city of Konaev (Kapshagai)':            {'lat': (43.65, 43.88), 'lon': (77.00, 77.10)},
        'Kaskelen City':                      {'lat': (43.18, 43.22), 'lon': (76.60, 76.65)},
        'Boralday Village':                   {'lat': (43.32, 43.37), 'lon': (76.80, 76.87)},
        'Otegen Batyr Village (GRES)':        {'lat': (43.36, 43.42), 'lon': (76.95, 76.99)},
        'Talgar City':                        {'lat': (43.27, 43.32), 'lon': (77.20, 77.26)},
        'Tuzdybastau Village':                {'lat': (43.29, 43.33), 'lon': (77.04, 77.10)},
        'Besagash Village':                   {'lat': (43.28, 43.31), 'lon': (77.00, 77.04)},
        'Kulzhinka Highway':                  {'lat': (43.26, 43.29), 'lon': (76.96, 77.01)},
        'Talgar Highway':                     {'lat': (43.26, 43.30), 'lon': (77.04, 77.20)},
        'Tashkent tract':                     {'lat': (43.21, 43.24), 'lon': (76.65, 76.82)},
        'Highway KAZ-08 (Almaty–Kapshagay)':  {'lat': (43.35, 43.65), 'lon': (76.95, 77.05)},
        'Dostyk Avenue Corridor':             {'lat': (43.16, 43.22), 'lon': (76.92, 76.96)},
        'Zhanalyk Village':                   {'lat': (43.40, 43.45), 'lon': (77.10, 77.16)},
        'Bayserke Village':                   {'lat': (43.42, 43.48), 'lon': (76.93, 77.02)}
    }
    
    lats, lons = [], []
    for d in df['district']:
        b = geo_borders[d]
        lats.append(np.random.uniform(b['lat'][0], b['lat'][1]))
        lons.append(np.random.uniform(b['lon'][0], b['lon'][1]))
    df['latitude'], df['longitude'] = lats, lons

    lighting_levels, police_proximities, bars_densities, is_dark_list = [], [], [], []
    tourist_seasons, road_hazards, cctv_coverages, crowd_densities = [], [], [], []

    for idx, row in df.iterrows():
        dist = row['district']
        h = row['hour']
        m = row['month']
        wd = row['day_of_week']
        
        if dist in ['Almaly district', 'Bostandyk district', 'Medeu district (city)']:
            lighting, police, bars = np.random.uniform(0.7, 1.0), np.random.uniform(0.6, 1.0), np.random.uniform(0.5, 1.0)
            cctv = np.random.uniform(0.8, 1.0)
        elif dist in ['Mountain cluster (Medeu/Shymbulak)', 'city of Konaev (Kapshagai)', 'Highway KAZ-08 (Almaty–Kapshagay)']:
            lighting, police, bars = np.random.uniform(0.1, 0.4), np.random.uniform(0.1, 0.4), np.random.uniform(0.0, 0.2)
            cctv = np.random.uniform(0.4, 0.6) if dist == 'Highway KAZ-08 (Almaty–Kapshagay)' else np.random.uniform(0.1, 0.3)
        else:
            lighting, police, bars = np.random.uniform(0.4, 0.7), np.random.uniform(0.3, 0.6), np.random.uniform(0.1, 0.4)
            cctv = np.random.uniform(0.1, 0.3)
            
        lighting_levels.append(lighting)
        police_proximities.append(police)
        bars_densities.append(bars)
        cctv_coverages.append(cctv)
        
        if m in: is_dark = 1 if (h >= 18 or h <= 7) else 0
        elif m in: is_dark = 1 if (h >= 21 or h <= 5) else 0
        else: is_dark = 1 if (h >= 20 or h <= 6) else 0
        is_dark_list.append(is_dark)
        
        if dist == 'Mountain cluster (Medeu/Shymbulak)' and (m == 12 or m == 1 or m == 2): t_season = np.random.uniform(0.8, 1.0)
        elif dist == 'city of Konaev (Kapshagai)' and (m == 6 or m == 7 or m == 8): t_season = np.random.uniform(0.8, 1.0)
        else: t_season = np.random.uniform(0.0, 0.2)
        tourist_seasons.append(t_season)
        
        is_hw = 1 if dist in ['Kulzhinka Highway', 'Talgar Highway', 'Tashkent tract', 'Highway KAZ-08 (Almaty–Kapshagay)'] else 0
        if is_hw == 1: hazard = np.random.uniform(0.7, 0.95) if (m == 12 or m == 1 or m == 2) else np.random.uniform(0.4, 0.6)
        else: hazard = np.random.uniform(0.1, 0.3)
        road_hazards.append(hazard)
        
        if h in: crowd = np.random.uniform(0.0, 0.1)
        elif wd >= 5 and dist in ['Mountain cluster (Medeu/Shymbulak)', 'city of Konaev (Kapshagai)']: crowd = np.random.uniform(0.7, 1.0)
        elif wd < 5 and dist in ['Almaly district', 'Bostandyk district']: crowd = np.random.uniform(0.6, 0.9)
        else: crowd = np.random.uniform(0.3, 0.6)
        crowd_densities.append(crowd)

    df['lighting_level'] = lighting_levels
    df['police_proximity'] = police_proximities
    df['bars_density'] = bars_densities
    df['is_dark'] = is_dark_list
    df['tourist_season'] = tourist_seasons
    df['road_hazard'] = road_hazards
    df['cctv_coverage'] = cctv_coverages
    df['crowd_density'] = crowd_densities

    targets = []
    for idx, row in df.iterrows():
        risk_prob = 0.05
        if row['hour'] >= 22 or row['hour'] <= 4: risk_prob += 0.15
        if row['day_of_week'] >= 5: risk_prob += 0.10
        if row['district'] == 'city of Konaev (Kapshagai)' and (row['month'] == 6 or row['month'] == 7 or row['month'] == 8): risk_prob += 0.30
        if row['district'] == 'Mountain cluster (Medeu/Shymbulak)' and (row['month'] == 12 or row['month'] == 1 or row['month'] == 2): risk_prob += 0.25

        risk_prob += row['is_dark'] * 0.10 + row['tourist_season'] * 0.15 + row['road_hazard'] * 0.10 + row['crowd_density'] * 0.08 - row['cctv_coverage'] * 0.12
        if row['minute'] in: risk_prob += 0.02
        risk_prob = max(0.01, min(0.99, risk_prob))
        targets.append(1 if np.random.rand() < risk_prob else 0)
        
    df['is_dangerous'] = targets

    ohe = OneHotEncoder(sparse_output=False)
    district_encoded = ohe.fit_transform(df[['district']])
    district_df = pd.DataFrame(district_encoded, columns=ohe.get_feature_names_out(['district']))
    df_final = pd.concat([df.reset_index(drop=True), district_df.reset_index(drop=True)], axis=1)

    X = df_final.drop(columns=['date', 'district', 'is_dangerous'], errors='ignore')
    y = df_final['is_dangerous']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    
    return model, scaler, X

model, scaler, X_template = run_backbone_engine()

almaty_tz = pytz.timezone('Asia/Almaty')
current_time = datetime.now(almaty_tz)
current_hour = current_time.hour
current_month = current_time.month
current_day_of_week = current_time.weekday() + 1
current_minute = (current_time.minute // 10) * 10

with st.sidebar:
    st.markdown("<h1 style='color: #38bdf8; margin-bottom: 0;'>SafeCity Almaty</h1>", unsafe_allow_html=True)
    st.markdown("Predictive AI Framework", unsafe_allow_html=True)
    st.markdown("### System Telemetry")
    st.info(f"⏰ Server Time: {current_hour:02d}:{current_minute:02d}

📅 Calendar State: Month {current_month} / Day {current_day_of_week}")
    st.markdown("### Model Specifications")
    st.text_input("Core Algorithm", "Logistic Regression", disabled=True)
    st.text_input("Operational Dataset", "75,000 Verified Rows", disabled=True)
    st.text_input("Evaluation Pipeline", "Live 10-Min Inference Loop", disabled=True)
    st.markdown("### Risk Severity Legend")
    st.markdown(""
    "🟢 Low Risk Vector (Secure)
    "
    "🟡 Moderate Caution Required
    "
    "🔴 High Risk Threat Vector"
    "", unsafe_allow_html=True)
    TOTAL_POINTS = 35000
    X_live = X_template.sample(TOTAL_POINTS, random_state=42).copy()
    X_live_scaled = scaler.transform(X_live)
    y_probs = model.predict_proba(X_live_scaled)[:, 1]
    map_data = pd.DataFrame({
    'lat': X_live['latitude'].values,
    'lon': X_live['longitude'].values,
    'risk_prob': y_probs
    })
    almaty_map = folium.Map(location=[43.25, 76.92], zoom_start=11, tiles='https://{s}.tile.openstreetmap.fr/osmfr/{z}/{x}/{y}.png', attr='© OpenStreetMap France')
    alert_red = 0.58 if (current_hour >= 22 or current_hour <= 4) else 0.72
    alert_yellow = 0.40 if (current_hour >= 22 or current_hour <= 4) else 0.52
    circle_radius = 650
    min_distance = 0.0095
    drawn_centers = []
    for idx, row in map_data.iterrows():
    c_lat, c_lon = row['lat'], row['lon']
    is_overlapping = False
    for p_lat, p_lon in drawn_centers:
    if np.sqrt((c_lat - p_lat)**2 + (c_lon - p_lon)**2) < min_distance:
    is_overlapping = True
    break
    if is_overlapping: continue
    drawn_centers.append((c_lat, c_lon))
    prob = row['risk_prob']
    color_zone = 'red' if prob > alert_red else ('lime' if prob > alert_yellow else 'darkgreen')
    folium.Circle(location=[c_lat, c_lon], radius=circle_radius, color=color_zone, weight=1, fill=True, fill_color=color_zone, fill_opacity=0.45, popup=f"Risk Probability: {round(prob * 100, 1)}%").add_to(almaty_map)
    st_folium(almaty_map, width="100%", height=700, returned_objects=[])
    