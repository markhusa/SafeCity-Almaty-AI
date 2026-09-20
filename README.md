SafeCity Almaty: Predictive AI Public Safety Framework

[🔗 OPEN INTERACTIVE MAP WEBSITE] (PLACE_FOR_FUTURE_WEBSITE_LINK)

A Geospatial AI (GeoAI) platform engineered to predict and monitor public safety threats and traffic accident risks across the Almaty urban agglomeration, including Kaskelen, Talgar, Konaev, and key regional highways. The system processes urban data streams to generate real-time risk assessments with a strict 10-minute refresh rate.

1. Key Engineering and Feature Architecture
   
   Dataset Scaling: Built a comprehensive database containing 75,000 operational records, utilizing manual probability distribution to accurately mirror the cyclical patterns of the urban environment.

   On-Demand Inference: The system operates in real time based on system temporal parameters, automatically rounding to the nearest 10-minute block for dynamic risk adaptation.

   Advanced Feature Engineering (5 Core Metrics):

   1. is_dark — Autonomous darkness metric adjusting to early winter sunsets (18:00) and extended summer days (21:00).
   2. tourist_season — Tracks peak seasonal loads on recreational areas (winter in the Medeu Mountain Cluster, summer beach season in Konaev City).
   3. road_hazard — Evaluates critical accident probabilities on high-speed entry routes (Tashkent tract, Kulzhinka, Talgar Highway), amplified by winter ice parameters.
   4. cctv_coverage — Models the mitigating impact of the Sergek CCTV network as a predictive security shield that directly lowers threat probability.
   5. crowd_density — Tracks pedestrian and vehicular traffic volume variations based on location type and day of the week (city core during weekdays versus recreational areas on weekends).
   
   Spatial Optimization (Collision Detection): Implemented a vectorized location boundary check (min_distance = 0.0072) utilizing folium.Circle markers fixed at 650 meters on the ground. This ensures maximum visual coverage and precise edge-to-edge alignment at any zoom scale without marker overlapping.

   Premium UI/UX Palette: Implemented a high-contrast analytical theme where safe zones are rendered in deep dark green (darkgreen), caution areas in neon lime (lime), and high-risk vectors in solid red (red).

 2. Analytical Insights and Model Performance

      The trained Logistic Regression model achieves high predictive granularity, demonstrating that public safety threats interact dynamically depending on environmental context.
   
      Scenario 1: Late Summer Night (July, 23:00)
      During summer nights, the central urban core remains relatively stable due to strong Sergek CCTV coverage and high street illumination levels, though nightlife districts show moderate caution indicated by lime    markers. Concurrently, major high-speed entry tracts and suburban highways experience a critical risk surge due to pitch darkness and high velocity factors. Most notably, Konaev City turns red, capturing peak weekend resort traffic and intense infrastructure load.

      Scenario 2: Early Winter Morning (January, 08:00)
      In January at 08:00 AM, dawn has not yet broken in Almaty, escalating the baseline hazard. While Konaev City returns to complete dark green due to the off-season winter lull, the southern Medeu and Shymbulak Mountain Cluster, along with the Dostyk Avenue Corridor, shift heavily into the red zone. The AI accurately detects a critical winter weekend confluence: extreme tourist density heading to ski resorts, freezing mountain road conditions, and imminent traffic gridlocks.

   3. Tech Stack
   
   Language: Python 3.13
   
   Machine Learning & Core Analytics: Scikit-learn (Logistic Regression, StandardScaler, OneHotEncoder), Pandas, NumPy
   
   Geospatial Engine: Folium (OpenStreetMap France Tiles)
   
   User Interface: Standalone responsive interactive dashboard with an injected glassmorphic dark-theme control panel.
   
