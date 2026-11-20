# Urban Risk Desk — Implementation Report & User Guide

## 1. Overview of Redesign & Improvements
The **Urban Risk Desk** application has been fully redesigned to match professional government intelligence dashboards (dark navy theme `#081321`, cyan/electric blue accents `#29d3e8`, subtle cards `#122338`, and risk indicators). All existing machine learning models, datasets, and FastAPI endpoints have been preserved and fully integrated.

---

## 2. Files Modified & Created
- **`app_prepared.py`**: Added the `GET /api/borough-summary` endpoint returning exact statistics for all 5 boroughs calculated from `nyc311_clean.csv`.
- **`urban-risk-desk.html`**: Redesigned front-end with a professional sidebar navigation, KPI cards, Leaflet map with genuine borough boundaries (`nyc_borough_boundaries.geojson`), choropleth risk coloring, hover info panels, click-to-zoom, resource priority board, and interactive What-If Simulator with Chart.js visualization.
- **`nyc_borough_boundaries.geojson`** (New): Authoritative GeoJSON containing true polygon boundaries for Manhattan, Brooklyn, Queens, Bronx, and Staten Island.

---

## 3. Key Features Implemented

1. **Interactive NYC Borough Map**:
   - Genuine borough polygons loaded via GeoJSON.
   - **Hover interaction without clicking**: Instantly displays a borough overview panel on the right (Total complaints, slow rate, open/closed counts, top complaint categories, and top ZIP codes).
   - **Click-to-zoom**: Smoothly zooms into the selected borough (`fitBounds`), highlights its boundary in cyan, and dims unselected boroughs.
   - **Reset Control**: Breadcrumb navigation (`New York City > [Borough]`) and "Reset City View" button to restore the citywide view.

2. **Resource Priority Board**:
   - Fully connected to `priority_by_zip_type.csv` via FastAPI (`GET /api/priority`).
   - Displays ZIP-code and complaint-type priority rankings with volume, predicted risk, impact weights, and priority scores.
   - Search filter for quick stakeholder lookup.

3. **What-If Resource Allocation Simulator**:
   - Interactive sliders for available response crews and complaint priority weights (Heating, Street Conditions, Noise).
   - Fully connected to the backend optimization endpoint (`POST /api/simulate`).
   - Visualizes allocation results dynamically using Chart.js bar graphs and detailed breakdown tables.

4. **Overview & Analytics Dashboards**:
   - KPI cards for Total Complaints, Slow Complaint Rate, Open Complaints, and Clustered Groups.
   - Doughnut and line charts powered by Chart.js.

---

## 4. How to Run the Application

1. **Start the FastAPI Backend**:
   ```bash
   python app_prepared.py
   ```
   *(Or run via Uvicorn: `uvicorn app_prepared:app --reload --host 127.0.0.1 --port 8000`)*

2. **Access the Dashboard**:
   Open your browser and navigate to:
   `http://127.0.0.1:8000`

---
*All project files, ML models, and backend endpoints remain fully operational and intact.*
