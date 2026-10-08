# I-BID Pilot — Streamlit Cloud Deployment Checklist

## 1. Repo contents (put these at the repo root)

| File | Required? | Notes |
|---|---|---|
| `I-BID_Pilot.py` | ✅ | Main app (36,658 lines, self-contained) |
| `requirements.txt` | ✅ | Included — streamlit, pandas, numpy, plotly, scipy |
| `packages.txt` | optional | Empty by default (system libs not needed) |
| `ibid_universal_collector.py` | optional | Your Universal Collector module. App runs without it (import is try/except protected), but the Données Historiques collector section will be disabled |
| `I-BID App header.png` | optional | Falls back to embedded base64 header |
| `I-BID page acceuil background.png` | optional | Wrapped in try/except |
| `I_BID Logo orange.png` | optional | Wrapped in try/except |

## 2. Deploy steps

1. Create a GitHub repo, push the files above.
2. Go to **share.streamlit.io** → **New app** → pick the repo.
3. Main file path: `I-BID_Pilot.py` → **Deploy**.
4. First boot takes ~5 min (scipy + plotly are heavy).

## 3. After deploy — connect the Android app

1. Copy your app URL: `https://<your-app>.streamlit.app`
2. Open `IBidAndroid/app/build.gradle` and replace `IBID_URL`:
   ```gradle
   buildConfigField "String", "IBID_URL", ""https://<your-app>.streamlit.app/""
   ```
3. Build the signed APK/AAB in Android Studio → distribute.

## 4. Verified facts about this version (v2.6.2)

- **Real imports**: streamlit, pandas, numpy, plotly (px + graph_objects + subplots), scipy.optimize.linprog (LP reallocation)
- **No OCR/PDF libs imported by this file** — mentions of pytesseract/fitz/pdfplumber are user-facing warning text only, inside the optional collector module
- **session_state-heavy**: the app relies on `st.session_state`; Streamlit Cloud handles this fine, but note user data is per-session — downloads are the persistence layer
- `use_container_width=True` used throughout → requires Streamlit ≥ 1.29 (we pin 1.40.1)

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| App sleeps after inactivity | Normal on free tier — first load takes a few seconds to wake |
| `ModuleNotFoundError: ibid_universal_collector` | Should NOT crash (guarded). If it does, ensure you uploaded the latest v2.6.2 file |
| LP/optimization section slow | scipy linprog on large scenario matrices is CPU-bound; free tier has 1 CPU — expected |
| File upload > 200 MB fails | `maxUploadSize = 500` set in config.toml |
