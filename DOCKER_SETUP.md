# Docker Setup for Road Network Analysis

This document provides instructions for setting up the local Overpass API instance required for the Road Network scoring pipeline.

## 1. Prerequisites
- Docker Desktop installed and running.
- At least 16GB of RAM (recommended for the India-wide dataset).
- ~20GB of free disk space.

## 2. Pull the Image
We use the optimized Overpass API image:
```bash
docker pull wiktorn/overpass-api
```

## 3. Deployment (India Dataset)
Run the following command to start the container. This will download and index the India OSM data.

```powershell
docker run -d `
  --name overpass_india `
  -p 12345:80 `
  -e OVERPASS_META=yes `
  -e OVERPASS_MODE=init `
  -e OVERPASS_PLANET_URL=https://download.geofabrik.de/asia/india-latest.osm.bz2 `
  -v overpass_db:/db `
  wiktorn/overpass-api
```

### Note on Ports:
- The pipeline is configured to look at `http://127.0.0.1:12345/api`.
- If you change the port, update `OSM_OVERPASS_URL` in `Road_Network/config.py`.

## 4. Health Check
Once the container is running (it may take 30-60 minutes to index initially), verify it by visiting:
`http://127.0.0.1:12345/api/status`

You should see a status page showing `Connected as: ...` and `Rate limit: 0` (which is normal for this Docker setup).

## 5. Troubleshooting
- **Container keeps restarting**: Check if you have enough disk space or if the Geofabrik URL has changed.
- **403 Forbidden**: Ensure you are using `127.0.0.1` and not `localhost` (to avoid IPv6 issues).
- **No data returned**: Ensure the point you are querying is within the geographic bounds of the dataset you downloaded (India).

## 6. Optimization
To stop the container when not in use:
```bash
docker stop overpass_india
```
To restart it:
```bash
docker start overpass_india
```
