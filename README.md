# Backend

Used to serve tensorflow model using API (model not included)

- POST `/predict` get prediction for one image name and data
- POST `/upload` upload a image
- GET `/list` list all image filenames
- GET `/images/:filename` Get image name

## Run
```bash
uvicorn main:app --reload
```

## Docs
https://api.severity.ml/docs

## POST Endpoints

POST `/upload` upload image using multipart/form-data
- recommended size is 500x500 rgb image
POST `/predict` predict using 
```json
{
  "filename": "string",
  "weather": "string",
  "brightness": 0,
  "speed": 0
}
```
