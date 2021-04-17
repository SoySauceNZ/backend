# Backend

Used to serve tensorflow model using API (model not included)

Currently One functionality
- POST `/predict` upload a image and get permutation

## Run
```bash
uvicorn main:app --reload
```

## Docs
http://localhost:8000/docs

## Endpoints

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