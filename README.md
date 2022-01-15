# GAN-BERT
- Fine-tuning Bert with GAN for semi supervised learning
- # Architecture
  <img src="https://github.com/ahmedkotb98/GAN-BERT/blob/main/images/ganbert.jpeg" width="600" height="600" />

## Training

- run notebook to train model and save your model and convert it to onnx format

## How to run the app

After installing necessary packages and don't forget to add onnx model in models folder, use the following command to run the app from project root directory-
  
```
uvicorn app.main:app
```
And visit http://127.0.0.1:8000/docs from your browser. You will be able to see swagger.

## How to run the app with docker

Make sure you are in the project root directory and you have started docker. Then create docker image using the following command.

```
docker-compose up --build
```

## Reference

- https://github.com/crux82/ganbert
