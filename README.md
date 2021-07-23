# FL3Tier

Implementation of federated learning on real-world setting.

## Overview
We implement a real-world federated learning project based on Java. There are one server and many clients to 
collaboratively train a model. We use DL4J as the deep learning library. The data we use is a Human 
Activity Recognition (HAR) dataset.

## Server-side
```
main
└─ java
       ├─ baselinemodel.java
       └─ server
              ├─ FederatedModel.java
              ├─ FileServer.java
              └─ ServerConnection.java
```
The base model is a MLP composed of one input layer, one output layer and one hidden layer with 1000
units using ReLU activation. The model can be changed in FederatedModel.java.

## Client-side
```
main
└─ java
       ├─ FileClient.java
       └─ localUpdate.java
```

## Communication
Considering there are no enough client devices to do experiments, we use java multithreading to 
simulate the real clients, for example, run 100 threads in one device. If you have the many real devices,
the project also works.

## Reference
https://github.com/Cautiousss/federated-learning