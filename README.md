# FL3Tier

Implementation of federated learning on real-world setting.

## Overview
We implement a real-world federated learning project based on Java. There are one server and many clients to 
collaboratively train a model. We use DL4J as the deep learning library.

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

## Client-side
```
main
└─ java
       ├─ FileClient.java
       └─ localUpdate.java
```
