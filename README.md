# FL3Tier

Implementation of federated learning on real-world setting.

## Overview
We implement a real-world federated learning project based on Java. There are one server and many clients to 
collaboratively train a model. We use DL4J as the deep learning library.

## Server-side
server
    ├── build.gradle
    ├── gradle
    │   └── wrapper
    │       ├── gradle-wrapper.jar
    │       └── gradle-wrapper.properties
    ├── gradlew
    ├── gradlew.bat
    ├── res
    │   ├── clientModel
    │   ├── dataset
    │   └── serverModel
    │       ├── server_model.zip
    │       └── trained_nn.zip
    ├── settings.gradle
    └── src
        └── main
            └── java
                ├── baselinemodel.java
                └── server
                    ├── FederatedModel.java
                    ├── FileServer.java
                    └── ServerConnection.java


## Client-side
