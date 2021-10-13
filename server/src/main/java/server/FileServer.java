package server;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.IOException;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashMap;
import java.util.Map;

public class FileServer {

    public static String onDeviceModelPath = "res/clientModel";
    public static Map<Integer, Map<String, INDArray>> cache = new HashMap<>();
    public static FederatedModel federatedmodel = new FederatedModel();
    private static ServerSocket serverSocket;
    private static int clientNum = 1;

    private void init(int port, int timeout) {
        try {
            serverSocket = new ServerSocket(port);
            serverSocket.setSoTimeout(timeout);
        } catch (Exception e) {
            System.out.println("Couldn't open socket. " + e.getMessage());
            return;
        }
        System.out.println("Server started on port " + port + " with timeout " + timeout + "ms");
    }

    private void run() throws InterruptedException {

        int curID = 1;
        // here plus 2
        while (curID < clientNum + 1) {
            try {
                Socket clientSocket = serverSocket.accept();
                new Thread((Runnable) new ServerConnection(clientSocket, curID)).start();
                System.out.println("client " + curID + " connected!");
                curID++;
            } catch (IOException e) {
                System.out.println("Error accepting client connection: " + e.getMessage());
            }

        }
        System.out.println("exit the while loop");
    }


    public static void main(String[] args) throws IOException, InterruptedException {
        // Run server
        int DEFAULT_PORT = 8000;
        int DEFAULT_TIMEOUT = 60 * 1000;//30 seconds
        int round = 1;

        //to do: select client
        federatedmodel.initModel();

        FileServer fileserver = new FileServer();
        fileserver.init(DEFAULT_PORT, DEFAULT_TIMEOUT);

        for (int r = 0; r < round; r++) {
            System.out.println("\n\nround:" + r);
            fileserver.run();

            // need to add time to wait for the upload
            Thread.sleep(60 * 5000 );
            System.out.println("the cache size is " + cache.size());

            federatedmodel.AverageWeights(2, 0.5, cache.size());
        }

    }

}
