import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

class MultiClients implements Runnable {

    @Override
    public void run() {
//        String DEFAULT_IP = "30.20.0.8";
        String DEFAULT_IP = "192.168.0.106";
        int DEFAULT_PORT = 8000;

        int DEFAULT_TIMEOUT = 5000;
        int layer = 2;

        FileClient c = FileClient.connect(DEFAULT_IP, DEFAULT_PORT, DEFAULT_TIMEOUT);

        //local update
        localUpdate localModel = new localUpdate();
        localModel.id = c.id + "";
        localModel.clientUpdate();

        Map<String, INDArray> map = new HashMap<>();
        Map<String, INDArray> paramTable = localUpdate.transferred_model.paramTable();
        map.put("weight", paramTable.get(String.format("%d_W", layer)));
        map.put("bias", paramTable.get(String.format("%d_b", layer)));
        try {
            c.uploadParamTable(map);
        } catch (IOException e) {
            e.printStackTrace();
        }

        //upload local model to server
//        c.upload(new File(FileClient.uploadDir + c.id + ".zip"), c.id + ".zip");
        //disconnect
        c.quit();
    }
}
