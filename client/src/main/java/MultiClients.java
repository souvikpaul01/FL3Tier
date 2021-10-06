import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

// this class is for simulation in one machine without too many real clients
class MultiClients implements Runnable {

    @Override
    public void run() {
        String DEFAULT_IP = "30.20.1.2";
        int DEFAULT_PORT = 8000;

        int DEFAULT_TIMEOUT = 5000;
        int layer = 2;

        FileClient c = FileClient.connect(DEFAULT_IP, DEFAULT_PORT, DEFAULT_TIMEOUT);
        try {
            c.download("server_model.zip");
        }catch (IOException e) {
            System.out.println(e.getMessage());
        }

        //local update
        localUpdate localModel = new localUpdate();
        localModel.id = c.id + "";
        try {
            localModel.clientUpdate();
        } catch (IOException e) {
            e.printStackTrace();
        }

        Map<String, INDArray> map = new HashMap<>();
        Map<String, INDArray> paramTable = localUpdate.transferred_model.paramTable();
        map.put("weight", paramTable.get(String.format("%d_W", layer)));
        map.put("bias", paramTable.get(String.format("%d_b", layer)));
        try {
            c.uploadParamTable(map);
        } catch (IOException e) {
            e.printStackTrace();
        }
        c.quit();
    }
}
