import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.dataset.DataSet;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Collections;
import java.util.List;
import java.util.Random;
public class localUpdate {


    String filenameTrain = "C:\\Users\\souvik\\Downloads\\dev\\FL3Tier\\client0\\res\\trashnet\\";
    String id = null;

    private static Logger log = LoggerFactory.getLogger(localUpdate.class);
    private static final int nEpochs = 100;
    private static final int batchSize = 19;
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int CHANNELS = 3;
    private static final int N_OUTCOMES = 6;
    public int nSamples = 2527;


    public static MultiLayerNetwork model = null;
    public static MultiLayerNetwork transferred_model = null;

    //Set up a fine-tune configuration
    public static FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new Nesterovs(5e-5))
            .seed(100)
            .build();


    public void clientUpdate() throws IOException {
        //load model from server
        System.out.println("loading model...");
        String inFile = FileClient.downloadDir + "server_model.zip";
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(inFile); //this method load a multilayernetwork model from a file
            System.out.println("load model finish!");
        } catch (IOException e) {
            e.printStackTrace();
        }

        //transfer model
        transferred_model = new TransferLearning.Builder(model)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(1)
                .build();

        System.out.println("loading data...");

        File folder = new File(filenameTrain);
        File[] digitFolders;
        digitFolders = folder.listFiles();

        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scalar = new ImagePreProcessingScaler(0, 1);
        INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT * WIDTH});
        INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

        int n = 0;
        assert digitFolders != null;
        for (File digitFolder : digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();

            for (File imgFile : imageFiles) {
                //System.out.println(imgFile);
                INDArray img = nativeImageLoader.asRowVector(imgFile);
                //INDArray img = nativeImageLoader.asMatrix(imgFile);
                log.info(img.shapeInfoToString());
                // System.out.println(img.length());
                scalar.transform(img);
                //System.out.println(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }
        }
        //Joining input and output matrices into a dataset
        DataSet dataSet = new DataSet(input, output);
        //Convert the dataset into a list
        List<DataSet> listDataSet = dataSet.asList();
        //Shuffle content of list randomly
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));
//            int batchSize = 10;

        //Build and return a dataset iterator
        DataSetIterator dsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);


        // RecordReader rr = new CSVRecordReader();
        //   filenameTrain = filenameTrain + "_" + id + ".csv";
        //  System.out.println("The training data file is: " + filenameTrain);
     /*   try {
            rr.initialize(new FileSplit(new File(filenameTrain)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }*/


        //DataSetIterator trainIter = new RecordReaderDataSetIterator(dsi, batchSize, 561, 6);
        System.out.println("load data finish!");
        transferred_model.setListeners(new ScoreIterationListener(133));  //Print score every 10 parameter updates

        transferred_model.fit(dsi, nEpochs);
        System.out.println("training finished");

        // get the updated client model's parameter table
        Map<String, INDArray> paramTable = transferred_model.paramTable();

        //write model
        String outFile = FileClient.uploadDir + id + ".zip";
        try {
            ModelSerializer.writeModel(transferred_model, outFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    /*public void clientUpdate1() {
        //load model from server
        System.out.println("loading model...");
        String inFile = FileClient.downloadDir + "server_model.zip";
        try {
            model = ModelSerializer.restoreMultiLayerNetwork(inFile); //this method load a multilayernetwork model from a file
            System.out.println("load model finish!");
        } catch (IOException e) {
            e.printStackTrace();
        }

        //transfer model
        transferred_model = new TransferLearning.Builder(model)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(1)
                .build();

        System.out.println("loading data...");
        RecordReader rr = new CSVRecordReader();
        filenameTrain = filenameTrain + "_" + id + ".csv";
        System.out.println("The training data file is: " + filenameTrain);
        try {
            rr.initialize(new FileSplit(new File(filenameTrain)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }


        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 561, 6);
        System.out.println("load data finish!");
//        transferred_model.setListeners(new ScoreIterationListener(50));  //Print score every 10 parameter updates

        transferred_model.fit(trainIter, nEpochs);
        System.out.println("training finished");

        // get the updated client model's parameter table
        Map<String, INDArray> paramTable = transferred_model.paramTable();

        //write model
        String outFile = FileClient.uploadDir + id + ".zip";
        try {
            ModelSerializer.writeModel(transferred_model, outFile, true);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void hello(){
        System.out.println("hello from localUpdate!");
        System.out.println(id);
    }
*/
    public static void  evaluateModel() throws IOException {
        int nSamples = 204;
        String filenameTrain = "C:\\Users\\souvik\\Downloads\\dev\\FL3Tier\\client0\\res\\test\\";
        File folder = new File(filenameTrain);
        File[] digitFolders = folder.listFiles();

        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scalar = new ImagePreProcessingScaler(0, 1);
        INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT * WIDTH});
        INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

        int n = 0;
        assert digitFolders != null;
        for (File digitFolder : digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();

            for (File imgFile : imageFiles) {
                //System.out.println(imgFile);
                INDArray img = nativeImageLoader.asRowVector(imgFile);
                //INDArray img = nativeImageLoader.asMatrix(imgFile);
                log.info(img.shapeInfoToString());
                // System.out.println(img.length());
                scalar.transform(img);
                //System.out.println(img);
                input.putRow(n, img);
                output.put(n, labelDigit, 1.0);
                n++;
            }

        }
        //Joining input and output matrices into a dataset
        DataSet dataSet = new DataSet(input, output);
        //Convert the dataset into a list
        List<DataSet> listDataSet = dataSet.asList();
        //Shuffle content of list randomly
        Collections.shuffle(listDataSet, new Random(System.currentTimeMillis()));

        //Build and return a dataset iterator
        DataSetIterator testDsi = new ListDataSetIterator<DataSet>(listDataSet, batchSize);


        System.out.print("Evaluating Model...");
        Evaluation eval = transferred_model.evaluate(testDsi);
        System.out.print(eval.stats());
    }
}