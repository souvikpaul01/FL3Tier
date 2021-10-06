package server;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class FederatedModel {

    public static int numInputs = 784;
    public static int numOutputs = 6;
    public static int batchSize = 20;
    private static final int HEIGHT = 28;
    private static final int WIDTH = 28;
    private static final int CHANNELS = 1;
    private static final int N_OUTCOMES = 6 ;
    int nSamples = 204;
    String filenameTrain = "/home/ubuntu/FL3Tier/server/res/trashnet";

    public static MultiLayerNetwork model = null;
    private static final String serverModel = "res/serverModel/server_model.zip";


    // average weights over mobile devices' models
    public void AverageWeights(int layer, double alpha, int K) throws IOException {
        System.out.println("The number of client is: " + K);

        //original model
        Map<String, INDArray> paramTable = model.paramTable();
        System.out.println(paramTable);
        INDArray weight = paramTable.get(String.format("%d_W", layer));
        INDArray bias = paramTable.get(String.format("%d_b", layer));
        INDArray avgWeights = weight.mul(alpha);
        System.out.println("the avgWeight is :\n" + avgWeights);
        INDArray avgBias = bias.mul(alpha);
        System.out.println("the avgBias is :\n" + avgBias);

        // average weights over mobile devices' models
        System.out.println("\nAveraging weights...");

        MultiLayerNetwork transferred_model = null;
        for (int i = 1; i < K + 1; i++) {
            System.out.println("enter K loop");
            System.out.println(FileServer.cache);
            if (FileServer.cache.containsKey(i)) {
                System.out.println("enter cache");
                paramTable = FileServer.cache.get(i);
                System.out.println("the get parameter is :\n" + paramTable);
//                weight = paramTable.get(String.format("%d_W", layer));
                weight = paramTable.get("weight");
                System.out.println("The client weight is: \n" + weight);
//                bias = paramTable.get(String.format("%d_b", layer));
                bias = paramTable.get("bias");
                System.out.println("The client bias is: \n" + bias);
                System.out.println("The process run in there");
                avgWeights = avgWeights.add(weight.mul(1.0 - alpha).div(K));
                avgBias = avgBias.add(bias.mul(1.0 - alpha).div(K));
            }
        }

        model.setParam(String.format("%d_W", layer), avgWeights);
        model.setParam(String.format("%d_b", layer), avgBias);

        System.out.println("\nWriting server model...");
        ModelSerializer.writeModel(model, serverModel, false);
        System.out.println("\nWriting server model Finished...");
        evaluateModel();

        FileServer.cache.clear();
    }

    public static void delete(List<File> files) {
        System.out.println("Deleting files...");
        int len = files.size();
        for (int i = 0; i < len; i++) {
            files.get(i).delete();
        }
        System.out.println("Files deleted");
    }

    public  void evaluateModel() throws IOException {


        File folder = new File(filenameTrain);
        File[] digitFolders = folder.listFiles();

        NativeImageLoader nativeImageLoader = new NativeImageLoader(HEIGHT, WIDTH);
        ImagePreProcessingScaler scalar = new ImagePreProcessingScaler(0,1);
        INDArray input = Nd4j.create(new int[]{nSamples, HEIGHT*WIDTH});
        INDArray output = Nd4j.create(new int[]{nSamples, N_OUTCOMES});

        int n = 0;
        for (File digitFolder: digitFolders) {
            int labelDigit = Integer.parseInt(digitFolder.getName());
            File[] imageFiles = digitFolder.listFiles();

            for (File imgFile : imageFiles) {
                INDArray img = nativeImageLoader.asRowVector(imgFile);
                //INDArray img = nativeImageLoader.asMatrix(imgFile);
                scalar.transform(img);
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


       /* System.out.print("Evaluating Model...");
        Evaluation eval = model.evaluate(testDsi);
        System.out.print(eval.stats());
*/
//        final String filenameTest = "res/dataset/test.csv";
        /*
        String filenameTest = "res/UCI-HAR/test_final.csv";
        //Load the test/evaluation data:
        RecordReader rrTest = new CSVRecordReader();
        try {
            rrTest.initialize(new FileSplit(new File(filenameTest)));
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 561, 6);
        System.out.println("\nEvaluate model....");
        Evaluation eval = new Evaluation(numOutputs);
        while (testIter.hasNext()) {
            DataSet t = testIter.next();
            INDArray features = t.getFeatures();
            INDArray labels = t.getLabels();
            INDArray predicted = model.output(features, false);
            eval.eval(labels, predicted);
        }
        // Print the evaluation statistics
        System.out.println(eval.stats());*/

    }


    public void initModel() throws IOException {

        System.out.println("initing model...");
        int seed = 100;
       // double learningRate = 0.001;
        int round = 10;
        int numHiddenNodes = 1000;
/*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes).nOut(numOutputs).build())
                .build();*/

  /*      MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new AdaDelta())
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nIn(CHANNELS).nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(16).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.MAX).build())

                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(32).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(3,3).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(128).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(64).build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder().kernelSize(1,1).stride(1,1).padding(1,1).activation(Activation.LEAKYRELU)
                        .nOut(numOutputs).build())
                .layer(new BatchNormalization())

                .layer(new SubsamplingLayer.Builder().kernelSize(2,2).stride(2,2).poolingType(SubsamplingLayer.PoolingType.AVG).build())

                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .name("output")
                        .nOut(numOutputs)
                        .dropOut(0.8)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS))
                .build();*/

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .l2(0.0005)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(1e-3))
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //nIn and nOut specify depth. nIn here is the nChannels and nOut is the number of filters to be applied
                        .nIn(CHANNELS)
                        .stride(1,1)
                        .nOut(20)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new ConvolutionLayer.Builder(5, 5)
                        //Note that nIn need not be specified in later layers
                        .stride(1,1)
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOutputs)
                        .activation(Activation.SOFTMAX)
                        .build())
                .setInputType(InputType.convolutionalFlat(HEIGHT, WIDTH, CHANNELS)) //See note below
                .build();


        model = new MultiLayerNetwork(conf);

        model.init();
        System.out.println("init model finish!\n");

        ModelSerializer.writeModel(model, serverModel, true);
        System.out.println("Write model to " + serverModel + " finish\n");

    }

}
