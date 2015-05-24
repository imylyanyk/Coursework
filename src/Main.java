import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Main {

    public static final boolean DEBUG_MODE = false;

    public static void main(String[] args) throws IOException, InterruptedException {
        System.out.println("Demo :: Start");
        //FeedForward.Demo();
        //TeachDemo();
        //AlmostReality();
        //PlainDemo();
        //PartTest();

        Multithread();

        System.out.println("End of Demo");
    }

    private static void Multithread() throws IOException, InterruptedException {
        MultithreadExample example = new MultithreadExample();
        example.Run();
    }

    private static void PlainDemo() {
        List<Integer> sizes = new ArrayList<Integer>();
        sizes.add(1);/* input */
        sizes.add(1); /* output */
        NeuralNetwork network = new NeuralNetwork(sizes, true);
        NeuralNetwork new_network = new NeuralNetwork(sizes, false);

        int MIN = 1000;
        for(int i=-MIN; i<MIN; i++) {
            double[] data = new double[2];
            data[0] = 1;
            data[1] = i;
            double val = i * 3 + 2;
            DoubleMatrix x = FeedForward.g(new DoubleMatrix(data));
            x.put(0, 0, 1);
            x.print();
            FeedForward.TeachNetwork(network, new_network, x, DoubleMatrix.zeros(1).put(0, 0, (val > 0) ? 1 : 0));
        }
        /*
        for (int i = 1; i < new_network.Layers.size(); i++) {
            new_network.Layers.get(i).Theta = new_network.Layers.get(i).Theta.div(MIN*2);
        }
        network.Layers.get(1).Theta.print();
        */
        network.Layers.get(1).Theta.print();

        for(int i=-10; i<10; i++) {
            double[] data = new double[2];
            data[0] = 1;
            data[1] = i;
            double val = i * 3 + 2;
            DoubleMatrix x = FeedForward.g(new DoubleMatrix(data));
            x.put(0,0,1);
            System.out.print("x = ");
            x.print();
            DoubleMatrix result = FeedForward.Process(network, x);
            System.out.print("val = " + val + ", ");
            result.print();
        }
    }

    public static void PartTest() {
        DoubleMatrix m = new DoubleMatrix(new double[][] {{1, 2}, {3, 4}});
        DoubleMatrix n = new DoubleMatrix(new double[][] {{1, 2}, {3, 4}});
        m.add(n).print();
        m.print();

    }

    public static void AlmostReality() throws IOException {

        long startTime = System.currentTimeMillis();

        List<Integer> sizes = new ArrayList<Integer>();
        sizes.add(784);
        sizes.add(40);
        sizes.add(10);
        NeuralNetwork network = new NeuralNetwork(sizes, true);
        NeuralNetwork new_network = network;

        DigitImageLoadingService service = new DigitImageLoadingService("Training-Labels", "Training-Images", true);
        List<DigitImage> images = service.loadDigitImages();

        System.out.println("Loaded images!");

        for(DigitImage image : images) {
            double[] data = new double[image.getData().length + 1];
            data[0] = 1;
            for (int i = 0; i < image.getData().length; i++) {
                data[i+1] = image.getData()[i];
            }
            FeedForward.TeachNetwork(network, new_network, new DoubleMatrix(data), DoubleMatrix.zeros(10).put(image.getLabel(), 1));
        }

        Random random = new Random();
        int correct = 0;

        int TRAINING_SIZE = 100;
        for (int i = 0; i < TRAINING_SIZE; i++) {
            DigitImage image = images.get(random.nextInt(images.size()));

            double[] data = new double[image.getData().length + 1];
            data[0] = 1;
            for (int j = 0; j < image.getData().length; j++) {
                data[j+1] = image.getData()[j];
            }


            DoubleMatrix result = FeedForward.Process(new_network, new DoubleMatrix(data));

            int best = 0;
            for (int j = 0; j < result.rows; j++) {
                if(result.get(best) < result.get(j))
                    best = j;
            }

            if(DEBUG_MODE) {
                System.out.println();
                result.print();
                image.print();
                System.out.println("Original label: " + image.getLabel() + ". Calculated: " + best);
            }
            if(best == image.getLabel())
                correct++;
        }


        System.out.println("Correct: " + correct + " out of " + TRAINING_SIZE);

        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time elapsed: " + estimatedTime + " ms.");
    }

    private static void Demo() {
        List<Integer> sizes = new ArrayList<Integer>();
        sizes.add(784 + 1);
        sizes.add(25 + 1);
        sizes.add(10);
        NeuralNetwork network = new NeuralNetwork(sizes, false);
        DoubleMatrix x = DoubleMatrix.rand(784 + 1);
        DoubleMatrix y = DoubleMatrix.zeros(10);
        x.put(0, 0, 1);
        y.put(5, 0, 1);

        DoubleMatrix res = FeedForward.Process(network, x);
        res.print();
    }

    private static void TeachDemo() {
        List<Integer> sizes = new ArrayList<Integer>();
        sizes.add(784);
        sizes.add(25);
        sizes.add(10);
        NeuralNetwork network = new NeuralNetwork(sizes, false);
        DoubleMatrix x = DoubleMatrix.rand(784 + 1);
        DoubleMatrix y = DoubleMatrix.zeros(10);
        x.put(0, 1);
        y.put(5, 1);

        //FeedForward.TeachNetwork(network, x, y);
    }

    private void exportImages() throws IOException {
        DigitImageLoadingService service = new DigitImageLoadingService("Training-Labels", "Training-Images", false);
        List<DigitImage> images = service.loadDigitImages();
    }
}
