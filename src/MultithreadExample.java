import org.jblas.DoubleMatrix;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Ivan on 24-May-15.
 */
public class MultithreadExample {

    List<DigitImage> Images;
    int NUMBER_OF_WORKERS;

    public MultithreadExample() {
        NUMBER_OF_WORKERS = 100;
    }



    public void Run() throws IOException, InterruptedException {
        /*TODO measure time*/
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

        int batchSize = images.size() / NUMBER_OF_WORKERS;
        System.out.println(batchSize);
        Thread[] threads = new Thread[NUMBER_OF_WORKERS];
        for (int i = 0; i < NUMBER_OF_WORKERS; i++) {
            threads[i] = new Thread(new MyRunnable(images, i * batchSize, (i+1)*batchSize, network, new_network));
            threads[i].start();
        }

        for(Thread thread : threads) {
            thread.join();
        }
        System.out.println("Learning ended");





        System.out.println("Testing started");
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

            if(Main.DEBUG_MODE) {
                System.out.println();
                result.print();
                image.print();
                System.out.println("Original label: " + image.getLabel() + ". Calculated: " + best);
            }
            if(best == image.getLabel()) {
                correct++;
                //System.out.print(best + " ");
            }
        }


        System.out.println("Correct: " + correct + " out of " + TRAINING_SIZE);

        long estimatedTime = System.currentTimeMillis() - startTime;
        System.out.println("Time elapsed: " + estimatedTime + " ms.");
    }

    public class MyRunnable implements Runnable {

        List<DigitImage> trainingSet;
        NeuralNetwork learnedNetwork;
        NeuralNetwork mainNetwork;
        int start, end;

        public MyRunnable(List<DigitImage> set, int _start, int _end, NeuralNetwork initialNetwork, NeuralNetwork network) {
            super();
            trainingSet = set;
            learnedNetwork = network;
            mainNetwork = initialNetwork;
            start = _start;
            end = _end;
        }

        public void run(){
            for (int im = start; im <= end && im < trainingSet.size(); im++) {

                DigitImage image = trainingSet.get(im);

                double[] data = new double[image.getData().length + 1];
                data[0] = 1;

                for (int i = 0; i < image.getData().length; i++) {
                    data[i+1] = image.getData()[i];
                }

                FeedForward.TeachNetwork(mainNetwork, learnedNetwork, new DoubleMatrix(data), DoubleMatrix.zeros(10).put(image.getLabel(), 1));
            }
            if(Main.DEBUG_MODE)
                System.out.println(start + " ended");
        }
    }
}
