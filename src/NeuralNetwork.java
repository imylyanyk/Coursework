import java.util.ArrayList;
import java.util.List;

/**
 * Created by Ivan on 28-Apr-15.
 */
public class NeuralNetwork {
    public List<NeuronLayer> Layers;

    public NeuralNetwork(List<Integer> layersSizes, boolean random) {
        Layers = new ArrayList<>(layersSizes.size());
        Layers.add(null);
        for (int i = 0; i < layersSizes.size()-1; i++) {
            Layers.add(new NeuronLayer(layersSizes.get(i+1), layersSizes.get(i) + 1, random));
        }
    }
}
