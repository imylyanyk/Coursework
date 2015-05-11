import org.jblas.DoubleMatrix;

import java.util.List;

/**
 * Created by Ivan on 28-Apr-15.
 */
public class NeuronLayer {
    public DoubleMatrix Theta;
    final double epsilon_init = 0.12;

    public NeuronLayer(int rows, int cols, boolean random) {
        if(random) {
            Theta = DoubleMatrix.rand(rows, cols);
            Theta = Theta.mul(2 * epsilon_init).sub(epsilon_init);
        }
        else {
            Theta = new DoubleMatrix(rows, cols);
        }
    }

    public NeuronLayer(DoubleMatrix matrix) {
        Theta = matrix;
    }
}
