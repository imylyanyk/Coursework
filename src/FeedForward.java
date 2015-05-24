import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;
import org.jblas.ranges.Range;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Ivan on 26-Apr-15.
 */
public class FeedForward {

    private final static double LEARNING_RATE = .1;

    public static void Demo() {
        DoubleMatrix a = new DoubleMatrix(new double[][] {{1, 2}, {3, 4}});
        DoubleMatrix b = new DoubleMatrix(new double[][] {{1, 2}, {3, 4}});
        DoubleMatrix c = a.mmul(b);
        a.print();
        b.print();
        c.print();

        a.transpose().print();
        a.print();
    }

    public static DoubleMatrix Process(NeuralNetwork network, DoubleMatrix x) {
        return TeachNetwork(network, null, x, null);
    }

    public static DoubleMatrix TeachNetwork(NeuralNetwork network, NeuralNetwork new_network, DoubleMatrix x, DoubleMatrix y/*vector*/) {
        int L = network.Layers.size();
        List<DoubleMatrix> a = new ArrayList<>();
        List<DoubleMatrix> z = new ArrayList<>();
        a.add(x);
        z.add(null);

        for (int i = 1; i < L; i++) {
            z.add(network.Layers.get(i).Theta.mmul(a.get(i - 1)));
            DoubleMatrix tmp = g(z.get(i));
            if(i != L-1)
                tmp = DoubleMatrix.concatVertically(DoubleMatrix.ones(1), tmp);
            a.add(tmp);
        }

        if(y == null || new_network == null) {
            // we don't know the right answer
            return a.get(L-1);
        }


        /* Backpropagation algo */

        List<DoubleMatrix> delta = new ArrayList<>();
        for (int i = 0; i < L; i++) {
            delta.add(null);
        }
        delta.set(L - 1, a.get(L - 1).sub(y));

        for (int i = L-2; i >= 1; i--) {
            DoubleMatrix tmp1 = network.Layers.get(i + 1).Theta.transpose().mmul(delta.get(i + 1));
            DoubleMatrix tmp2 = tmp1.mul(gStroke(DoubleMatrix.concatVertically(DoubleMatrix.ones(1), z.get(i))));

            // todo remove first row?
            DoubleMatrix tmp3 = DoubleMatrix.zeros(tmp2.rows - 1, tmp2.columns);
            int rIndex = 0;
            for(DoubleMatrix row : tmp2.rowsAsList()) {
                if(rIndex > 0)
                    tmp3.putRow(rIndex - 1, row);
                rIndex ++;
            }

            delta.set(i, tmp3);
        }

        //System.out.println("---");
        for (int i = 1; i < L; i++) {
            DoubleMatrix theta = delta.get(i).mmul(a.get(i-1).transpose());

            //network.Layers.get(i).Theta = network.Layers.get(i).Theta.add( theta.mul(-LEARNING_RATE) );
            new_network.Layers.get(i).Theta = new_network.Layers.get(i).Theta.add(theta.mul(-LEARNING_RATE));
        }

        return y;
    }

    public static DoubleMatrix gStroke(DoubleMatrix a) {
        DoubleMatrix res = g(a);
        res = res.mul(res.rsub(1));
        return res;
    }

    public static DoubleMatrix g(DoubleMatrix a) {
        DoubleMatrix res = a.dup();
        for (int i = 0; i < res.rows; i++) {
            for(int j=0;j<res.columns;j++) {
                res.put(i, j, g(res.get(i, j)));
            }
        }
        return res;
    }

    private static double g(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
