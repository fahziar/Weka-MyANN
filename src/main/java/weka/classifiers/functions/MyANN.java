package weka.classifiers.functions;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instances;

/**
 * Created by fahziar on 23/11/2015.
 */
public class MyANN extends Classifier {
    // layer[nomer layer][nomer neuron] = nilai pada neuron tsb
    private double [][] layer;
    // weight[nomer layer][nomer neuron  pada layer][nomer neuron tujuan] = nilai pada neuron tsb
    private double [][][] weight;
    // delta weight
    // dw[nomer layer][nomer neuron pada layer][nomer neuron tujuan] = delta weight
    private double [][][] dw;
    private double learningRate;
    private double momentum;

    @Override
    public void buildClassifier(Instances data) throws Exception {

    }

    // Untuk single layer
    // gradient descent batch
    // untuk delta rule tinggal panggil ini setiap iterasi pada data, panggil applyDw kemudian resetDw()
    private void gradientDescentUpdateDw(int target){
        for (int i=0; i<dw[0].length; i++){
            for (int j=0; j<dw[0][i].length; j++){
                dw[0][i][j] = dw[0][i][j] + learningRate * (target - layer[1][0]) * weight[0][i][j];
            }
        }
    }


    //Reset delta weight
    private void resetDw(){
        for (int i=0; i<dw.length; i++){
            for (int j=0; j<dw[i].length; j++){
                for (int k=0; k<dw[i][j].length; k++){
                    dw[i][j][k] = 0;
                }
            }
        }
    }

    private void applyDw(){
        for (int i=0; i<dw[0].length; i++){
            for (int j=0; j<dw[0][i].length; j++){
                weight[0][i][j] = dw[0][i][j] + weight[0][i][j];
            }
        }
    }

    private void backPropagation(int[] target){
        //Reset delta weight
        resetDw();

        //Array yang berisi error dari tiap-tiap unit
        double [][] deltas = new double[layer.length][];
        for (int i=0; i<deltas.length; i++){
            deltas[i] = new double[layer[i].length];
        }

        //Output layer adalah layer terluar
        //hitung error untuk output layer
        for (int i=0; i<layer[layer.length - 1].length; i++){
            double o = layer[layer.length - 1][i];
            double t = target[i];
            deltas[layer.length - 1][i] = o * (1 - o) * (t - o);
        }

        //hitung error untuk hidden unit
        for (int i=0; i<layer.length - 1; i++){
            for (int j=0; j<layer[i].length; j++){
                double deltaOutput = 0;
                for (int k=0; k<weight[i][j].length; k++) {
                     deltaOutput = weight[i][j][k] * deltas[i + 1][k];
                }
                deltas[i][j] = layer[i][j] * (1 - layer[i][j]) * deltaOutput;
            }
        }

        //hitung nilai delta w untuk setiap weight
        for (int i=0; i<weight.length; i++){
            for (int j=0; j<weight[i].length; j++){
                for (int k=0; k<weight[i][j].length; k++){
                    dw[i][j][k] = learningRate * layer[i][j] * deltas[i][j] + momentum*dw[i][j][k];
                }
            }
        }

    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);

        return result;
    }

    public void testPltr(){
        double[][] data = {
                {1, 1, 0, 1},
                {1, 0, -1, -1},
                {1, -1, -0.5, -1}
        };
    }

    //test
    public static void main(String[] args){

    }
}
