package weka.classifiers.functions;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instances;

/**
 * Created by fahziar on 23/11/2015.
 */
public class MyANN extends Classifier {
    // layer[nomer layer][nomer neuron] = nilai pada neuron tsb
    private int [][] layer;
    // weight[nomer layer][nomer neuron  pada layer][nomer neuron tujuan] = nilai pada neuron tsb
    private int [][][] weight;
    // delta weight
    // dw[nomer layer][nomer neuron pada layer][nomer neuron tujuan] = delta weight
    private int [][][] dw;

    @Override
    public void buildClassifier(Instances data) throws Exception {

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
}
