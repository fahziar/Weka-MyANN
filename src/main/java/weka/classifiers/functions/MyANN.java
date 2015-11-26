package weka.classifiers.functions;

import java.util.Enumeration;
import java.util.Vector;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

/**
 * Created by fahziar on 23/11/2015.
 */
public class MyANN extends Classifier 
    implements OptionHandler {
    // layer[nomer layer][nomer neuron] = nilai pada neuron tsb
    private double [][] layer;
    // weight[nomer layer][nomer neuron  pada layer][nomer neuron tujuan] = nilai pada neuron tsb
    private double [][][] weight;
    // delta weight
    // dw[nomer layer][nomer neuron pada layer][nomer neuron tujuan] = delta weight
    private double [][][] dw;
    private String hiddenLayers;
    private int learningRule;
    private int activationFunction;
    private double learningRate;
    private double momentum;
    private int maxIteration;
    private double MSE;
    
    public static void main(String [] argv) {
        runClassifier(new MyANN(), argv);
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
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        init(data);
    }
    
    private void init(Instances data) throws Exception {
        int nLayer=2;
        int nOutput = data.numClasses();
        
        NominalToBinary filter = new NominalToBinary();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        int nInput = data.numAttributes();
        
        int nMaxNeuron = Math.max(nInput, nOutput);
        
        if (!hiddenLayers.equals("n")){
            String[] splits = hiddenLayers.split(",");
            int[] layers = new int[splits.length];
            int i=0;
            for (String split: splits) {
                layers[i] = Integer.parseInt(split);
                i++;
            }
            nLayer += layers.length;
            i = Utils.maxIndex(layers);
            nMaxNeuron = Math.max(nMaxNeuron, layers[i]);
        }
        
        layer = new double [nLayer][nMaxNeuron];
        weight = new double [nLayer][nMaxNeuron][nMaxNeuron];
        dw = new double [nLayer][nMaxNeuron][nMaxNeuron];
    }
    
    @Override
    public Enumeration<Option> listOptions() {
    
        Vector newVector = new Vector(4);

        newVector.addElement(new Option(
                  "\tLearning Rate for the backpropagation algorithm.\n"
                  +"\t(Value should be between 0 - 1, Default = 0.3).",
                  "L", 1, "-L <learning rate>"));
        newVector.addElement(new Option(
                  "\tMomentum Rate for the backpropagation algorithm.\n"
                  +"\t(Value should be between 0 - 1, Default = 0.2).",
                  "M", 1, "-M <momentum>"));
        newVector.addElement(new Option(
                  "\tMaximum iteration to train.\n"
                  +"\t(Default = 500).",
                  "N", 1,"-N <maximum iteration>"));
        newVector.addElement(new Option(
                  "\tThe hidden layers to be created for the network.\n"
                  + "\t(Value should be a list of comma separated Natural \n"
                  + "\tnumbers or the letter 'n' = no hidden layer, \n"
                  + "\tdefault 'n' \n",
                  "H", 1, "-H <comma seperated numbers for nodes on each layer>"));

        return newVector.elements();
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        //the defaults can be found here!!!!
        String learningString = Utils.getOption('L', options);
        if (learningString.length() != 0) {
          learningRate = new Double(learningString);
        } else {
          learningRate = 0.3;
        }
        String momentumString = Utils.getOption('M', options);
        if (momentumString.length() != 0) {
          momentum = new Double(momentumString);
        } else {
          momentum = 0.2;
        }
        String epochsString = Utils.getOption('N', options);
        if (epochsString.length() != 0) {
          maxIteration = Integer.parseInt(epochsString);
        } else {
          maxIteration = 500;
        }
        String hiddenLayersString = Utils.getOption('H', options);
        if (hiddenLayersString.length() != 0) {
          this.hiddenLayers = hiddenLayersString;
        } else {
          this.hiddenLayers = "n";
        }

        Utils.checkForRemainingOptions(options);
    }
    
    @Override
    public String[] getOptions() {

        String[] options = new String[8];
        int current = 0;
        options[current++] = "-L"; options[current++] = "" + learningRate; 
        options[current++] = "-M"; options[current++] = "" + momentum;
        options[current++] = "-N"; options[current++] = "" + maxIteration; 
        options[current++] = "-H"; options[current++] = "" + hiddenLayers;

        while (current < options.length) {
          options[current++] = "";
        }
        return options;
    }
    
    public static double sigmoid(double x) {return 1 / (1 + Math.exp(-x));}
    
    public static double sign(double x) {return Math.signum(x);}
    
    public static double linear(double x) {return x;}
}
