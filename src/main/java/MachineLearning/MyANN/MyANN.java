package MachineLearning.MyANN;

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
/**
 <!-- globalinfo-start -->
 * Class for generating a pruned or unpruned C4.5 decision tree. For more information, see<br/>
 * <br/>
 * Ross Quinlan (1993). C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers, San Mateo, CA.
 * <p/>
 <!-- globalinfo-end -->
 **/
public class MyANN extends Classifier
        implements OptionHandler {
    // layer[nomer layer][nomer neuron] = nilai pada neuron tsb
    private double [][] layer;
    // weight[nomer layer][nomer neuron  pada layer][nomer neuron tujuan] = nilai pada neuron tsb
    private double [][][] weight;
    // delta weight
    // dw[nomer layer][nomer neuron pada layer][nomer neuron tujuan] = delta weight
    private double [][][] dw;


    private String hiddenLayers = "h";
    private int learningRule = 1;
    private int activationFunction = 1;
    private double learningRate = 1.0;
    private double momentum = 1.0;
    private int maxIteration = 100;
    private double MSE = 10.0;

    public static void main(String [] argv) {
        runClassifier(new MyANN(), argv);
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

    //Dummy data
    public void testPltr(){
        double[][] data = {
                {1, 1, 0, 1},
                {1, 0, -1, -1},
                {1, -1, -0.5, -1}
        };
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

    /**
     <!-- options-start -->
     * Valid options are: <p/>
     *
     * <pre> -L &lt;learning rate&gt;
     *  Learning Rate for the backpropagation algorithm.
     *  (Value should be between 0 - 1, Default = 0.3).</pre>
     *
     * <pre> -M &lt;momentum&gt;
     *  Momentum Rate for the backpropagation algorithm.
     *  (Value should be between 0 - 1, Default = 0.2).</pre>
     *
     * <pre> -N &lt;number of epochs&gt;
     *  Number of epochs to train through.
     *  (Default = 500).</pre>
     *
     * <pre> -V &lt;percentage size of validation set&gt;
     *  Percentage size of validation set to use to terminate
     *  training (if this is non zero it can pre-empt num of epochs.
     *  (Value should be between 0 - 100, Default = 0).</pre>
     *
     * <pre> -S &lt;seed&gt;
     *  The value used to seed the random number generator
     *  (Value should be &gt;= 0 and and a long, Default = 0).</pre>
     *
     * <pre> -E &lt;threshold for number of consequetive errors&gt;
     *  The consequetive number of errors allowed for validation
     *  testing before the netwrok terminates.
     *  (Value should be &gt; 0, Default = 20).</pre>
     *
     * <pre> -G
     *  GUI will be opened.
     *  (Use this to bring up a GUI).</pre>
     *
     * <pre> -A
     *  Autocreation of the network connections will NOT be done.
     *  (This will be ignored if -G is NOT set)</pre>
     *
     * <pre> -B
     *  A NominalToBinary filter will NOT automatically be used.
     *  (Set this to not use a NominalToBinary filter).</pre>
     *
     * <pre> -H &lt;comma seperated numbers for nodes on each layer&gt;
     *  The hidden layers to be created for the network.
     *  (Value should be a list of comma separated Natural
     *  numbers or the letters 'a' = (attribs + classes) / 2,
     *  'i' = attribs, 'o' = classes, 't' = attribs .+ classes)
     *  for wildcard values, Default = a).</pre>
     *
     * <pre> -C
     *  Normalizing a numeric class will NOT be done.
     *  (Set this to not normalize the class if it's numeric).</pre>
     *
     * <pre> -I
     *  Normalizing the attributes will NOT be done.
     *  (Set this to not normalize the attributes).</pre>
     *
     * <pre> -R
     *  Reseting the network will NOT be allowed.
     *  (Set this to not allow the network to reset).</pre>
     *
     * <pre> -D
     *  Learning rate decay will occur.
     *  (Set this to cause the learning rate to decay).</pre>
     *
     <!-- options-end -->
     **/
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

    /**
     * This will return a string describing the classifier.
     * @return The string.
     */
    public String globalInfo() {
        return
                "A Classifier that uses backpropagation to classify instances.\n"
                        + "This network can be built by hand, created by an algorithm or both. "
                        + "The network can also be monitored and modified during training time. "
                        + "The nodes in this network are all sigmoid (except for when the class "
                        + "is numeric in which case the the output nodes become unthresholded "
                        + "linear units).";
    }


}