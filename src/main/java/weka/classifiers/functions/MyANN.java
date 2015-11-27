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
/**
 <!-- globalinfo-start -->
 * Class for artificial neural network classifier.<br/>
 * <br/>
 * <p/>
 <!-- globalinfo-end -->
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
 * <pre> -N &lt;maximum iterations&gt;
 *  Maximum number of iterations to train through.
 *  (Default = 500).</pre>
 *
 * <pre> -E &lt;MSE&gt;
 *  MSE value as threshold for training termination.
 *  (Default = 0.25).</pre>
 * 
 * <pre> -H &lt;comma seperated numbers for nodes on each layer&gt;
 *  The hidden layers to be created for the network.
 *  (Value should be a list of comma separated Natural
 *  numbers or the letter 'n' = no hidden layer,
 *  Default = 'n')</pre>
 * 
 * <pre> -R &lt;learning rule&gt;
 *  The learning rule for single perceptron.
 *  (Value should be 'ptr', 'batch', 'delta', 
 *  default = 'ptr')</pre>
 * 
 * <pre> -F &lt;activation function&gt;
 *  The activation function for network.
 *  (Value should be 'sigmoid', 'sign', 'step', 'linear',
 *  default 'linear')</pre>
 *
 <!-- options-end -->
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

    private String hiddenLayers = "";
    private String learningRule = "";
    private String activationFunction = "";
    private double learningRate;
    private double momentum;
    private int maxIteration;
    private double MSE;
    private boolean mlp;
    
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
        mlp = false;
        int[] layers = new int[2];
        layers[1] = data.numClasses();
        
        //convert any nominal to binary
        NominalToBinary filter = new NominalToBinary();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        layers[0] = data.numAttributes();
        
        if (!hiddenLayers.equals("n")){ //multi layer perceptron
            mlp = true;
            String[] splits = hiddenLayers.split(",");
            layers = new int[splits.length+layers.length];
            layers[0] = data.numAttributes();
            layers[layers.length-1] = data.numClasses();
            for (int i=0; i<splits.length; i++) {
                layers[i+1] = Integer.parseInt(splits[i]);
            }
        }
        
        //initialize layer, weight, and dw
        layer = new double [layers.length][];
        weight = new double [layers.length][][];
        dw = new double [layers.length][][];
        for (int i=0; i<layers.length; i++) {
            layer[i] = new double[layers[i]];
            if (i<layers.length-1) {
                weight[i] = new double[layers[i]][layers[i+1]];
                dw[i] = new double[layers[i]][layers[i+1]];
            }
        }
    }
    
    @Override
    public Enumeration<Option> listOptions() {
    
        Vector newVector = new Vector(7);

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
                  "\tMSE value as threshold for training termination.\n"
                  +"\t(Default = 0.25).",
                  "E", 1,"-E <MSE>"));
        newVector.addElement(new Option(
                  "\tThe hidden layers to be created for the network.\n"
                  + "\t(Value should be a list of comma separated Natural \n"
                  + "\tnumbers or the letter 'n' = no hidden layer, \n"
                  + "\tdefault 'n') \n",
                  "H", 1, "-H <comma seperated numbers for nodes on each layer>"));
        newVector.addElement(new Option(
                  "\tThe learning rule for single perceptron.\n"
                  + "\t(Value should be 'ptr', 'batch', 'delta', \n"
                  + "\tdefault 'ptr') \n",
                  "R", 1, "-R <learning rule>"));
        newVector.addElement(new Option(
                  "\tThe activation function for network.\n"
                  + "\t(Value should be 'sigmoid', 'sign', 'step', 'linear', \n"
                  + "\tdefault 'linear') \n",
                  "F", 1, "-F <activation function>"));

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
        String MSEString = Utils.getOption('E', options);
        if (MSEString.length() != 0) {
          MSE = new Double(MSEString);
        } else {
          MSE = 0.25;
        }
        String hiddenLayersString = Utils.getOption('H', options);
        if (hiddenLayersString.length() != 0) {
          this.hiddenLayers = hiddenLayersString;
        } else {
          this.hiddenLayers = "n";
        }
        String learningRuleString = Utils.getOption('R', options);
        if (learningRuleString.length() != 0) {
          learningRule = learningRuleString;
        } else {
          learningRule = "ptr";
        }
        String activationFunctionString = Utils.getOption('F', options);
        if (activationFunctionString.length() != 0) {
          activationFunction = activationFunctionString;
        } else {
          activationFunction = "linear";
        }

        Utils.checkForRemainingOptions(options);
    }
    
    @Override
    public String[] getOptions() {

        String[] options = new String[14];
        int current = 0;
        options[current++] = "-L"; options[current++] = "" + learningRate; 
        options[current++] = "-M"; options[current++] = "" + momentum;
        options[current++] = "-N"; options[current++] = "" + maxIteration;
        options[current++] = "-E"; options[current++] = "" + MSE; 
        options[current++] = "-H"; options[current++] = hiddenLayers;
        options[current++] = "-R"; options[current++] = learningRule;
        options[current++] = "-F"; options[current++] = activationFunction;

        while (current < options.length) {
          options[current++] = "";
        }
        return options;
    }
    
    public static double sigmoid(double x) {return 1 / (1 + Math.exp(-x));}
    
    public static double sign(double x) {return Math.signum(x);}
    
    public static double linear(double x) {return x;}
    
    public static double step(double x) {return x;}
}
