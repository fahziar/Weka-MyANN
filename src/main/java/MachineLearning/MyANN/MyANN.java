package MachineLearning.MyANN;

import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

/**
 * Created by fahziar on 23/11/2015.
 */
public class MyANN extends Classifier
        implements OptionHandler {
    // layer[nomer layer][nomer neuron] = nilai pada neuron tsb
    private double [][] layer;
    // weight[nomer layer][nomer neuron  pada layer][nomer neuron tujuan]
    // = nilai pada neuron tsb
    private double [][][] weight;
    // delta weight
    // dw[nomer layer][nomer neuron pada layer][nomer neuron tujuan]
    // = delta weight
    private double [][][] dw;

    private String hiddenLayers = "n";
    private String learningRule = "ptr";
    private String activationFunction = "linear";
    private double learningRate = 0.2;
    private double momentum = 0.3;
    private int maxIteration = 500;
    private double MSE = 0.01;
    private boolean mlp = false;
    private String weightOption = "n";
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        getCapabilities().testWithFail(data);
        init(data);
        
    	int NInstance = data.numInstances();
    	double[] target = new double[NInstance];
    	double[] output = new double[NInstance];
    	double E;
        boolean run = true;
    	int it = 0;
    	do {
            E = 0.0;
            printWeights();
            for (int i = 0; i < NInstance; i++) {
                Instance instance = data.instance(i);
                //initialize layer (instance attributes -> layer[0] values)
                setInputLayer(instance);
                //forward propagate
                forwardPropagation();
                //save output, save targets
                target[i] = instance.classValue();
                output[i] = getOutput();
                if (!mlp){
                    if (!learningRule.equals("batch")){
                        gradientDescentUpdateDw(target[i]);
                        applyDw();
                        resetDw();
                    }
                    //count cumulative E
                    E += Math.pow(target[i] - output[i], 2) / 2;
                }
            }
            if (mlp){
                E = backPropagation(target);
    		applyDw();
                resetDw();
            }
            if (learningRule.equals("batch")) {
                for (int k = 0; k < data.numAttributes(); k++) {
                    double sumDelta = 0.0;
                    for (int j = 0; j < data.numInstances(); j++) {
                        sumDelta += data.instance(j).value(k)
                                * (target[j] - output[j]);
                    }
                    for (int l=0; l<weight[0][k].length; l++){
                        weight[0][k][l] += learningRate * sumDelta;
                    }
                }
            }
            if (learningRule.equals("ptr")) {
                run = (Double.compare(E, 0.0) != 0); 
            } else {
                run = (Double.compare(E, MSE) > 0);
            }
            System.out.println("\nError: " + E);
            it++;
    	} while (run && it < maxIteration);
    }
    
    private void init(Instances data) throws Exception {
        int[] layers = new int[2];
        layers[1] = data.numClasses();
        
        Normalize filter1 = new Normalize();
        filter1.setInputFormat(data);
        data = Filter.useFilter(data, filter1);
        NominalToBinary filter = new NominalToBinary();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        layers[0] = data.numAttributes();
        
        if (!hiddenLayers.equals("n")){ //multi layer perceptron
            mlp = true;
            String[] splits = hiddenLayers.split(",");
            layers = new int[splits.length+layers.length];
            layers[0] = data.numAttributes() + 1;
            layers[layers.length-1] = data.numClasses();
            for (int i=0; i<splits.length; i++) {
                layers[i+1] = Integer.parseInt(splits[i]);
            }
        }
        
        //initialize layer, weight, and dw
        layer = new double [layers.length][];
        weight = new double [layers.length-1][][];
        dw = new double [layers.length-1][][];
        for (int i=0; i<layers.length; i++) {
            layer[i] = new double[layers[i]];
            if (i<layers.length-1) {
                weight[i] = new double[layers[i]][];
                dw[i] = new double[layers[i]][];
                for (int j=0; j<layers[i]; j++) {
                    weight[i][j] = new double[layers[i+1]];
                    dw[i][j] = new double[layers[i+1]];
                }
            }
        }
        
        //initialize weight and value in input layer
        double w = 0;
        if (weightOption.equals("n")){
            //random weight
            Random r = new Random();
            w = 0 + (1 - 0) * r.nextDouble();
        } else {
            w = Double.parseDouble(weightOption);
        }
        for (int i=0; i<layer[0].length; i++){
            for (int j=0; j<weight[0][i].length; j++){
                weight[0][i][j] = w;
            }
        }
    }

    // Untuk single layer
    // gradient descent batch
    // untuk delta rule tinggal panggil ini setiap iterasi pada data,
    // panggil applyDw kemudian resetDw()
    private void gradientDescentUpdateDw(double target){
        for (int i=0; i<dw[0].length; i++){
            for (int j=0; j<dw[0][i].length; j++){
                dw[0][i][j] = dw[0][i][j] + learningRate
                        * (target - layer[1][0]) * layer[0][i];
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
    
    /**
     * Sebelum memanggil prosedur ini, pastikan layer dan weight 
     * sudah terinisialisasi
     */
    private void forwardPropagation () {
    	int nLayer = this.layer.length;
    	for (int i = 0; i < nLayer-1; i++) {
    		int nUnit = this.layer[i].length;
    		int nNextUnit = this.weight[i][0].length;
    		double[] tmpOutput = new double[nNextUnit];
    		for (double d : tmpOutput) {
    			d = 0.0;
    		}
    		for (int j = 0; j < nUnit; j++) {
    			for (int k = 0; k < nNextUnit; k++) {
    				tmpOutput[k] += this.layer[i][j]
                                        * this.weight[i][j][k];
    			}
    		}
    		switch (this.activationFunction) {
    		case "sigmoid" : 
    			for (int l = 0; l < nNextUnit; l++) {
    				this.layer[i+1][l] = sigmoid(tmpOutput[l]);
    			}
    			break;
    		case "sign" :
    			for (int l = 0; l < nNextUnit; l++) {
    				this.layer[i+1][l] = sign(tmpOutput[l]);
    			}
    			break;
    		case "step" :
    			for (int l = 0; l < nNextUnit; l++) {
    				this.layer[i+1][l] = step(tmpOutput[l]);
    			}
    			break;
    		default :
    			for (int l = 0; l < nNextUnit; l++) {
    				this.layer[i+1][l] = linear(tmpOutput[l]);
    			}
    			break;
    		}
    	}
    }

    private double backPropagation(double[] target){
        //Reset delta weight
        resetDw();

        //Array yang berisi error dari tiap-tiap unit
        double [][] deltas = new double[layer.length][];
        for (int i=0; i<deltas.length; i++){
            deltas[i] = new double[layer[i].length];
        }

        //Output layer adalah layer terluar
        //hitung error untuk output layer
        double E = 0.0;
        for (int i=0; i<layer[layer.length - 1].length; i++){
            double o = layer[layer.length - 1][i];
            double t = target[i];
            deltas[layer.length - 1][i] = o * (1 - o) * (t - o);
            E += deltas[layer.length - 1][i];
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
                    dw[i][j][k] = learningRate * layer[i][j] * deltas[i][j]
                            + momentum*dw[i][j][k];
                }
            }
        }
        return E;
    }
    
    private double getOutput() {
    	int outputLayer = layer.length - 1;
        int idxMax = Utils.maxIndex(layer[outputLayer]);
    	return layer[outputLayer][idxMax];
    }
    
    private void setInputLayer(Instance instance) {
        layer[0][1] = 1.0;
    	for (int i = 1; i < instance.numAttributes(); i++) {
    		layer[0][i] = instance.value(i);
    	}
    }
    
    @Override
    public double classifyInstance(Instance instance) {
        setInputLayer(instance);        
        forwardPropagation();
        return getOutput();
    }
    
    private void printWeights() {
    	System.out.println("Weights: ");
    	for (int i = 0; i < layer.length - 1; i++) {
    		for (int j = 0; j < layer[i].length; j++) {
    			for (int k = 0; k < layer[i+1].length; k++) {
    				System.out.print(weight[i][j][k] + " ");
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
    public Enumeration<Option> listOptions() {
        Vector newVector = new Vector(8);

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
                  + "\t(Value should be 'ptr', 'batch', 'incremental', \n"
                  + "\tdefault 'ptr') \n",
                  "R", 1, "-R <learning rule>"));
        newVector.addElement(new Option(
                  "\tThe activation function for network.\n"
                  + "\t(Value should be 'sigmoid', 'sign', 'step', 'linear', \n"
                  + "\tdefault 'linear') \n",
                  "F", 1, "-F <activation function>"));
        newVector.addElement(new Option(
                  "\tThe weight initialization for network.\n"
                  + "\t(Value should be double or 'n',\n"
                  + "\tdefault 'n' = random) \n",
                  "W", 1, "-W <weight>"));

        return newVector.elements();
    }
    
    @Override
    public void setOptions(String[] options) throws Exception {
        String learningString = Utils.getOption('L', options);
        if (learningString.length() != 0) {
          learningRate = new Double(learningString);
        }
        String momentumString = Utils.getOption('M', options);
        if (momentumString.length() != 0) {
          momentum = new Double(momentumString);
        }
        String epochsString = Utils.getOption('N', options);
        if (epochsString.length() != 0) {
          maxIteration = Integer.parseInt(epochsString);
        }
        String MSEString = Utils.getOption('E', options);
        if (MSEString.length() != 0) {
          MSE = new Double(MSEString);
        }
        String hiddenLayersString = Utils.getOption('H', options);
        if (hiddenLayersString.length() != 0) {
          this.hiddenLayers = hiddenLayersString;
        }
        String learningRuleString = Utils.getOption('R', options);
        if (learningRuleString.length() != 0) {
          learningRule = learningRuleString;
        }
        String activationFunctionString = Utils.getOption('F', options);
        if (activationFunctionString.length() != 0) {
          activationFunction = activationFunctionString;
        }
        String weightString = Utils.getOption('W', options);
        if (weightString.length() != 0) {
          weightOption = weightString;
        }

        Utils.checkForRemainingOptions(options);
    }
    
    @Override
    public String[] getOptions() {
        String[] options = new String[16];
        int current = 0;
        options[current++] = "-L"; options[current++] = "" + learningRate; 
        options[current++] = "-M"; options[current++] = "" + momentum;
        options[current++] = "-N"; options[current++] = "" + maxIteration;
        options[current++] = "-E"; options[current++] = "" + MSE; 
        options[current++] = "-H"; options[current++] = hiddenLayers;
        options[current++] = "-R"; options[current++] = learningRule;
        options[current++] = "-F"; options[current++] = activationFunction;
        options[current++] = "-W"; options[current++] = weightOption;

        while (current < options.length) {
          options[current++] = "";
        }
        return options;
    }
    
    public static double sigmoid(double x) {return 1 / (1 + Math.exp(-x));}
    
    public static double sign(double x) {return Math.signum(x);}
    
    public static double linear(double x) {return x;}
    
    public static double step(double x) {
        if (x>0)
            return 1;
        else
            return 0;
    }

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