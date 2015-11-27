package MachineLearning.MyANN;

import java.util.Enumeration;
import java.util.Random;
import java.util.Scanner;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;

public class App {
    public Instances data = null;
    public MyANN model = new MyANN();

    public void loadData(String filename) throws Exception{
        data = DataSource.read(filename);
    }

    public void setClassAttribute(int index) {
        data.setClassIndex(index);
    }

    public void buildClassifier() throws Exception {
        model.buildClassifier(data);
    }
	
    public void testModelGivenDatatest(String datatestFile){
        Instances datatest;
        try {
            datatest = DataSource.read(datatestFile);
            Evaluation eval = new Evaluation(data);
            eval.evaluateModel(model, datatest);
            System.out.println(eval.toSummaryString(
                    "\nResults\n======\n", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
	
    public void crossValidation(){
        try {
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(model, data, 10, new Random(1));
            System.out.println(eval.toSummaryString(
                    "\nResults\n======\n", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
	
    public void percentageSplit(double percentage){
        try {
            int trainSize = (int) Math.round(
                    data.numInstances() * percentage / 100); 
            int testSize = data.numInstances() - trainSize; 
            data.randomize(new Random(1));
            Instances train = new Instances(data, 0, trainSize); 
            Instances test = new Instances(data, trainSize, testSize);
            model.buildClassifier(train);

            Evaluation eval = new Evaluation(train);
            eval.evaluateModel(model, test);
            System.out.println(eval.toSummaryString(
                    "\nResults\n======\n", false));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
	
    public void saveModel(String filename) throws Exception {
        SerializationHelper.write(filename + ".model", model);
    }
	
    public void loadModel(String modelFile) throws Exception{
        model = (MyANN) SerializationHelper.read(modelFile);
    }
	
    public void classify(String unlabeledFile, String output) throws Exception{
        if (model == null){
            System.out.println("model is null");
            return;
        }
        Instances unlabeled = DataSource.read(unlabeledFile);
        unlabeled.setClassIndex(unlabeled.numAttributes() - 1);
        
        Instances labeled = new Instances(unlabeled); // create copy
        
        for (int i = 0; i < unlabeled.numInstances(); i++) {
            double clsLabel = model.classifyInstance(unlabeled.instance(i));
            labeled.instance(i).setClassValue(clsLabel);
        }
        
        DataSink.write(output, labeled);
    }

    public static void main(String[] args) throws Exception {
        App myweka = new App();
        Scanner input = new Scanner(System.in);
        System.out.println("Welcome to MyWEKA\n"
            + "1. Load Data\n"
            + "2. Load Model");
        String cmdString = input.nextLine();

        switch (cmdString) {
            case "1": 
                System.out.println("Input filename: ");
                String file = input.nextLine();
                myweka.loadData(file);
                
                System.out.println("Set class index: ");
                cmdString = input.nextLine();
                myweka.setClassAttribute(Integer.parseInt(cmdString));
                
                System.out.println("Enter parameters for MyANN? (y/n): ");
                cmdString = input.nextLine();
                if (cmdString.equals("y")){
                    for (Enumeration<Option> e = myweka.model.listOptions(); 
                            e.hasMoreElements();) {
                        Option option = e.nextElement();
                        System.out.println(option.synopsis()
                            + "\n" + option.description());
                    }
                    cmdString = input.nextLine();
                    myweka.model.setOptions(cmdString.split("\\s+"));
                }
                System.out.println("\nParameters are:");
                String[] options = myweka.model.getOptions();
                for (String option: options)
                    System.out.print(option + " ");
                
                System.out.println("\nBuilding MyANN model...");
                myweka.buildClassifier();
                
                System.out.println("Save model? (y/n): ");
                cmdString = input.nextLine();
                if (cmdString.equals("y")){
                    System.out.println("Input filename: ");
                    cmdString = input.nextLine();
                    myweka.saveModel(cmdString);
                    System.out.println("Saved.");
                }
                
                System.out.println("Evaluate model by:\n"
                    + "1. 10-fold Cross Validation\n"
                    + "2. Percentage Split");
                cmdString = input.nextLine();
                if (cmdString.equals("1")) {
                    myweka.crossValidation();
                } else if (cmdString.equals("2")) {
                    System.out.println("Enter percentage:");
                    Double percent = input.nextDouble();
                    myweka.percentageSplit(percent);
                }
                break;
            case "2": 
                System.out.println("Input filename: ");
                cmdString = input.nextLine();
                myweka.loadModel(cmdString);
                
                System.out.println("Classify unlabeled data? (y/n): ");
                if (!cmdString.equals("y"))
                    break;
                
                System.out.println("Input filename: ");
                cmdString = input.nextLine();
                
                System.out.println("Output filename: ");
                String out = input.nextLine();
                myweka.classify(cmdString, out);
                System.out.println("Finished. Labeled data => " + out);
                break;
            default:
                break;
        }
    }
}
