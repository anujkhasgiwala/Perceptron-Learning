package com.usu.perceptron;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.Utils;

public class InputParser {
	
	static Map<Integer, Integer> actualOutput = new HashMap<Integer, Integer>();
	static double learningRate = 0.001;
	List<Double> errors = new ArrayList<>();
	double deltaWeight = 0;
	
	public static void main(String[] args) throws Exception {
		Map<Integer, Perceptron> perceptronInputs = new HashMap<Integer, Perceptron>();
		new InputParser().readInputs(perceptronInputs);
		
		new InputParser().logicalGates();
	}
	
	public void readInputs(Map<Integer, Perceptron> perceptronInputs) throws Exception{
		BufferedReader in = new BufferedReader(new FileReader("files/Data.arff"));
		
		String[] tmpOptions = new String[2];
		String classname;
		tmpOptions = Utils.splitOptions("weka.classifiers.trees.J48");
		classname      = tmpOptions[0];
		tmpOptions[0]  = "";
		Classifier cls = (Classifier) Utils.forName(Classifier.class, classname, tmpOptions);
		//fold validation
		Instances data = new Instances(in);
		data.setClassIndex(data.numAttributes() - 1);
		Random random = new Random();
		Instances randData = new Instances(data);
	    randData.randomize(random);
	    randData.stratify(10);
	    
	    Evaluation evalAll = new Evaluation(randData);
	    
	    for(int i = 0; i < 10; i++) {
	    	Evaluation eval = new Evaluation(randData);
	        Instances train = randData.trainCV(10, i);
	        Instances test = randData.testCV(10, i);
	        Classifier clsCopy = Classifier.makeCopy(cls);
	        clsCopy.buildClassifier(train);
	        eval.evaluateModel(clsCopy, test);
	        evalAll.evaluateModel(clsCopy, test);
	    }
	    System.out.println(evalAll.toSummaryString("=== " + 10 + "-fold Cross-validation ===", false));
	    
	    in = new BufferedReader(new FileReader("files/Data.txt"));
	    String line;
		Perceptron iv;
		int key = 0;
		while ((line = in.readLine()) != null) {
			if(!line.contains("X1")) {
				iv = new Perceptron();
				String[] tokens = line.split("\t");
				iv.setX1(Double.parseDouble(tokens[0]));
				iv.setX2(Double.parseDouble(tokens[1]));
				iv.setTargetOutput(Integer.parseInt(tokens[2]));
				
				perceptronInputs.put(key++, iv);
			}
		}
	    
		in.close();	
		
		calculateAll(perceptronInputs, "output.csv");
	}
	
	public int calculateOutput(Map<Integer, Perceptron> perceptronInputs, int key) {
		Perceptron perceptron = perceptronInputs.get(key);
		
		double sum = perceptron.getW0() + perceptron.getX1() * perceptron.getW1() + perceptron.getX2() * perceptron.getW2();
		
		return (sum > perceptron.getThreshold()) ? 1 : 0;
	}
	
	public double calculateError(Map<Integer, Perceptron> perceptronInputs) {
		double error = 0.0;
		for (Integer key : perceptronInputs.keySet()) {
			error += actualOutput.get(key) - perceptronInputs.get(key).getTargetOutput();
			errors.add(error);
		}
		return error;
	}
	
	public void calculateDeltaWeight(Map<Integer, Perceptron> perceptronInputs, double error) {
		for (Integer key : perceptronInputs.keySet()) {
			deltaWeight = learningRate * perceptronInputs.get(key).getX1() * error;
			perceptronInputs.get(key).setW1(perceptronInputs.get(key).getW1() + deltaWeight);
			deltaWeight = learningRate * perceptronInputs.get(key).getX2() * error;
			perceptronInputs.get(key).setW2(perceptronInputs.get(key).getW2() + deltaWeight);
		}
	}
	
	public void calculateAll(Map<Integer, Perceptron> perceptronInputs, String fileName) throws Exception{
		double error = 0.0;
		
		for (Integer key1 : perceptronInputs.keySet()) {
			actualOutput.put(key1, calculateOutput(perceptronInputs, key1));
		}
		error = calculateError(perceptronInputs);
		
		for(int i = 0; i < 1000; i++) {
			calculateDeltaWeight(perceptronInputs, error);
		}
		
		BufferedWriter writer  = new BufferedWriter(new FileWriter("files/"+fileName));
		writer.write("epoch");
		writer.write(",");
		writer.write("error");
		writer.newLine();
		for(int e = 0; e < errors.size(); e++) {
			writer.write(Integer.toString(e));
			writer.write(",");
			writer.write(Double.toString(errors.get(e)));
			writer.newLine();
		}
		writer.close();
	}
	
	public void logicalGates() throws Exception{
		int inputs[] = {0,0,0,1,1,0,1,1};
		int andOutputs[] = {0,0,0,1};
		int orOutputs[] = {0,1,1,1};
		int nandOutputs[] = {1,1,1,0};
		int norOutputs[] = {1,0,0,0};
		int xorOutputs[] = {0,1,1,0};
		
		logicalGatesCall(inputs, andOutputs, "andOutput.csv");
		logicalGatesCall(inputs, orOutputs, "orOutput.csv");
		logicalGatesCall(inputs, nandOutputs, "nandOutput.csv");
		logicalGatesCall(inputs, norOutputs, "norOutput.csv");
		logicalGatesCall(inputs, xorOutputs, "xorOutput.csv");
	}
	
	public void logicalGatesCall(int inputs[], int outputs[], String fileName) throws Exception {
		actualOutput = new HashMap<Integer, Integer>();
		errors = new ArrayList<>();
		deltaWeight = 0;
		
		Map<Integer, Perceptron> perceptronInputs = new HashMap<Integer, Perceptron>();
		Perceptron iv;
		int i = 0;
		while (i<4) {
			iv = new Perceptron();
			iv.setX1(inputs[2*i]);
			iv.setX2(inputs[2*i+1]);
			iv.setTargetOutput(outputs[i]);

			perceptronInputs.put(i++, iv);
		}
		calculateAll(perceptronInputs, fileName);
	}
}
