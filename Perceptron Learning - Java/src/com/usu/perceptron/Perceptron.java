package com.usu.perceptron;

import java.util.Random;

public class Perceptron {
	double x1, x2;
	int targetOutput;
	double w0, w1, w2;
	double threshold = 0.5;
	
	public double getX1() {
		return x1;
	}
	
	public void setX1(double x1) {
		this.x1 = x1;
	}
	
	public double getX2() {
		return x2;
	}
	
	public void setX2(double x2) {
		this.x2 = x2;
	}
	
	public int getTargetOutput() {
		return targetOutput;
	}
	
	public void setTargetOutput(int correctOutput) {
		this.targetOutput = correctOutput;
	}
	
	public Perceptron() {
		w0 = -0.2 + (0.2 - (-0.2)) * new Random().nextDouble();
		w1 = -0.2 + (0.2 - (-0.2)) * new Random().nextDouble();
		w2 = -0.2 + (0.2 - (-0.2)) * new Random().nextDouble();
	}

	public double getW1() {
		return w1;
	}

	public void setW1(double w1) {
		this.w1 = w1;
	}

	public double getW2() {
		return w2;
	}

	public void setW2(double w2) {
		this.w2 = w2;
	}

	public double getW0() {
		return w0;
	}
	
	public double getThreshold() {
		return threshold;
	}
}
