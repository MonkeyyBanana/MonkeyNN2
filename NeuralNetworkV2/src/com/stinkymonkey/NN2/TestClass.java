package com.stinkymonkey.NN2;

public class TestClass {
	public static void main(String[] args) {
		/*
		NeuronActivator lol = NeuronActivator.Sqrt;
		float a = 0.23456f;
		System.out.println(lol.getName() + " " + lol.acivator(a));
		*/
		float[] lol = new float[] {1.0f, 2.0f, 3.0f};
		System.out.println(toString(lol));
		TestArray(lol, 0.0f);
		System.out.println(toString(lol));
	}
	
	private static void TestArray(float[] fill, float x) {
		for (int i = 0; i < fill.length; i++)
			fill[i] = x;
	}
	
	private static String toString(float[] x) {
		String out = "";
		for (int i = 0; i < x.length; i++) {
			out += x[i] + " ";
		}
		return out;
	}
}
