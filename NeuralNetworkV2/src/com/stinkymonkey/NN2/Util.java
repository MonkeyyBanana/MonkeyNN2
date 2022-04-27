package com.stinkymonkey.NN2;

import java.util.Random;

public class Util {
	private static Random rand = new Random(1);
	
	public static void setSeed(int x) {
		rand = new Random(x);
	}
	
	public static float randFloat() {
		return (float) rand.nextFloat();
	}
	
	public static int randInt() {
		return (int) rand.nextInt();
	}
	
	public static float Lerp(float a, float b, float c) {
		return a + (b - a) * c;
	}
	
	public static void Fill(float[] f, float x) {
		for (int i = 0; i < f.length; i++)
			f[i] = x;
	}
	
	public static void Multiply(float[] f, float x) {
		for (int i = 0; i < f.length; i++)
			f[i] *= x;
	}
	
	public static int Largest(float[] f, int start, int end) {
		int out = start;
		int index = start + 1;
		while (index < end) {
			if (f[out] < f[index])
				out = index;
			index++;
		}
		return out;
	}
	
	public static void Copy(float[] src, float[] to, int length) {
		for (int i = 0; i < length; i++)
			to[i] = src[i];
	}
}