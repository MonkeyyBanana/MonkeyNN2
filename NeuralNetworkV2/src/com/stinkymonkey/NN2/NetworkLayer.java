package com.stinkymonkey.NN2;

public class NetworkLayer {
	public static float MIN_BIAS = 0.0f;
	public static float MAX_BIAS = 0.0f;
	
	public int numNeurons;
	public boolean reCurr;
	public float[] bias;
	
	public NeuronActivator actFunc;
	
	public NetworkLayer(int numNeuron, boolean recurr, NeuronActivator activator) {
		numNeurons = numNeuron;
		reCurr = recurr;
		actFunc = activator;
		bias = new float[numNeurons];
	}
	
	// Randomize biases according to MIN and MAX biases NOT PRESET
	public void RandomizeBiases() {
		for (int i = 0; i < numNeurons; i++)
			bias[i] = Util.randFloat() * (MAX_BIAS - MIN_BIAS) + MIN_BIAS;
	}
	
	// Copy biases from other layer
	public void copyBias(NetworkLayer src) {
		float[] basis = src.bias;
		for (int i = 0; i < numNeurons; i++)
			bias[i] = basis[i];
	}
	
	// Randomly change biases using the chance parameter 0 ~ 1
	public void Mutate(float chance) {
		for (int i = 0; i < numNeurons; i++)
			if (Util.randFloat() <= chance)
				bias[i] = Util.randFloat() * (MAX_BIAS - MIN_BIAS) + MIN_BIAS;
	}
	
	// Randomly breed with another layer with same selections
	public void Breed(NetworkLayer src) {
		for (int i = 0; i < numNeurons; i++) {
			float chance = Util.randFloat();
			bias[i] = bias[i] * chance + src.bias[i] * (1.0f - chance);
		}
	}
}
