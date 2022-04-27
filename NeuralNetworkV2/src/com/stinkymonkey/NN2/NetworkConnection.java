package com.stinkymonkey.NN2;

public class NetworkConnection {
	public static float MIN_WEIGHT = 0.0f;
	public static float MAX_WEIGHT = 0.0f;
	
	// Synpase = Connections	
	public int numSynapses;
	public float[] weights;
	
	public NetworkConnection(NetworkLayer in, NetworkLayer out) {
		numSynapses = in.numNeurons * out.numNeurons;
		weights = new float[numSynapses];
	}
	
	// Randomize Weights MAX and MIN weights NOT PRESET
	public void RandomizeWeights() {
		for (int i = 0; i < numSynapses; i++)
			weights[i] = Util.randFloat() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
	}
	
	// Copy from other connection
	public void CopyWeights(NetworkConnection src) {
		float[] basis = src.weights;
		for (int i = 0; i < numSynapses; i++)
			weights[i] = basis[i];
	}
	
	// Mutate randomly weights according to chance 0 ~ 1
	public void Mutate(float chance) {
		for (int i = 0; i < numSynapses; i++)
			if (Util.randFloat() <= chance)
				weights[i] = Util.randFloat() * (MAX_WEIGHT - MIN_WEIGHT) + MIN_WEIGHT;
	}
	
	// Breed with another Connection
	public void Breed(NetworkConnection src) {
		for (int i = 0; i < numSynapses; i++) {
			float chance = Util.randFloat();
			weights[i] = weights[i] * chance + src.weights[i] * (1.0f - chance);
		}
	}
}
