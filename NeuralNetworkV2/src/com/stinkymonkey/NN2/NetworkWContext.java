package com.stinkymonkey.NN2;

public class NetworkWContext {
	public float[][] hidBuffer, hidRecurrBuffer;
	
	public void Sync(NeuralNetwork src) {
		hidBuffer = new float[src.hidLayer.length][];
		hidRecurrBuffer = new float[src.hidLayer.length][];
		for (int i = 0; i < src.hidLayer.length; i++) {
			hidBuffer[i] = new float[src.hidLayer[i].numNeurons];
			if (src.hidLayer[i].reCurr)
				hidRecurrBuffer[i] = new float[src.hidLayer[i].numNeurons];
		}
	}
}
