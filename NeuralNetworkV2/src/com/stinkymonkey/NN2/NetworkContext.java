package com.stinkymonkey.NN2;

public class NetworkContext {
	public float[] inData;
	public float[] outData;
	public float[] hidData;
	public float[][] hidRecurrData;
	
	public void Sync(NeuralNetwork src) {
		src.syncArrays(inData, outData, hidData, hidRecurrData);
		Reset(true);
	}
	
	// Reset in and out arrays
	public void Reset(boolean resetInOut) {
		if (resetInOut) {
			Util.Fill(inData, 0.0f);
			Util.Fill(outData, 0.0f);
		}
		Util.Fill(hidData, 0.0f);
		for (int i = 0; i < hidRecurrData.length; i++)
			if (hidRecurrData[i] != null)
				Util.Fill(hidRecurrData[i], 0.0f);
	}
}
