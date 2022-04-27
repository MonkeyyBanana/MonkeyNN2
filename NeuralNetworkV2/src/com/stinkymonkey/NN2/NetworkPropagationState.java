package com.stinkymonkey.NN2;

public class NetworkPropagationState {
	public float loss;
	public float[][] weights, biases, recurrWeights, weightMems, biasMems, recurrWeightMems, buf, recurrBuf, state;
	public float[] inMem;
	public NetworkDerivativeMemory deriMemory;
	
	public void Sync(NeuralNetwork src, NetworkContext cont, NetworkWContext fcont, NetworkDerivativeMemory deriMem) {
		state = new float[src.hidLayer.length][];
		weights = new float[src.hidLayer.length + 1][];
		biases = new float[src.hidLayer.length + 1][];
		buf = new float[src.hidLayer.length + 1][];
		recurrBuf = new float[src.hidLayer.length][];
		biasMems = deriMem.biasMems;
		weightMems = deriMem.weightMems;
		recurrWeightMems = deriMem.recurrWeightMems;
		recurrWeights = new float[src.hidLayer.length][];
		deriMemory = deriMem;
		
		for (int i = 0; i < src.hidLayer.length; i++) {
			state[i] = new float[src.hidLayer[i].numNeurons];
			weights[i] = src.hidConnection[i].weights;
			biases[i] = src.hidLayer[i].bias;
			
			if (i == 0)
				buf[i] = cont.inData;
			else
				buf[i] = fcont.hidBuffer[i - 1];
			
			if (src.hidLayer[i].reCurr) {
				recurrWeights[i] = src.hidRecurrConnection[i].weights;
				recurrBuf[i] = fcont.hidRecurrBuffer[i];
			}
		}
		
		int leng = src.hidLayer.length;
		weights[leng] = src.outConnection.weights;
		biases[leng] = src.outLayer.bias;
		
		if (leng > 0) 
			buf[leng] = fcont.hidBuffer[leng - 1];
		else
			buf[leng] = cont.inData;
	}
	
	public void Reset() {
		loss = 0.0f;
		for (int i = 0; i < buf.length; i++) {
			Util.Fill(buf[i], 0.0f);
			if (i < recurrBuf.length && recurrBuf[i] != null)
				Util.Fill(recurrBuf[i], 0.0f);
		}
	}
}
