package com.stinkymonkey.NN2;

public class NetworkDerivativeMemory {
	public float[][] weightMems, biasMems, recurrWeightMems, outFullConnectedWeightMems, recurrBPBuffer, altRecurrBPBuffer;
	
	public void Sync(NeuralNetwork src) {
		biasMems = new float[src.hidLayer.length + 1][];
		weightMems = new float[src.hidLayer.length + 1][];
		recurrWeightMems = new float[src.hidLayer.length + 1][];
		recurrBPBuffer = new float[src.hidLayer.length + 1][];
		altRecurrBPBuffer = new float[src.hidLayer.length + 1][];
		
		for (int i = 0; i < src.hidLayer.length; i++) {
			weightMems[i] = new float[src.hidConnection[i].numSynapses];
			biasMems[i] = new float[src.hidLayer[i].numNeurons];
			
			if (src.hidLayer[i].reCurr) {
				recurrWeightMems[i] = new float[src.hidRecurrConnection[i].numSynapses];
				recurrBPBuffer[i] = new float[src.hidLayer[i].numNeurons];
				altRecurrBPBuffer[i] = new float[src.hidLayer[i].numNeurons];
			}
		}
		
		int leng = src.hidLayer.length;
		biasMems[leng] = new float[src.outLayer.numNeurons];
		weightMems[leng] = new float[src.outConnection.numSynapses];
	}
	
	public void SwapBPBuffers() {
		float[][] temp = recurrBPBuffer;
		recurrBPBuffer = altRecurrBPBuffer;
		altRecurrBPBuffer = temp;
	}
	
	public void Reset() {
		for (int i = 0; i < biasMems.length; i++) {
			Util.Fill(biasMems[i], 0.0f);
			Util.Fill(weightMems[i], 0.0f);
			
			if (i < recurrWeightMems.length && recurrWeightMems[i] != null) {
				Util.Fill(recurrWeightMems[i], 0.0f);
                Util.Fill(recurrBPBuffer[i], 0.0f);
                Util.Fill(altRecurrBPBuffer[i], 0.0f);
			}
			if (outFullConnectedWeightMems != null && i < outFullConnectedWeightMems.length)
				Util.Fill(outFullConnectedWeightMems[i], 0.0f);
		}
	}
	
	public void Scale(float x) {
		for (int i = 0; i < biasMems.length; i++) {
			Util.Multiply(biasMems[i], x);
			Util.Multiply(weightMems[i], x);
			
			if (i < recurrWeightMems.length && recurrWeightMems[i] != null)
				Util.Multiply(recurrWeightMems[i], x);
			
			if (outFullConnectedWeightMems != null && i < outFullConnectedWeightMems.length)
				Util.Multiply(outFullConnectedWeightMems[i], x);
		}
	}
	
	public void ResetBuffer()
	{
		 for (int i = 0; i < biasMems.length; i++) {
             if (i < recurrWeightMems.length && recurrWeightMems[i] != null) {
                 Util.Fill(recurrBPBuffer[i], 0.0f);
                 Util.Fill(altRecurrBPBuffer[i], 0.0f);
             }
         }
	}
}
