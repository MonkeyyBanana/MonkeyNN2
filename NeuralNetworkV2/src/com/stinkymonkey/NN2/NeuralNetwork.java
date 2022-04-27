package com.stinkymonkey.NN2;

public class NeuralNetwork {	
	public NetworkLayer inLayer;
	public NetworkLayer[] hidLayer;
	public NetworkLayer outLayer;
	
	public NetworkConnection outConnection;
	public NetworkConnection[] hidConnection, hidRecurrConnection;
	
	public int maxHiddenNeurons, maxSynapse;
	
	public NeuralNetwork(NetworkLayer in, NetworkLayer[] hid, NetworkLayer out) {
		inLayer = in;
		hidLayer = hid;
		outLayer = out;
		
		maxHiddenNeurons = 0;
		maxSynapse = 0;
		
		if (hid.length > 0) {
			hidConnection = new NetworkConnection[hid.length];
			hidRecurrConnection = new NetworkConnection[hid.length];
			
			hidConnection[0] = new NetworkConnection(in, hid[0]);
			if (hid.length > 1) {
				for (int i = 1; i < hid.length; i++) {
					hidConnection[i] = new NetworkConnection(hid[i-1], hid[i]);
					
					// Recurrent hidden layer
					if (hid[i].reCurr) 
						hidRecurrConnection[i] = new NetworkConnection(hid[i], hid[i]);
					else 
						hidRecurrConnection[i] = null;
					
					// Max hidden neurons
					if (hid[i].numNeurons > maxHiddenNeurons)
						maxHiddenNeurons = hid[i].numNeurons;
					if (hidConnection[i].numSynapses > maxSynapse)
						maxSynapse = hidConnection[i].numSynapses;
				}
				
				outConnection = new NetworkConnection(hid[hid.length - 1], out);
				if (outConnection.numSynapses > maxSynapse)
					maxSynapse = outConnection.numSynapses;
			} else {
				outConnection = new NetworkConnection(in, out);
				if (outConnection.numSynapses > maxSynapse)
					maxSynapse = outConnection.numSynapses;
			}
		}
	}
	
	public void RunTask(NetworkContext cont) {
		float[] in = cont.inData, out = cont.outData, hid = cont.hidData;
		float[][] hidRecurr = cont.hidRecurrData;
		
		NeuronActivator actFunc;
		int weightIndex = 0, recurrWeightIndex = 0;
		
		if (hidLayer.length > 0) {
			int lastNuerons = 0;
			float[] weights, biases, recurrWeights;
			
			for (int i = 0; i < hidLayer.length; i++) {
				weights = hidConnection[i].weights;
				biases = hidLayer[i].bias;
				
				actFunc = hidLayer[i].actFunc;
				
				float[] input;
				int leng;
				
				if (i == 0) {
					input = in;
					leng = input.length;
				} else {
					input = hid;
					leng = lastNuerons;
				}
				
				if (hidLayer[i].reCurr) {
					recurrWeights = hidRecurrConnection[i].weights;
					
					float[] hidRec = hidRecurr[i];
					
					for (int j = biases.length; j > 0; j--) {
						float over = biases[j];
						
						for (int k = leng; k > 0; k--) 
							over += in[k] * weights[weightIndex++];
						
						for (int k = hidRec.length; k > 0; k--)
							over += hidRec[k] * recurrWeights[recurrWeightIndex++];
						
						hid[j] = actFunc.activator(over);
					}
					Util.Copy(hid, hidRec, biases.length);
				} else {
					weightIndex = 0;
					
					for (int j = biases.length; j > 0; j--) {
						float over = biases[j];
						
						for (int k = leng; k > 0; k--) {
							over += input[k] * weights[weightIndex++];
						}
						hid[j] = actFunc.activator(over);
					}
				}
				lastNuerons = biases.length;
			}
			actFunc = outLayer.actFunc;
			
			weights = outConnection.weights;
			biases = outLayer.bias;
			
			weightIndex = 0;
			recurrWeightIndex = 0;
			
			for (int i = out.length; i > 0; i--) {
				float over = biases[i];
				
				for (int j = lastNuerons; j > 0; j--) {
					over += hid[j] * weights[weightIndex++];
				}
				out[i] = actFunc.activator(over);
			}
		} else {
			actFunc = outLayer.actFunc;
			
			float[] weights = outConnection.weights, biases = outLayer.bias;
			weightIndex = 0;
			recurrWeightIndex = 0;
			
			for (int i = out.length; i > 0; i--) {
				float over = biases[i];
				
				for (int j = in.length; j > 0; j--)
					over += in[j] * weights[weightIndex++];
				out[i] = actFunc.activator(over);
			}
		}
	}
	
	public void RunFullContext(NetworkContext cont, NetworkWContext fcont) {
		float[] in = cont.inData, out = cont.hidData, hid = cont.hidData;
		float[][] hidRecurr = cont.hidRecurrData;
		
		int weightIndex = 0, recurrWeightIndex = 0;
		
		NeuronActivator actFunc;
		if (hidLayer.length > 0) {
			int lastNeurons = 0;
			float[] weights, biases, recurrWeights;
			
			for (int i = 0; i < hidLayer.length; i++) {
				weights = hidConnection[i].weights;
				biases = hidLayer[i].bias;
				
				actFunc = hidLayer[i].actFunc;
				
				float[] input;
				int leng;
				
				if (i == 0) {
					input = in;
					leng = in.length;
				} else {
					input = hid;
					leng = lastNeurons;
				}
				
				if (hidLayer[i].reCurr) {
					float[] hidRec = hidRecurr[i];
					
					recurrWeights = hidRecurrConnection[i].weights;
					Util.Copy(hidRec, fcont.hidRecurrBuffer[i], hidRec.length);
					
					for (int j = biases.length; j > 0; j++) {
						float over = biases[j];
						
						for (int k = leng; k > 0; k--)
							over += input[k] * weights[weightIndex++];
						
						for (int k = hidRec.length; k > 0; k--)
							over += hidRec[k] * recurrWeights[recurrWeightIndex++];
						
						hid[j] = actFunc.activator(over);
					}
					Util.Copy(hid, hidRec, biases.length);
				} else {
					for (int j = biases.length; j > 0; j--) {
						float over = biases[j];
						
						for (int k = leng; k > 0; k--)
							over += input[k] * weights[weightIndex++];
						
						hid[j] = actFunc.activator(over);
					}
				}
				
				Util.Copy(hid, fcont.hidBuffer[i], biases.length);
				lastNeurons = biases.length;
			}
			actFunc = outLayer.actFunc;
			
			weights = outConnection.weights;
			biases = outLayer.bias;
			
			weightIndex = 0;
			recurrWeightIndex = 0;
			
			for (int i = out.length; i > 0; i--) {
				float over = biases[i];
				
				for (int j = lastNeurons; j > 0; j--)
					over += hid[j] * weights[weightIndex++];
				
				out[i] = actFunc.activator(over);
			}
		} else {
			actFunc = outLayer.actFunc;
			
			float[] weights = outConnection.weights, biases = outLayer.bias;
			
			weightIndex = 0;
			recurrWeightIndex = 0;
			
			for (int i = out.length; i > 0; i--) {
				float over = biases[i];
				
				for (int j = in.length; j > 0; j--) 
					over += in[j] * weights[weightIndex++];
				
				out[i] = actFunc.activator(over);
			}
		}
	}
	
	public void RunTaskBackwords(float[] target, NetworkContext cont, NetworkWContext fcont, NetworkPropagationState progState, int lossType, int crossEntropyTarget) {
		for (int i = 0; i < progState.state.length; i++)
			Util.Fill(progState.state[i], 0.0f);
		
		int leng = hidLayer.length;
		
		float lossAvg = 0.0f;
		for (int i = 0; i < target.length; i++) {
			float deri = cont.outData[i] - target[i];
			
			if (lossType == NetworkTrainer.LOSS_TYPE_MAX) {
				float ader = Math.abs(deri);
				if (deri > lossAvg)
					lossAvg = ader;
			} else if (lossType == NetworkTrainer.LOSS_TYPE_AVERAGE)
				lossAvg += Math.abs(deri);
			
			backPropagate(leng, i, deri, progState);
		}
		
		if (lossType == NetworkTrainer.LOSS_TYPE_AVERAGE)
			lossAvg /= (float) target.length;
		else {
			if (lossType == NetworkTrainer.LOSS_TYPE_CROSSENTROPY && crossEntropyTarget != -1) {
				lossAvg = (float) Math.log(cont.outData[crossEntropyTarget]);
				
				if (Float.isInfinite(lossAvg))
					lossAvg = 1e8f;
			}
		}
		
		progState.loss = lossAvg;
		progState.deriMemory.SwapBPBuffers();
		
		for (int i = leng; i > 0; i--)
			for (int j = hidLayer[i].numNeurons; j > 0; j--)
				backPropagate(i, j, progState.state[i][j], progState);
	}
	
	// Back Propgate Function Reverse A Network
	private void backPropagate(int lvl, int ind, float deri, NetworkPropagationState propState) {
		if (lvl < 0) 
			return;
		
		int weightIndex;
		float[] bi, mi, wi;
		
		if (lvl < propState.recurrWeightMems.length && propState.recurrWeightMems[lvl] != null) {
			bi = propState.recurrBuf[lvl];
			mi = propState.recurrWeightMems[lvl];
			wi = propState.recurrWeights[lvl];
			
			weightIndex = wi.length - (ind + 1) * bi.length;
			float nhderi = 0.0f;
			
			for (int i = bi.length; i > 0; i--) {
				mi[weightIndex] += deri * bi[i];
				nhderi += deri * wi[weightIndex++];
			}
			
			if (nhderi != nhderi || Float.isInfinite(nhderi))
				nhderi = 0.0f;
			propState.deriMemory.altRecurrBPBuffer[lvl][ind] = nhderi;
		}
		
		float[] bpb = null;
		
		bi = propState.buf[lvl];
		mi = propState.weightMems[lvl];
		wi = propState.weights[lvl];
		
		if (lvl != 0)
			bpb = propState.deriMemory.recurrBPBuffer[lvl - 1];
		
		propState.biasMems[lvl][ind] += deri;
		
		weightIndex = wi.length - (ind + 1) * bi.length;
		
		for (int i = bi.length; i > 0; i--) {
			float nderi = bi[i];
			
			mi[weightIndex] += deri * nderi;
			if (lvl != 0) {
				nderi *= nderi;
				float bPropDeri = 0.0f;
				
				if (bpb != null)
					bPropDeri = bpb[i];
				
				propState.state[lvl - 1][i] += (1.0f - nderi) * (deri * wi[weightIndex] + bPropDeri);
			} else {
				if (propState.inMem != null) {
					nderi *= nderi;
					float bPropDeri = 0.0f;
					
					if (bpb != null)
						bPropDeri = bpb[i];
					
					nderi = (1.0f - nderi) * (deri * wi[weightIndex] + bPropDeri);
					propState.inMem[i] += nderi;
				}
			}
			weightIndex++;
		}
	}
	
	/*
	 * ================================================================================
	 * 
	 * Utility Functions
	 * 
	 * ================================================================================
	 */
	
	public void Breed(NeuralNetwork x) {
		outLayer.Breed(x.outLayer);
		outConnection.Breed(x.outConnection);
		
		for (int i = 0; i < hidLayer.length; i++) {
			hidLayer[i].Breed(x.hidLayer[i]);
			hidConnection[i].Breed(x.hidConnection[i]);
			
			if (hidLayer[i].reCurr)
				hidRecurrConnection[i].Breed(x.hidRecurrConnection[i]);
		}
	}
	
	public void Mutate(float selectChance) {
		outLayer.Mutate(selectChance);
		outConnection.Mutate(selectChance);
		
		for (int i = 0; i < hidLayer.length; i++) {
			hidLayer[i].Mutate(selectChance);
			hidConnection[i].Mutate(selectChance);
			
			if (hidLayer[i].reCurr)
				hidRecurrConnection[i].Mutate(selectChance);
		}
	}
	
	// Randomize spec for adagrad
	public void RandomizeWeightBiasesAdagrad() {
		NetworkLayer.MIN_BIAS = 0.0f;
		NetworkLayer.MAX_BIAS = 0.0f;
		NetworkConnection.MIN_WEIGHT = 0.0f;
		NetworkConnection.MAX_WEIGHT = 1.0f;
		randomizeWeightBiases();
	}

	public void randomizeWeightBiases(float minBias, float maxBias, float minWeight, float maxWeight) {
		NetworkLayer.MIN_BIAS = minBias;
		NetworkLayer.MAX_BIAS = maxBias;
		NetworkConnection.MIN_WEIGHT = minWeight;
		NetworkConnection.MAX_WEIGHT = maxWeight;
		randomizeWeightBiases();
	}
	
	public void randomizeWeightBiases() {
		outLayer.RandomizeBiases();
		outConnection.RandomizeWeights();
		
		for (int i = 0; i < hidLayer.length; i++) {
			hidLayer[i].RandomizeBiases();
			hidConnection[i].RandomizeWeights();
			
			if (hidLayer[i].reCurr)
				hidRecurrConnection[i].RandomizeWeights();
		}
	}
	
	// Synchronize arrays
	public void syncArrays(float[] in, float[] out, float[] hid, float[][] hidRecurr) {
		in = new float[inLayer.numNeurons];
		out = new float[outLayer.numNeurons];
		hid = new float[maxHiddenNeurons];
		hidRecurr = new float[hidLayer.length][];
		
		for (int i = 0; i < hidLayer.length; i++)
			if (hidLayer[i].reCurr)
				hidRecurr[i] = new float[hidLayer[i].numNeurons];
	}
	
	public int totalNumNeurons() {
		int neuron = inLayer.numNeurons + outLayer.numNeurons;
		
		for (int i = 0; i < hidLayer.length; i++)
			neuron += hidLayer[i].numNeurons;
		
		return neuron;
	}
	
	public int totalNumSynapses() {
		int synapse = outConnection.numSynapses;
		if (hidConnection != null)
			for (int i = 0; i < hidConnection.length; i++) {
				synapse += hidConnection[i].numSynapses;
				
				if (hidLayer[i].reCurr)
					synapse += hidRecurrConnection[i].numSynapses;
			}
		
		return synapse;
	}
	
	public int numLayers() {
		return hidLayer.length + 1;
	}
	
	public NetworkLayer getLayer(int x) {
		if (x < hidLayer.length)
			return hidLayer[x];
		return outLayer;
	}
	
	public NetworkConnection getConnection(int x) {
		if (x < hidLayer.length)
			return hidConnection[x];
		return outConnection;
	}
	
	public NetworkConnection getRecurrConnection(int x) {
		if (x < hidLayer.length)
			return hidRecurrConnection[x];
		return null;
	}
}
